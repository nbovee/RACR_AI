from __future__ import annotations
import logging
import logging
import uuid
import rpyc
import torch.nn as nn
from typing import Type
from queue import PriorityQueue
from importlib import import_module
from rpyc.core.protocol import Connection
from time import sleep
from typing import Callable
from torch.utils.data import DataLoader
from rpyc.utils.factory import DiscoveryError

import src.experiment_design.tasks.tasks as tasks
from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataset import BaseDataset
from src.experiment_design.records.master_dict import MasterDict


class HandshakeFailureException(Exception):
    """
    Raised if a node fails to establish a handshake with any of its specified partners.
    """
    def __init__(self, message):
        super().__init__(message)


class AwaitParticipantException(Exception):
    """
    Raised if the observer node waits too long for a participant node to change its status to 
    "ready".
    """
    def __init__(self, message):
        super().__init__(message)


@rpyc.service
class NodeService(rpyc.Service):
    """
    Implements all the endpoints common to both participant and observer nodes.
    """
    ALIASES: list[str]

    active_connections: dict[str, NodeService]
    node_name: str
    status: str
    logger: logging.Logger
    partners: list[str]

    def __init__(self):
        super().__init__()
        self.status = "initializing"
        self.node_name = self.ALIASES[0].upper().strip()
        self.active_connections = {}

    def on_connect(self, conn: Connection):
        if isinstance(conn.root, NodeService):
            self.logger.info(
                f"on_connect method called; memoizing connection to {conn.root.get_node_name()}"
            )
            self.active_connections[conn.root.get_node_name()] = conn.root

    def on_disconnect(self, conn: Connection):
        self.logger.info("on_disconnect method called; removing saved connection.")
        for name in self.active_connections:
            if self.active_connections[name] == conn:
                self.active_connections.pop(name)
                self.logger.info(f"Removed connection to {name}")

    def get_connection(self, node_name: str) -> NodeService:
        node_name = node_name.upper().strip()
        self.logger.debug(f"Attempting to get connection to {node_name}.")
        if node_name in self.active_connections:
            self.logger.debug(f"Connection to {node_name} already memoized.")
            return self.active_connections[node_name]
        self.logger.debug(
            f"Connection to {node_name} not memoized; attempting to access via registry."
        )
        conn = rpyc.connect_by_service(node_name, service=self)  # type: ignore
        self.active_connections[node_name] = conn.root
        self.logger.info(f"New connection to {node_name} established and memoized.")
        return self.active_connections[node_name]

    @rpyc.exposed
    def handshake(self, n_attempts: int = 10, wait_increase_factor: int | float = 1):
        partners = self.partners.copy()
        success = False
        wait_time = 1
        while n_attempts > 0:
            for i in range(len(partners)-1, 0, -1):
                try:
                    self.get_connection(partners[i])
                    partners.pop(i)
                except DiscoveryError:
                    continue
            if not len(partners):
                success = True
                break
            else:
                n_attempts -= 1
                wait_time *= wait_increase_factor
            sleep(wait_time)

        if not success:
            raise HandshakeFailureException(f"Node {self.node_name} failed to handshake with {partners}")

    @rpyc.exposed
    def get_status(self) -> str:
        self.logger.debug(f"get_status exposed method called; returning '{self.status}'")
        return self.status

    @rpyc.exposed
    def get_node_name(self) -> str:
        self.logger.debug(f"get_node_name exposed method called; returning '{self.node_name}'")
        return self.node_name


@rpyc.service
class ObserverService(NodeService):
    """
    The service exposed by the observer device during experiments.
    """

    ALIASES: list[str] = ["OBSERVER"]

    master_dict: MasterDict
    playbook: dict[str, list[tasks.Task]]

    def __init__(self,
                 partners: list[str],
                 playbook: dict[str, list[tasks.Task]]
                 ):
        super().__init__()
        self.partners = partners
        self.logger = logging.getLogger("main_logger")
        self.master_dict = MasterDict()
        self.playbook = playbook

    @rpyc.exposed
    def await_participants(self, n_attempts: int = 10):
        success = False
        while n_attempts > 0:
            if all([(self.get_connection(p).get_status() == "ready") for p in self.partners]):
                success = True
                break
            n_attempts -= 1
            sleep(1)

        if not success:
            straglers = [
                p for p in self.partners
                if self.get_connection(p).get_status() != "ready"
            ]
            raise AwaitParticipantException(f"Observer had to wait too long for nodes {straglers}")

    @rpyc.exposed
    def delegate(self):
        for partner, tasklist in self.playbook.items():
            pnode = self.get_connection(partner) 
            assert isinstance(pnode, ParticipantService)
            for task in tasklist:
                pnode.give_task(task)

    @rpyc.exposed
    def get_master_dict(self) -> MasterDict:
        return self.master_dict

    @rpyc.exposed
    def get_dataset_reference(self, dataset_module: str, dataset_instance: str) -> BaseDataset:
        """
        Allows remote nodes to access datasets stored on the observer as if they were local objects.
        """
        module = import_module(f"src.experiment_design.datasets.{dataset_module}")
        dataset = getattr(module, dataset_instance)
        return dataset

    @rpyc.exposed
    def run(self):
        assert self.status == "ready"
        for partner in self.partners:
            pnode = self.get_connection(partner)
            assert isinstance(pnode, ParticipantService)
            pnode.run()
        self.status = "waiting"
        while any([(self.get_connection(p).get_status() != "finished" for p in self.partners)]):
            sleep(5)
        self.on_finish()

    def on_finish(self):
        self.status = "finished"


@rpyc.service
class ParticipantService(NodeService):
    """
    The service exposed by all participating nodes regardless of their node role in the test case.
    A test case is defined by mapping node roles to physical devices available on the local
    network, and a node role is defined by passing a model and runner to the ZeroDeployedServer
    instance used to deploy this service. This service expects certain methods to be available
    for each.
    """

    ALIASES = ["PARTICIPANT"]

    model: WrappedModel
    inbox: PriorityQueue[tasks.Task] = PriorityQueue()
    task_map: dict[type, Callable]

    def __init__(self,
                 ModelCls: Type[nn.Module] | None,
                 ):
        super().__init__()
        self.logger = logging.getLogger(f"{self.node_name}_logger")
        self.ModelCls = ModelCls
        self.task_map = {
            tasks.SimpleInferenceTask: self.simple_inference,
            tasks.SingleInputInferenceTask: self.inference_sequence_per_input,
            tasks.InferOverDatasetTask: self.infer_dataset,
            tasks.FinishSignalTask: self.on_finish
        }

    @rpyc.exposed
    def prepare_model(self):
        observer_svc = self.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        master_dict = observer_svc.get_master_dict()
        if not isinstance(self.ModelCls, nn.Module):
            self.model = WrappedModel(master_dict=master_dict)
        else:
            self.model = WrappedModel(pretrained=self.ModelCls(), master_dict=master_dict)

    @rpyc.exposed
    def give_task(self, task: tasks.Task):
        self.inbox.put(task)

    @rpyc.exposed
    def run(self):
        assert self.status == "ready"
        self.status = "running"
        if self.inbox is not None:
            while self.status == "running":
                current_task = self.inbox.get()
                self.process(current_task)

    @rpyc.exposed
    def get_ready(self):
        self.handshake()
        self.status = "ready"

    def process(self, task: tasks.Task):
        task_class = type(task)
        corresponding_method = self.task_map[task_class]
        corresponding_method(task)

    def on_finish(self):
        assert self.inbox.empty()
        self.status = "finished"

    def simple_inference(self, task: tasks.SimpleInferenceTask):
        assert self.model is not None
        inference_id = task.inference_id if task.inference_id is not None else str(uuid.uuid4())
        out = self.model.forward(
            task.input, inference_id=inference_id, start=task.start_layer, end=task.end_layer
        )
 
        if task.downstream_node is not None and isinstance(task.end_layer, int):
            downstream_node_svc = self.get_connection(task.downstream_node)
            assert isinstance(downstream_node_svc, ParticipantService)
            downstream_task = tasks.SimpleInferenceTask(
                self.node_name, out, inference_id=inference_id, start_layer=task.end_layer
            )
            downstream_node_svc.give_task(downstream_task)

    def inference_sequence_per_input(self, task: tasks.SingleInputInferenceTask):
        """
        If you want to use a partitioner or conduct multiple inferences per input, this is where
        you'd implement that behavior, most likely using the provided self.simple_inference method,
        possibly with start_layer and end_layer being determined with a partitioner.
        """
        raise NotImplementedError(
            f"inference_sequence_per_input not implemented for {self.node_name} Executor"
        )

    def infer_dataset(self, task: tasks.InferOverDatasetTask):
        """
        Run the self.inference_sequence_per_input method for each element in the dataset.
        """
        dataset_module, dataset_instance = task.dataset_module, task.dataset_instance
        observer_svc = self.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        dataset = observer_svc.get_dataset_reference(dataset_module, dataset_instance)
        dataloader = DataLoader(dataset, batch_size=1)

        for input in dataloader:
            subtask = tasks.SingleInputInferenceTask(input, from_node="SELF")
            self.inference_sequence_per_input(subtask)
