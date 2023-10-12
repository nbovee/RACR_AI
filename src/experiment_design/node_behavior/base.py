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


# TODO: NodeServices should absorb the functionality from Runner


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


class BaseRunner:
    """
    At the heart of all nodes (participant or observer) is a component responsible for guiding
    the node through its required sequence of tasks. It is the job of this component to:
        1.) manage the state of the node via its `status` property 
        2.) initiate handshakes with its partner nodes (nodes it must communicate with during
            the experiment)
        3.) send tasks to another node's inbox (a threadsafe PriorityQueue)

    These are the main capabilities handled by this base class, which is not itself used in the 
    testbed. Instead, two subclasses of BaseRunner extend its functionality to cover lots of 
    the capabilities required by observer nodes and participant nodes. These subclasses are
    defined below.

    The base class for observers is called `BaseDelegator`, adding the ability to wait for all
    participants to become ready before sending out a predefined list of tasks for each applicable
    node and sending each participant a start signal.

    The base class for participants is called `BaseExecutor`, adding the ability to map each
    possible type of task it receives to a corresponding method that defines how the task will 
    be handled.
    """

    node: NodeService
    active_connections: dict[str, NodeService]
    status: str
    partners: list[str]

    def __init__(self, node: NodeService):
        self.node = node
        self.active_connections = self.node.active_connections
        self.status = self.node.status

        self.get_ready()

    def get_ready(self):
        self.status = "ready"

    def start(self):
        self.status = "running"

    def on_finish(self):
        self.status = "finished"


class BaseExecutor(BaseRunner):
    """
    The executor guides the participating node through its required sequence of tasks. It works by
    pulling tasks from the node's inbox (a threadsafe PriorityQueue) and processing each task 
    according to the corresponding method outlined in the `task_map` attribute. The user can
    customize the behavior of a participating node by overwriting the methods that correspond to 
    each task that may be given to the node.

    This should hopefully provide a straightforward interface for us to extend the testbed's 
    functionality later on by adding new task types. The end user is able to do so as well.
    """

    node: ParticipantService
    inbox: PriorityQueue
    model: WrappedModel | None
    task_map: dict[type, Callable]

    def __init__(self, node: ParticipantService):
        super().__init__(node)

    def get_ready(self):
        self.task_map = {
            tasks.SimpleInferenceTask: self.simple_inference,
            tasks.SingleInputInferenceTask: self.inference_sequence_per_input,
            tasks.InferOverDatasetTask: self.infer_dataset,
            tasks.FinishSignalTask: self.on_finish
        }
        self.model = self.node.model
        self.handshake()
        self.status = "ready"

    def process(self, task: tasks.Task):
        task_class = type(task)
        corresponding_method = self.task_map[task_class]
        corresponding_method(task)

    def start(self):
        super().start()
        if self.inbox is not None:
            while self.status == "running":
                current_task = self.inbox.get()
                self.process(current_task)

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
            downstream_node_svc = self.node.get_connection(task.downstream_node)
            assert isinstance(downstream_node_svc, ParticipantService)
            downstream_task = tasks.SimpleInferenceTask(
                self.node.node_name, out, inference_id=inference_id, start_layer=task.end_layer
            )
            downstream_node_svc.give_task(downstream_task)

    def inference_sequence_per_input(self, task: tasks.SingleInputInferenceTask):
        """
        If you want to use a partitioner or conduct multiple inferences per input, this is where
        you'd implement that behavior, most likely using the provided self.simple_inference method,
        possibly with start_layer and end_layer being determined with a partitioner.
        """
        raise NotImplementedError(
            f"inference_sequence_per_input not implemented for {self.node.node_name} Executor"
        )

    def infer_dataset(self, task: tasks.InferOverDatasetTask):
        """
        Run the self.inference_sequence_per_input method for each element in the dataset.
        """
        dataset_module, dataset_instance = task.dataset_module, task.dataset_instance
        observer_svc = self.node.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        dataset = observer_svc.get_dataset_reference(dataset_module, dataset_instance)
        dataloader = DataLoader(dataset, batch_size=1)

        for input in dataloader:
            subtask = tasks.SingleInputInferenceTask(input, from_node="SELF")
            self.inference_sequence_per_input(subtask)


class BaseDelegator(BaseRunner):
    """
    The delegator waits for its partners to set their status to "ready" as a part of its own
    setup sequence, then sends a predetermined set of tasks for each partner before sending a
    start signal to each.
    """

    node: ObserverService
    playbook: dict[str, list[tasks.Task]]

    def __init__(self):
        return

    def link_node(self, node: ObserverService):
        self.node = node
        self.active_connections = self.node.active_connections
        self.status = self.node.status
        self.partners = self.node.partners

    def set_playbook(self, playbook: dict[str, list[tasks.Task]]):
        self.playbook = playbook

    def get_ready(self):
        self.handshake()
        self.await_participants()
        self.send_playbook()
        self.status = "ready"

    def await_participants(self, n_attempts: int = 10):
        success = False
        while n_attempts > 0:
            if all([(self.node.get_connection(p).get_status() == "ready") for p in self.partners]):
                success = True
                break
            n_attempts -= 1
            sleep(1)

        if not success:
            straglers = [
                p for p in self.partners
                if self.node.get_connection(p).get_status() != "ready"
            ]
            raise AwaitParticipantException(f"Observer had to wait too long for nodes {straglers}")

    def send_playbook(self):
        for partner, tasklist in self.playbook.items():
            for task in tasklist:
                pnode = self.node.get_connection(partner) 
                assert isinstance(pnode, ParticipantService)
                pnode.give_task(task)

    def start(self):
        for partner in self.partners:
            pnode = self.node.get_connection(partner)
            assert isinstance(pnode, ParticipantService)
            pnode.run()
        self.status = "waiting"
        while any([(self.node.get_connection(p).get_status() != "finished" for p in self.partners)]):
            sleep(5)
        self.on_finish()

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

    def __init__(self, partners: list[str]):
        super().__init__()
        self.status = "initializing"
        self.node_name = self.ALIASES[0].upper().strip()
        if self.node_name.upper().strip() != "OBSERVER":
            self.logger = logging.getLogger(f"{self.node_name}_logger")
        self.active_connections = {}
        self.partners = partners

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

    def set_partners(self, partners: list[str]) -> None:
        self.partners = partners

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

    def __init__(self, partners: list[str]):
        super().__init__(partners)
        self.logger = logging.getLogger("main_logger")
        self.master_dict = MasterDict()
        self.handshake(n_attempts=10, wait_increase_factor=1.25)

    def prepare_delegator(self, delegator: BaseDelegator):
        """
        Makes sure the delegator is ready to officially start the experiment. Note that the 
        delegator must already have a playbook and list of partners, set using `set_playbook` and
        `set_partners`.
        """
        self.delegator = delegator
        self.delegator.link_node(self)
        self.delegator.get_ready()

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
    def send_log(self):
        pass

    @rpyc.exposed
    def run(self):
        assert self.status == "ready"
        self.delegator.start()


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

    executor: BaseExecutor
    model: WrappedModel
    inbox: PriorityQueue[tasks.Task] = PriorityQueue()

    def __init__(self,
                 ModelCls: Type[nn.Module] | None,
                 RunnerCls: Type[BaseExecutor]
                 ):
        super().__init__()
        self.ModelCls = ModelCls
        self.RunnerCls = RunnerCls

    def prepare_model(self):
        observer_svc = self.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        master_dict = observer_svc.get_master_dict()
        if not isinstance(self.ModelCls, nn.Module):
            self.model = WrappedModel(master_dict=master_dict)
        else:
            self.model = WrappedModel(pretrained=self.ModelCls(), master_dict=master_dict)

    def prepare_runner(self, RunnerCls):
        self.runner = RunnerCls(self)

    @rpyc.exposed
    def give_task(self, task: tasks.Task):
        self.inbox.put(task)

    @rpyc.exposed
    def run(self):
        assert self.status == "ready"
        self.executor.start()
