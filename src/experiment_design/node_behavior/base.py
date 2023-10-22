from __future__ import annotations
import atexit
import logging
import logging
import sys
import threading
import uuid
import rpyc
import asyncio
import rpyc.core.protocol
from rpyc.utils.classic import obtain
import torch.nn as nn
from typing import Any, Type, Union
from queue import PriorityQueue
from importlib import import_module
from rpyc.core.protocol import Connection, PingError
from time import sleep
from typing import Callable
from rpyc.utils.factory import DiscoveryError

import src.experiment_design.tasks.tasks as tasks
from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataset import BaseDataset
from src.experiment_design.records.master_dict import MasterDict


logger = logging.getLogger("tracr_logger")

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True

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

    active_connections: dict[str, Any]
    node_name: str
    status: str
    partners: list[str]
    classname: str = "NodeService"

    def __init__(self):
        super().__init__()
        self.status = "initializing"
        self.node_name = self.ALIASES[0].upper().strip()
        self.active_connections = {}

    def class_object_reference(self):
        return getattr(sys.modules[__name__], self.classname)

    def on_connect(self, conn: Connection):
        assert conn.root is not None
        conn._config["sync_request_timeout"] = 60

        try:
            node_name = conn.root.get_node_name()
        except AttributeError:
            # must be the VoidService exposed by the Experiment object, not another NodeService
            node_name = "APP.PY"

        self.active_connections[node_name] = conn
        return

    def on_disconnect(self, conn: Connection):
        logger.info("on_disconnect method called; removing saved connection.")
        closed = []
        active_conns = self.active_connections.copy()
        for name in active_conns:
            c = active_conns[name]
            if c.closed:
                closed.append(name)
                logger.debug(f"Connection to {name} is closed")
                continue
        for dead_connection in closed:
            try:
                self.active_connections.pop(dead_connection)
                logger.info(f"Removed connection to {dead_connection} (connection closed)")
            except KeyError:
                pass

    def get_connection(self, node_name: str):
        node_name = node_name.upper().strip()
        if node_name in self.active_connections:
            logger.debug(f"using saved connection to {node_name}")
            return self.active_connections[node_name]
        logger.debug(f"attempting to connect to {node_name} via registry.")
        conn = rpyc.connect_by_service(
            node_name, service=self, config=rpyc.core.protocol.DEFAULT_CONFIG  # type: ignore
        )
        conn._config["sync_request_timeout"] = 60
        self.active_connections[node_name] = conn
        logger.info(f"new connection to {node_name} established and saved.")
        return self.active_connections[node_name]

    def handshake(self):
        logger.info(f"{self.node_name} starting handshake with partners {str(self.partners)}")
        for p in self.partners:
            for _ in range(3):
                try:
                    logger.debug(f"{self.node_name} attempting to connect to {p}")
                    self.get_connection(p)
                    break
                except DiscoveryError:
                    sleep(1)
                    continue
        if all([(p in self.active_connections.keys()) for p in self.partners]):
            logger.info(f"Successful handshake with {str(self.partners)}")
        else:
            straglers = [p for p in self.partners if p not in self.active_connections.keys()]
            logger.info(f"Could not handshake with {str(straglers)}")

    @rpyc.exposed
    def get_ready(self):
        get_ready_thd = threading.Thread(target=self._get_ready, daemon=True)
        get_ready_thd.start()

    def _get_ready(self):
        self.handshake()
        self.status = "ready"

    @rpyc.exposed
    def run(self):
        run_thd = threading.Thread(target=self._run, daemon=True)
        run_thd.start()

    def _run(self):
        raise NotImplementedError

    @rpyc.exposed
    def get_status(self) -> str:
        logger.debug(f"get_status exposed method called; returning '{self.status}'")
        return self.status

    @rpyc.exposed
    def get_node_name(self) -> str:
        logger.debug(f"get_node_name exposed method called; returning '{self.node_name}'")
        return self.node_name


@rpyc.service
class ObserverService(NodeService):
    """
    The service exposed by the observer device during experiments.
    """

    ALIASES: list[str] = ["OBSERVER"]

    master_dict: MasterDict
    playbook: dict[str, list[tasks.Task]]
    classname: str = "ObserverService"

    def __init__(self,
                 partners: list[str],
                 playbook: dict[str, list[tasks.Task]]
                 ):
        super().__init__()
        self.partners = partners
        self.master_dict = MasterDict()
        self.playbook = playbook
        atexit.register(self.close_participants)
        logger.info("Finished initializing ObserverService object.")

    def delegate(self):
        logger.info(f"Delegating tasks.")
        for partner, tasklist in self.playbook.items():
            pnode = self.get_connection(partner) 
            for task in tasklist:
                logger.debug(f"Giving {str(type(task))} to {partner}")
                pnode.root.give_task(task)

    def _get_ready(self):
        logger.info("Making sure all participants are ready.")
        for partner in self.partners:
            node = self.get_connection(partner).root
            node.get_ready()

        success = False
        n_attempts = 10
        while n_attempts > 0:
            if all([(self.get_connection(p).root.get_status() == "ready") for p in self.partners]):
                success = True
                logger.info("All participants are ready!")
                break
            n_attempts -= 1
            sleep(1)

        if not success:
            straglers = [
                p for p in self.partners
                if self.get_connection(p).root.get_status() != "ready"
            ]
            raise AwaitParticipantException(f"Observer had to wait too long for nodes {straglers}")

        self.delegate()
        self.status = "ready"

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

    def _run(self):
        assert self.status == "ready"
        for p in self.partners:
            self.get_connection(p).root.run()
        self.status = "waiting"

        while True:
            if all([(self.get_connection(p).root.get_status() == "finished") for p in self.partners]):
                logger.info("All nodes have finished!")
                break
            sleep(5)

        self.on_finish()

    def on_finish(self):
        self.status = "finished"

    def close_participants(self):
        for p in self.partners:
            logger.info(f"sending self-destruct signal to {p}")
            try:
                node = self.get_connection(p).root
                node.self_destruct()
                logger.info(f"{p} self-destructed successfully")
            except (DiscoveryError, EOFError, TimeoutError):
                logger.info(f"{p} was already shut down")


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
    done_event: Union[threading.Event, None]
    high_priority_lock: threading.Condition = threading.Condition()
    mid_priority_lock: threading.Condition = threading.Condition()

    def __init__(self,
                 ModelCls: Union[Type[nn.Module], None],
                 ):
        super().__init__()
        self.ModelCls = ModelCls
        self.task_map = {
            tasks.SimpleInferenceTask: self.simple_inference,
            tasks.SingleInputInferenceTask: self.inference_sequence_per_input,
            tasks.InferOverDatasetTask: self.infer_dataset,
            tasks.FinishSignalTask: self.on_finish
        }

    @rpyc.exposed
    def prepare_model(self):
        logger.info("Preparing model.")
        observer_svc = self.get_connection("OBSERVER").root
        master_dict = observer_svc.get_master_dict()
        if not isinstance(self.ModelCls, nn.Module):
            self.model = WrappedModel(master_dict=master_dict)
        else:
            self.model = WrappedModel(pretrained=self.ModelCls(), master_dict=master_dict)

    @rpyc.exposed
    def give_task(self, task: tasks.Task):
        logger.info(f"Receiving {task.__class__}.")
        accept_task_thd = threading.Thread(target=self._accept_task, args=[task], daemon=True)
        accept_task_thd.start()
        return

    def _accept_task(self, task):
        logger.info(f"Accepting {task.__class__} to inbox in thread.")
        task = obtain(task)
        self.inbox.put(task)

    def _run(self):
        assert self.status == "ready"
        self.status = "running"
        if self.inbox is not None:
            while self.status == "running":
                current_task = self.inbox.get()
                self.process(current_task)

    def _get_ready(self):
        logger.info("Getting ready.")
        self.handshake()
        self.prepare_model()
        self.status = "ready"

    @rpyc.exposed
    def self_destruct(self):
        """
        Sets a threading.Event object to let the zerodeploy remote script know it's time to 
        close the server and remove the tempdir from the remote machine's filesystem.
        """
        assert self.done_event is not None
        self.done_event.set()

    def link_done_event(self, done_event: threading.Event):
        """
        Once the participant service has been deployed on the remote machine, it is given an 
        Event object to set once it's ready to self-destruct.
        """
        self.done_event = done_event

    def process(self, task):
        """
        It is important to note that the task will be a netref to the actual task, so we use
        the _rpyc_getattr method to figure out the true underlying type of the task.
        """
        task_class = task.__class__
        corresponding_method = self.task_map[task_class]
        corresponding_method(task)

    def on_finish(self, task):
        assert self.inbox.empty()
        self.status = "finished"

    def simple_inference(self, task: tasks.SimpleInferenceTask):
        assert self.model is not None
        inference_id = task.inference_id if task.inference_id is not None else str(uuid.uuid4())
        logger.info(f"Running simple inference on layers {str(task.start_layer)} through {str(task.end_layer)}")
        out = self.model(
            task.input, inference_id=inference_id, start=task.start_layer, end=task.end_layer
        )
 
        if task.downstream_node is not None and isinstance(task.end_layer, int):
            downstream_node_svc = self.get_connection(task.downstream_node).root
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
        observer_svc = self.get_connection("OBSERVER").root
        dataset = observer_svc.get_dataset_reference(dataset_module, dataset_instance)
        for idx in range(len(dataset)):
            input, _ = obtain(dataset[idx])
            subtask = tasks.SingleInputInferenceTask(input, from_node="SELF")
            self.inference_sequence_per_input(subtask)

