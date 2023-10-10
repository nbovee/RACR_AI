from __future__ import annotations
from time import sleep
import uuid
from queue import PriorityQueue
from typing import Callable
import rpyc
from torch import Tensor
from PIL import Image
from torch.utils.data import DataLoader
from rpyc.utils.factory import DiscoveryError

import tasks.tasks as tasks
from rpc_services.node_service import NodeService
from rpc_services.observer_service import ObserverService
from rpc_services.participant_service import ParticipantService
from models.model_hooked import WrappedModel


class HandshakeFailureException(Exception):
    """
    Raised if a node fails to establish a handshake with any of its specified partners.
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
        3.) send tasks to another node's inbox

    These are the main capabilities handled by this base class, which is not itself used in the 
    testbed. Instead, two subclasses of BaseRunner extend its functionality to cover lots of 
    the capabilities required by observer nodes and participant nodes.

    The base class for observers is called `BaseDelegator`, and the base class for participants 
    is called `BaseExecutor`.

    The runner guides the participating node through its required sequence of tasks. It works by
    pulling tasks from the node's inbox (a threadsafe PriorityQueue) and processing each task 
    according to the corresponding method outlined in the `task_map` attribute. The user can
    customize the behavior of a participating node by overwriting the methods that correspond to 
    each task that may be given to the node.

    This should hopefully provide a straightforward interface for us to extend the testbed's 
    functionality later on by adding new task types. The end user is able to do so as well.
    """

    node: NodeService
    inbox: PriorityQueue
    active_connections: dict[str, NodeService]
    model: WrappedModel | None
    status: str
    task_map: dict[type, Callable]
    partners: list[str]

    def __init__(self, node: NodeService):
        self.node = node
        self.active_connections = self.node.active_connections
        self.inbox = self.node.inbox
        self.status = self.node.status

        self.task_map = {
            tasks.SimpleInferenceTask: self.simple_inference,
            tasks.SingleInputInferenceTask: self.inference_sequence_per_input,
            tasks.InferOverDatasetTask: self.infer_dataset,
            tasks.FinishSignalTask: self.on_finish
        }

        self.get_ready()

    def get_ready(self):
        if isinstance(self.node, ParticipantService):
            self.model = self.node.model
        self.handshake()
        self.status = "ready"

    def handshake(self, n_attempts: int = 10):
        partners = self.partners.copy()
        success = False
        while n_attempts > 0:
            for i in range(len(partners), 0, -1):
                try:
                    self.node.get_connection(partners[i])
                    partners.pop(i)
                except DiscoveryError:
                    continue
            if not len(partners):
                success = True
                break
            else:
                n_attempts -= 1
            sleep(1)

        if not success:
            raise HandshakeFailureException(f"Node {self.node.node_name} failed to handshake with {partners}")

    def process(self, task: tasks.Task):
        task_class = type(task)
        corresponding_method = self.task_map[task_class]
        corresponding_method(task)

    def start(self):
        self.status = "running"
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
        out = self.model.forward(task.input, inference_id=inference_id, start=task.start_layer, end=task.end_layer)
 
        if task.downstream_node is not None and isinstance(task.end_layer, int):
            downstream_node_svc = self.node.get_connection(task.downstream_node)
            downstream_task = tasks.SimpleInferenceTask(self.node.node_name, out, inference_id=inference_id, start_layer=task.end_layer)
            downstream_node_svc.give_task(downstream_task)

    def inference_sequence_per_input(self, task: tasks.SingleInputInferenceTask):
        """
        If you want to use a partitioner or conduct multiple inferences per input, this is where
        you'd implement that behavior, most likely using the provided self.simple_inference method,
        possibly with start_layer and end_layer being determined with a partitioner.
        """
        raise NotImplementedError(f"inference_sequence_per_input not implemented for {self.node.node_name} runner")

    def infer_dataset(self, task: tasks.InferOverDatasetTask):
        """
        Run the self.inference_sequence_per_input method for each element in the dataset.
        """
        dataset_dirname, dataset_instance = task.dataset_dirname, task.dataset_instance
        observer_svc = self.node.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        dataset = observer_svc.get_dataset_reference(dataset_dirname, dataset_instance)
        dataloader = DataLoader(dataset, batch_size=1)

        for input in dataloader:
            subtask = tasks.SingleInputInferenceTask("SELF", input)
            self.inference_sequence_per_input(subtask)

class BaseParticipantRunner:
    """
    The runner guides the participating node through its required sequence of tasks. It works by
    pulling tasks from the node's inbox (a threadsafe PriorityQueue) and processing each task 
    according to the corresponding method outlined in the `task_map` attribute. The user can
    customize the behavior of a participating node by overwriting the methods that correspond to 
    each task that may be given to the node.

    This should hopefully provide a straightforward interface for us to extend the testbed's 
    functionality later on by adding new task types. The end user is able to do so as well.
    """

    node: NodeService
    inbox: PriorityQueue
    active_connections: dict[str, NodeService]
    model: WrappedModel | None
    status: str
    task_map: dict[type, Callable]
    partners: list[str]

    def __init__(self, node: NodeService):
        self.node = node
        self.active_connections = self.node.active_connections
        self.inbox = self.node.inbox
        self.status = self.node.status

        self.task_map = {
            tasks.SimpleInferenceTask: self.simple_inference,
            tasks.SingleInputInferenceTask: self.inference_sequence_per_input,
            tasks.InferOverDatasetTask: self.infer_dataset,
            tasks.FinishSignalTask: self.on_finish
        }

        self.get_ready()

    def get_ready(self):
        if isinstance(self.node, ParticipantService):
            self.model = self.node.model
        self.handshake()
        self.status = "ready"

    def handshake(self, n_attempts: int = 10):
        partners = self.partners.copy()
        success = False
        while n_attempts > 0:
            for i in range(len(partners), 0, -1):
                try:
                    self.node.get_connection(partners[i])
                    partners.pop(i)
                except DiscoveryError:
                    continue
            if not len(partners):
                success = True
                break
            else:
                n_attempts -= 1
            sleep(1)

        if not success:
            raise HandshakeFailureException(f"Node {self.node.node_name} failed to handshake with {partners}")

    def process(self, task: tasks.Task):
        task_class = type(task)
        corresponding_method = self.task_map[task_class]
        corresponding_method(task)

    def start(self):
        self.status = "running"
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
        out = self.model.forward(task.input, inference_id=inference_id, start=task.start_layer, end=task.end_layer)
 
        if task.downstream_node is not None and isinstance(task.end_layer, int):
            downstream_node_svc = self.node.get_connection(task.downstream_node)
            downstream_task = tasks.SimpleInferenceTask(self.node.node_name, out, inference_id=inference_id, start_layer=task.end_layer)
            downstream_node_svc.give_task(downstream_task)

    def inference_sequence_per_input(self, task: tasks.SingleInputInferenceTask):
        """
        If you want to use a partitioner or conduct multiple inferences per input, this is where
        you'd implement that behavior, most likely using the provided self.simple_inference method,
        possibly with start_layer and end_layer being determined with a partitioner.
        """
        raise NotImplementedError(f"inference_sequence_per_input not implemented for {self.node.node_name} runner")

    def infer_dataset(self, task: tasks.InferOverDatasetTask):
        """
        Run the self.inference_sequence_per_input method for each element in the dataset.
        """
        dataset_dirname, dataset_instance = task.dataset_dirname, task.dataset_instance
        observer_svc = self.node.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        dataset = observer_svc.get_dataset_reference(dataset_dirname, dataset_instance)
        dataloader = DataLoader(dataset, batch_size=1)

        for input in dataloader:
            subtask = tasks.SingleInputInferenceTask("SELF", input)
            self.inference_sequence_per_input(subtask)


class BaseObserverRunner(BaseRunner):
    """
    Waits for its partners to set their status to "ready" as a part of its own setup sequence,
    then sends a predetermined set of tasks for each partner before sending a start signal to 
    each.
    """

