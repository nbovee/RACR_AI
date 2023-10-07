from __future__ import annotations
import uuid
import numpy as np
from queue import PriorityQueue
from typing import Callable
from torch import Tensor
from PIL import Image
from torch.utils.data import DataLoader

from base_lib.communication import Task
from base_lib.node_service import NodeService
from observer_lib.observer_service import ObserverService
from participant_lib.model_hooked import WrappedModel
from participant_lib.participant_service import ParticipantService


class BaseRunner:
    """
    The runner has a more general role than the name suggests. A better name for this component
    for the future might be "runner" or something similar. The runner takes a DataRetriever and a 
    WrappedModel as parameters to __init__ so it can essentially guide the participating node 
    through its required sequence of tasks.
    """

    node: NodeService
    rulebook: dict[str, Callable[[BaseRunner, Task], None]]
    inbox: PriorityQueue
    active_connections: dict[str, NodeService]
    model: WrappedModel | None
    status: str

    def __init__(self,
                 node: NodeService,
                 rulebook: dict[str, Callable[[BaseRunner, Task], None]]
                 ):
        self.node = node
        self.rulebook = rulebook
        self.active_connections = self.node.active_connections
        self.inbox = self.node.inbox
        self.status = self.node.status

        self.get_ready()

    def get_ready(self):
        if isinstance(self.node, ParticipantService):
            self.model = self.node.model
        self.status = "ready"

    def process(self, task: Task):
        if task.task_type == "finish":
            self.finish()
            return
        try:
            func = self.rulebook[task.task_type]
        except KeyError:
            raise ValueError(
                f"Rulebook has no key for task type {task.task_type}"
            )
        func(self, task)

    def start(self):
        self.status = "running"
        if self.inbox is not None:
            while self.status == "running":
                current_task = self.inbox.get()
                self.process(current_task)

    def finish(self):
        self.status = "finished"

    def simple_inference(self,
                         input: Tensor | Image.Image,
                         inference_id: str | None = None,
                         start_layer: int = 0,
                         end_layer: int | float = np.inf,
                         downstream_node: str | None = None
                         ):
        assert self.model is not None
        if inference_id is None:
            inference_id = str(uuid.uuid4())
        out = self.model.forward(input, inference_id=inference_id, start=start_layer, end=end_layer)
        
        if downstream_node is not None:
            downstream_node_svc = self.node.get_connection(downstream_node)
            downstream_task = SingleInferenceTask(out, inference_id=inference_id, start_layer=end_layer)
            downstream_node_svc.give_task(downstream_task)

    def inference_sequence_per_input(self, input: Tensor | Image.Image):
        """
        If you want to use a partitioner or conduct multiple inferences per input, this is where
        you'd implement that behavior, most likely using the provided self.simple_inference method,
        possibly with start_layer and end_layer being determined with a partitioner.
        """
        self.simple_inference(input: Tensor | Image.Image)

    def infer_dataset(self, dataset_dirname: str, dataset_instance: str):
        """
        Run the self.inference_sequence_per_input method for each element in the dataset.
        """
        observer_svc = self.node.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        dataset = observer_svc.get_dataset_reference(dataset_dirname, dataset_instance)
        dataloader = DataLoader(dataset, batch_size=1)

        for input in dataloader:
            self.inference_sequence_per_input(input)

