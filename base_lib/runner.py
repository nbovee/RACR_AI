from __future__ import annotations
import uuid
from queue import PriorityQueue
from typing import Callable
from torch import Tensor
from PIL import Image
from torch.utils.data import DataLoader

import base_lib.tasks as tasks
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
    inbox: PriorityQueue
    active_connections: dict[str, NodeService]
    model: WrappedModel | None
    status: str
    task_map: dict[type, Callable]

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
        self.status = "ready"

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
        self.status = "finished"

    def simple_inference(self, task: tasks.SimpleInferenceTask):
        assert self.model is not None
        inference_id = task.inference_id if task.inference_id is not None else str(uuid.uuid4())
        out = self.model.forward(task.input, inference_id=inference_id, start=task.start_layer, end=task.end_layer)
 
        if task.downstream_node is not None and isinstance(task.end_layer, int):
            downstream_node_svc = self.node.get_connection(task.downstream_node)
            downstream_task = tasks.SimpleInferenceTask(self.node.node_name, out, inference_id=inference_id, start_layer=task.end_layer)
            downstream_node_svc.give_task(downstream_task)

    def inference_sequence_per_input(self, input: Tensor | Image.Image):
        """
        If you want to use a partitioner or conduct multiple inferences per input, this is where
        you'd implement that behavior, most likely using the provided self.simple_inference method,
        possibly with start_layer and end_layer being determined with a partitioner.
        """
        self.simple_inference(input: Tensor | Image.Image)    # pyright: ignore

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

