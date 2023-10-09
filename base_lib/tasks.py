import uuid
import numpy as np
from PIL import Image
from torch import Tensor
from dataclasses import dataclass, field


@dataclass(order=True)
class Task:
    """
    Each node, including the Observer node, has an attribute named "inbox", which is a
    PriorityQueue of Task objects. They are sorted by their "priority" attribute in ascending
    order (lowest first). When the "start" method is called on a node, its runner will begin
    popping tasks from the inbox. The runner has a method corresponding to each type of task
    it expects to see in its inbox. The `task_map` attribute shows which is paired with which.

    The runner will wait for new tasks to arrive if its inbox is empty. The node will only stop
    when its runner processes a `FinishSignalTask` object.
    """

    from_node: str = field(compare=False)
    priority: int = 5    # from 1 to 10

    def __init__(self, from_node: str, priority: int = 5):
        self.priority = priority
        self.from_node = from_node


class SimpleInferenceTask(Task):
    """
    Sending this task to a node's inbox is like saying:

    'Here is an input - complete one inference exactly as specified in this task.'

    The node has no say in how the inference is partitioned, the `inference_id`, or where to send 
    the intermediary data (if applicable); this type of task tells the node exactly what to do.
    """

    priority: int = 5
    input: Tensor | Image.Image
    inference_id: str | None = None
    start_layer: int = 0
    end_layer: int | float = np.inf
    downstream_node: str | None = None

    def __init__(self,
                 from_node: str,
                 input: Tensor | Image.Image,
                 inference_id: str | None = None,
                 start_layer: int = 0,
                 end_layer: int | float = np.inf,
                 downstream_node: str | None = None
                 ):
        super().__init__(from_node)
        self.input = input
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.downstream_node = downstream_node
        if self.start_layer == 0 and inference_id is None:
            self.inference_id = str(uuid.uuid4())


class SingleInputInferenceTask(Task):
    """
    Sending this task to a node's inbox is like saying:

    'Here is an input - your runner should have an `inference_sequence_per_input` method that
    specifies how you handle it.'

    The node is free to do whatever it wants with the input. It can use a partitioner to calculate
    the best split point, it can use a scheduler to perform an inference at each possible split 
    point, and it can send intermediary data to any available participant. The user can define
    this behavior by overwriting the `inference_sequence_per_input` method of their custom
    Runner class.

    The `inference_id` attribute can be left as None or filled with a uuid, depending on whether
    the receiving node will be finishing an incomplete inference. 
    """

    priority: int = 5
    input: Tensor | Image.Image
    inference_id: str | None = None

    def __init__(self,
                 from_node: str,
                 input: Tensor | Image.Image,
                 inference_id: str | None = None
                 ):
        super().__init__(from_node)
        self.input = input
        self.inference_id = inference_id


class InferOverDatasetTask(Task):
    """
    Sending this task to a node's inbox is like saying:

    'Here is the name of a dataset instance that should be available to you via the observer's 
    `get_dataset_reference` method. Use your `inference_sequence_per_input` method for each input 
    in the dataset.'

    The node's runner will use its `infer_dataset` method to build a torch DataLoader and iterate
    over each instance in the dataset. This behavior is pretty general, so usually the user won't 
    have to overwrite the `infer_dataset` method that comes with BaseRunner.
    """

    priority: int = 5
    dataset_dirname: str 
    dataset_instance: str 

    def __init__(self,
                 from_node: str,
                 dataset_dirname: str,
                 dataset_instance: str
                 ):
        super().__init__(from_node)
        self.dataset_dirname = dataset_dirname
        self.dataset_instance = dataset_instance


class FinishSignalTask(Task):
    """
    Sort of like a sentry value that lets a Runnner know when it's done. Priority is set to an 
    abnormally high value to ensure it's always processed last.
    """

    priority: int = 11

    def __init__(self, from_node: str):
        super().__init__(from_node)
