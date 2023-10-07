import uuid
import numpy as np
from PIL import Image
from torch import Tensor
from dataclasses import dataclass, field


@dataclass(order=True)
class Task:

    from_node: str = field(compare=False)
    priority: int = 5    # from 1 to 10

    def __init__(self, from_node: str, priority: int = 5):
        self.priority = priority
        self.from_node = from_node


class SimpleInferenceTask(Task):

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
    pass


class InferOverDatasetTask(Task):
    pass


class FinishSignalTask(Task):
    """
    Sort of like a sentry value that lets a Runnner know when it's done. Priority is set to an 
    abnormally high value to ensure it's always processed last.
    """

    priority: int = 11

    def __init__(self, from_node: str):
        super().__init__(from_node)
