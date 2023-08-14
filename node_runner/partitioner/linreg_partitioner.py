from .partitioner import Partitioner
from model.model_hooked import WrappedModel
from typing import Any
import torch
import uuid

class RegressionPartitioner(Partitioner):
    _TYPE = "regression"

    def __init__(self, num_breakpoints, clip_min_max=True, repeats = 1) -> None:
        super().__init__()
        self.breakpoints = num_breakpoints
        self.repeats = repeats if repeats > 0 else 1
        if clip_min_max:
            self.counter = cycle(range(1, self.breakpoints))
        else:
            self.counter = cycle(range(0, self.breakpoints + 1))
    
    def create_data(self, model: WrappedModel, iterations = 1):
        for i in range(iterations):
            model(torch.randn(1, 3, *model.base_input_size), inference_id = f"profile")
            temp_dict = model.inference_dict.pop("profile")["layer_information"]
            print(temp_dict)
            # open('file.txt', 'a') as f:

    def update_regression(self):
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        res = next(self.counter)
        for i in range(self.repeats):
            yield res
