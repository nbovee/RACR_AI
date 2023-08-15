from .partitioner import Partitioner
from model.model_hooked import WrappedModel
from typing import Any
import torch
import uuid

class RegressionPartitioner(Partitioner):
    _TYPE = "regression"

    def __init__(self, num_breakpoints, clip_min_max=True) -> None:
        super().__init__()
        self.breakpoints = num_breakpoints
        self.clip = clip_min_max
    
    def create_data(self, model: WrappedModel, iterations = 2):
    # does not currently account for data precision below full float
        temp_data = []
        for i in range(iterations):
            model(torch.randn(1, 3, *model.base_input_size), inference_id = f"profile")
            temp_data.append(model.inference_dict.pop("profile")["layer_information"])
        print(temp_data)    
            # open('file.txt', 'a') as f:

    def update_regression(self):
        pass

    def _get_network_speed_megabits(self):
        # needs work
        return 10
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # yield the first layer where sum of estimated remaining inference time is greater than transfer time + remaining inference time on the server
        min = 1 if self.clip else 0
        max = self.breakpoints -1 if self.clip else self.breakpoints
        res = next(self.counter)
        for i in range(self.repeats):
            yield res
