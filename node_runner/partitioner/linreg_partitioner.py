from .partitioner import Partitioner
from typing import Any
import torch
import numpy
import os
import csv
import matplotlib.pyplot as plt

class RegressionPartitioner(Partitioner):
    _TYPE = "regression"

    class linreg():

        def __init__(self, *args) -> None:
            self.model = torch.nn.Linear(1,1)
            self.criterion = torch.nn.MSELoss()
            self.optim = torch.optim.SGD(self.model.parameters(), lr=1e-2)
            self.training_iter = 20
            self.loss_list = []
        
        def forward(self, x):
            return self.model(torch.unsqueeze(x,-1))
        
        def train_pass(self, y, pred):
            loss = self.criterion(pred, y)
            # storing the calculated loss in a list
            self.loss_list.append(loss.data)
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # print(loss)
        
        def plot_regression(self):
            plt.plot(self.loss_list, 'r')
            plt.tight_layout()
            plt.grid('True', color='y')
            plt.xlabel("Epochs/Iterations")
            plt.ylabel("Loss")
            plt.show()

    def __init__(self, num_breakpoints, clip_min_max=True) -> None:
        super().__init__()
        self.breakpoints = num_breakpoints
        self.clip = clip_min_max
        self.regression = {}

    def create_data(self, model, iterations = 10):
    # does not currently account for data precision below full float
        temp_data = []
        for i in range(iterations):
            model(torch.randn(1, *model.base_input_size), inference_id = f"profile")
            from_model = model.master_dict.pop("profile")["layer_information"].values()
            temp_data.extend(from_model)
  
        for datapoint in temp_data:
            # note this is append mode so local files need to be cleared before deploying elsewhere
            with open(f'TestCases/AlexnetSplit/partitioner_datapoints/local/{datapoint["class"]}.csv' , 'a') as f:
                f.write(f"{datapoint['parameter_bytes']}, {datapoint['inference_time']}\n")

    def update_regression(self):
        # these should be randomized
        # parse directory, create dict for each filename. Dict holds a list with two values, w & b
        for layer_type in os.listdir('TestCases/AlexnetSplit/partitioner_datapoints/local/'):
            x, y = [], []
            with open(f'TestCases/AlexnetSplit/partitioner_datapoints/local/{layer_type}' , 'r') as f:
                reader = csv.reader(f)
                for line in reader:
                    x.append(float(line[0]))
                    y.append(float(line[1]))
                if max(x) == min(x):
                    print(f"Unstable data for linreg, skipping {layer_type.split('.')[0]}")
                else:
                    self.regression[layer_type.split(".")[0]] = RegressionPartitioner.linreg()
                    current_linreg = self.regression[layer_type.split(".")[0]]
                    # lazy normalize
                    mmax = max(max(x),max(y))
                    x = torch.as_tensor(x)/mmax
                    y = torch.as_tensor(y)/mmax
                               
                    for i in range(current_linreg.training_iter):
                        # variable dataset so batchsize of 1
                        for v, z in zip(x, y):
                            pred = current_linreg.forward(v)
                            current_linreg.train_pass(z, pred)

                        # priting the values for understanding
                        # print('{}, \t{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))

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
