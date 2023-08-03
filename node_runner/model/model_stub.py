from enum import Enum
import abc
import torch.nn as nn

class ModelEnum(Enum):
    AlexNet = 1

class ModelBase(nn.Module):
    @abc.abstractmethod
    def forward(self, tensor, start_layer = None, stop_layer = None):
        raise NotImplementedError