from enum import Enum
import abc
from typing import Any
from itertools import cycle

class PartitionerEnum(Enum):
    Cycle = 1
    Neurosurgeon = 2

class PartitionerBase():
    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
    
class CyclePartitioner(PartitionerBase):
    def __init__(self, num_breakpoints, clip_min_max = True) -> None:
        super().__init__()
        self.breakpoints = num_breakpoints
        if clip_min_max:
            self.counter = cycle(range(1, self.breakpoints))
        else:
            self.counter = cycle(range(0, self.breakpoints+1))
            
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return next(self.counter)
