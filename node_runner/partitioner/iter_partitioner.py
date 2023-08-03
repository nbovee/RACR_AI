from .partitioner import Partitioner
from itertools import count, cycle
from typing import Any


class CyclePartitioner(Partitioner):
    _TYPE = "cycle"

    def __init__(self, num_breakpoints, clip_min_max=True) -> None:
        super().__init__()
        self.breakpoints = num_breakpoints
        if clip_min_max:
            self.counter = cycle(range(1, self.breakpoints))
        else:
            self.counter = cycle(range(0, self.breakpoints + 1))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return next(self.counter)


class CountPartitioner(Partitioner):
    _TYPE = "count"

    def __init__(self, start) -> None:
        super().__init__()
        self.breakpoints = start
        self.counter = count(start)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return next(self.counter)
