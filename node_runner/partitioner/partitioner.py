import abc
from typing import Any
from itertools import cycle


class Partitioner:
    """factory class for the method of determining split location in a model. Custom partitioners can be written in their own module and dropped into this directory for automatic import."""

    subclasses = {}

    # @classmethod implicit
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls._TYPE] = cls

    @classmethod
    def create(cls, class_type, *args, **kwargs):
        if class_type not in cls.subclasses:
            raise ValueError("Bad type {}".format(class_type))
        return cls.subclasses[class_type](*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
