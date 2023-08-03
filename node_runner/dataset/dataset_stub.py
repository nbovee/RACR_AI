from enum import Enum

class DatasetEnum(Enum):
    ImageNet = 1

class DatasetBase():
    def has_next(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError
    