from dataclasses import dataclass, field
from queue import PriorityQueue
import threading

@dataclass(order=True)
class Request:

    priority: int
    request_type: str = field(compare=False)
    from_node: str = field(compare=False)


class Envelope:
    pass


class DataReceiver:
    pass


class DataSender:

    OUTBOX_MAXSIZE = -1

    outbox: PriorityQueue

    def __init__(self):
        self.outbox = PriorityQueue(maxsize=self.OUTBOX_MAXSIZE)

    def accept(self, request: Request):
        self.outbox.put(request)



