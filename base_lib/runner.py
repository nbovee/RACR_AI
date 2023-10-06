from __future__ import annotations
from queue import PriorityQueue
from typing import Callable

from base_lib.communication import Request
from base_lib.node_service import NodeService
from participant_lib.model_hooked import WrappedModel
from participant_lib.participant_service import ParticipantService


class BaseRunner:
    """
    The runner has a more general role than the name suggests. A better name for this component
    for the future might be "runner" or something similar. The runner takes a DataRetriever and a 
    WrappedModel as parameters to __init__ so it can essentially guide the participating node 
    through its required sequence of tasks.
    """

    node: NodeService
    rulebook: dict[str, Callable[[BaseRunner, Request], None]]
    inbox: PriorityQueue
    active_connections: dict[str, NodeService]
    model: WrappedModel | None
    status: str

    def __init__(self,
                 node: NodeService,
                 rulebook: dict[str, Callable[[BaseRunner, Request], None]]
                 ):
        self.node = node
        self.rulebook = rulebook
        self.active_connections = self.node.active_connections
        self.inbox = self.node.inbox
        self.status = self.node.status

        self.get_ready()

    def get_ready(self):
        if isinstance(self.node, ParticipantService):
            self.model = self.node.model
        self.status = "ready"

    def process(self, request: Request):
        if request.request_type == "finish":
            self.finish()
            return
        try:
            func = self.rulebook[request.request_type]
        except KeyError:
            raise ValueError(
                f"Rulebook has no key for request type {request.request_type}"
            )
        func(self, request)

    def start(self):
        self.status = "running"
        if self.inbox is not None:
            while self.status == "running":
                current_request = self.inbox.get()
                self.process(current_request)

    def finish(self):
        self.status = "finished"


