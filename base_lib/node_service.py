from __future__ import annotations
from queue import PriorityQueue
import rpyc
from rpyc.core.protocol import Connection

from communication import Task


@rpyc.service
class NodeService(rpyc.Service):
    """
    Implements all the endpoints common to both participant and observer nodes.
    """
    ALIASES: list[str]

    active_connections: dict[str, NodeService]
    node_name: str
    inbox: PriorityQueue[Task]
    status: str

    def __init__(self):
        super().__init__()
        self.node_name = self.ALIASES[0].upper().strip()
        self.active_connections = {}

    def get_connection(self, node_name: str) -> NodeService:
        node_name = node_name.upper().strip()
        if node_name in self.active_connections:
            return self.active_connections[node_name]
        conn = rpyc.connect_by_service(node_name)
        self.active_connections[node_name] = conn.root
        return self.active_connections[node_name]

    def on_connect(self, conn: Connection):
        if isinstance(conn.root, NodeService):
            self.active_connections[conn.root.get_node_name()] = conn.root

    def on_disconnect(self, conn: Connection):
        for name in self.active_connections:
            if self.active_connections[name] == conn:
                self.active_connections.pop(name)

    @rpyc.exposed
    def echo(self, input: str) -> str:
        return input

    @rpyc.exposed
    def get_node_name(self) -> str:
        return self.node_name

    @rpyc.exposed
    def give_task(self, task: Task):
        self.inbox.put(task)

