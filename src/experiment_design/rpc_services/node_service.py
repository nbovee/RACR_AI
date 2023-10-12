from __future__ import annotations
import logging
import rpyc
from rpyc.core.protocol import Connection


@rpyc.service
class NodeService(rpyc.Service):
    """
    Implements all the endpoints common to both participant and observer nodes.
    """
    ALIASES: list[str]

    active_connections: dict[str, NodeService]
    node_name: str
    status: str
    logger: logging.Logger

    def __init__(self):
        super().__init__()
        self.status = "initializing"
        self.node_name = self.ALIASES[0].upper().strip()
        if self.node_name.upper().strip() != "OBSERVER":
            self.logger = logging.getLogger(f"{self.node_name}_logger")
        self.active_connections = {}

    def get_connection(self, node_name: str) -> NodeService:
        node_name = node_name.upper().strip()
        self.logger.debug(f"Attempting to get connection to {node_name}.")
        if node_name in self.active_connections:
            self.logger.debug(f"Connection to {node_name} already memoized.")
            return self.active_connections[node_name]
        self.logger.debug(
            f"Connection to {node_name} not memoized; attempting to access via registry."
        )
        conn = rpyc.connect_by_service(node_name, service=self)  # type: ignore
        self.active_connections[node_name] = conn.root
        self.logger.info(f"New connection to {node_name} established and memoized.")
        return self.active_connections[node_name]

    def on_connect(self, conn: Connection):
        if isinstance(conn.root, NodeService):
            self.logger.info(
                f"on_connect method called; memoizing connection to {conn.root.get_node_name()}"
            )
            self.active_connections[conn.root.get_node_name()] = conn.root

    def on_disconnect(self, conn: Connection):
        self.logger.info("on_disconnect method called; removing saved connection.")
        for name in self.active_connections:
            if self.active_connections[name] == conn:
                self.active_connections.pop(name)
                self.logger.info(f"Removed connection to {name}")

    @rpyc.exposed
    def get_status(self) -> str:
        self.logger.debug(f"get_status exposed method called; returning '{self.status}'")
        return self.status

    @rpyc.exposed
    def get_node_name(self) -> str:
        self.logger.debug(f"get_node_name exposed method called; returning '{self.node_name}'")
        return self.node_name

ns = NodeService()
ns._connect()
