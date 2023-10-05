import rpyc
from rpyc.core.protocol import Connection
from torch import Tensor
from typing import Any

from communication import DataReceiver, DataSender, Envelope, Request

@rpyc.service
class NodeService(rpyc.Service):
    """
    Implements all the endpoints common to both participant and observer nodes.
    """
    ALIASES: list[str]

    client_connections: dict[str, Connection]
    data_receiver: DataReceiver
    data_sender: DataSender
    role: str

    def __init__(self):
        super().__init__()

    def on_connect(self, conn):
        self.client_connections[conn.root.getrole()] = conn

    def on_disconnect(self, conn):
        for host in self.client_connections:
            if self.client_connections[host] == conn:
                self.client_connections.pop(host)

    @rpyc.service
    def echo(self, input: str) -> str:
        return input

    @rpyc.service
    def get_role(self) -> str:
        return self.role

    @rpyc.service
    def send_data(self, data: Envelope) -> None:
        self.data_receiver.accept(data)

    @rpyc.service
    def get_data(self, request: Request) -> Any:
        self.data_sender.process(request)
