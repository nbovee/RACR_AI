import rpyc
import sys
from importlib import import_module
from pathlib import Path


class ObserverService(rpyc.Service):
    """
    The service exposed by the observer device during experiments.
    """
    ALIASES: list[str] = ["OBSERVER"]
    USR_DATASETS_PATH: Path = Path(__file__).parent / "DatasetObjects"

    def __init__(self):
        super().__init__()
        sys.path.append(str(self.USR_DATASETS_PATH.absolute()))
        self.client_connections = {}

    def on_connect(self, conn):
        self.client_connections[conn.root.gethost()] = conn

    def on_disconnect(self, conn):
        for host in self.client_connections:
            if self.client_connections[host] == conn:
                self.client_connections.pop(host)

    def exposed_dataset_load(self, module_name:str, dataset_instance:str):
        module = import_module(module_name)
        self.dataset = getattr(module, dataset_instance)

    def exposed_dataset_len(self):
        if not self.dataset:
            return None
        return len(self.dataset)

    def exposed_dataset_getitem(self, idx: int):
        if not self.dataset:
            return None
        return self.dataset[idx]

