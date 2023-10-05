import rpyc
import sys
from importlib import import_module
from pathlib import Path


@rpyc.service
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

    @rpyc.exposed
    def dataset_load(self, module_name:str, dataset_instance:str):
        module = import_module(module_name)
        self.dataset = getattr(module, dataset_instance)

    @rpyc.exposed
    def dataset_len(self):
        if not self.dataset:
            return None
        return len(self.dataset)

    @rpyc.exposed
    def dataset_getitem(self, idx: int):
        if not self.dataset:
            return None
        return self.dataset[idx]

    @rpyc.exposed
    def send_log(self):
        pass
