import threading
import rpyc
import sys
from importlib import import_module
from pathlib import Path

from base_lib.dataset import BaseDataset
from base_lib.node_service import NodeService


@rpyc.service
class ObserverService(NodeService):
    """
    The service exposed by the observer device during experiments.
    """

    ALIASES: list[str] = ["OBSERVER"]
    USR_DATASETS_PATH: Path = Path(__file__).parent.parent / "MyData" / "Datasets"

    performance_metrics: dict
    performance_metrics_lock: threading.Lock

    def __init__(self):
        super().__init__()
        sys.path.append(str(self.USR_DATASETS_PATH.absolute()))

    @rpyc.exposed
    def get_dataset_reference(self, dataset_dirname:str, dataset_instance:str) -> BaseDataset:
        """
        Allows remote nodes to access datasets stored on the observer as if they were local objects.
        """
        module = import_module(f"{dataset_dirname}.instances")
        dataset = getattr(module, dataset_instance)
        return dataset

    @rpyc.exposed
    def send_log(self):
        pass
