import rpyc
import sys
from importlib import import_module
from pathlib import Path

from datasets.dataset import BaseDataset
from rpc_services.node_service import NodeService
from records.master_dict import MasterDict


@rpyc.service
class ObserverService(NodeService):
    """
    The service exposed by the observer device during experiments.
    """

    ALIASES: list[str] = ["OBSERVER"]
    USR_DATASETS_PATH: Path = Path(__file__).parent.parent / "MyData" / "Datasets"

    master_dict: MasterDict

    def __init__(self):
        super().__init__()
        self.master_dict = MasterDict()
        sys.path.append(str(self.USR_DATASETS_PATH.absolute()))

    @rpyc.exposed
    def get_master_dict(self) -> MasterDict:
        return self.master_dict

    @rpyc.exposed
    def get_dataset_reference(self, dataset_module: str, dataset_instance: str) -> BaseDataset:
        """
        Allows remote nodes to access datasets stored on the observer as if they were local objects.
        """
        module = import_module(f"datasets.{dataset_module}")
        dataset = getattr(module, dataset_instance)
        return dataset

    @rpyc.exposed
    def send_log(self):
        pass
