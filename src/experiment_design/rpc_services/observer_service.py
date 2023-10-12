import rpyc
import logging
from importlib import import_module

from datasets.dataset import BaseDataset
from experiment_design.runners.runner import BaseDelegator
from rpc_services.node_service import NodeService
from records.master_dict import MasterDict


@rpyc.service
class ObserverService(NodeService):
    """
    The service exposed by the observer device during experiments.
    """

    ALIASES: list[str] = ["OBSERVER"]

    master_dict: MasterDict
    delegator: BaseDelegator

    def __init__(self, delegator: BaseDelegator):
        super().__init__()
        self.logger = logging.getLogger("main_logger")
        self.master_dict = MasterDict()
        self.prepare_delegator(delegator)

    def prepare_delegator(self, delegator: BaseDelegator):
        """
        Makes sure the delegator is ready to officially start the experiment. Note that the 
        delegator must already have a playbook and list of partners, set using `set_playbook` and
        `set_partners`.
        """
        self.delegator = delegator
        self.delegator.link_node(self)
        self.delegator.get_ready()

    @rpyc.exposed
    def get_master_dict(self) -> MasterDict:
        return self.master_dict

    @rpyc.exposed
    def get_dataset_reference(self, dataset_module: str, dataset_instance: str) -> BaseDataset:
        """
        Allows remote nodes to access datasets stored on the observer as if they were local objects.
        """
        module = import_module(f"src.experiment_design.datasets.{dataset_module}")
        dataset = getattr(module, dataset_instance)
        return dataset

    @rpyc.exposed
    def send_log(self):
        pass

    @rpyc.exposed
    def run(self):
        assert self.status == "ready"
        self.delegator.start()
