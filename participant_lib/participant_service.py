import rpyc
from typing import Type
from torch.utils.data import Dataset

from dataloader import BaseDataLoader
from model_hooked import WrappedModel
from scheduler import BaseScheduler

@rpyc.service
class ParticipantService(rpyc.Service):
    """
    The service exposed by all participating nodes regardless of their node role in the test case.
    A test case is defined by mapping node roles to physical devices available on the local
    network, and a node role is defined by passing a dataloader, model, and scheduler to the 
    ZeroDeployedServer instance used to deploy this service. This service expects certain methods
    to be available for each.
    """

    ALIASES = ["PARTICIPANT"]

    dataloader: BaseDataLoader
    scheduler: BaseScheduler
    model: WrappedModel
    master_dict: dict

    @classmethod
    def add_aliases(cls, new_aliases: list[str]):
        cls.ALIASES.extend(new_aliases)

    def __init__(self,
                 DataLoaderCls: Type[BaseDataLoader],
                 ModelCls: Type[WrappedModel],
                 SchedulerCls: Type[BaseScheduler],
                 role: str,
                 downstream_dataset: Dataset | None = None):
        super().__init__()
        self.role = role
        ParticipantService.add_aliases([role])
        self.client_connections = {}
        self.prepare_dataloader(DataLoaderCls)
        self.prepare_model(ModelCls)
        self.prepare_scheduler(SchedulerCls)

    def prepare_dataloader(self, DataLoaderCls):
        dataloader = DataLoaderCls()
        dataloader.init_dataset(self.client_connections)
        self.dataloader = dataloader

    def prepare_model(self, ModelCls):
        self.model = ModelCls(master_dict = self.master_dict)

    def prepare_scheduler(self, SchedulerCls):
        self.scheduler = SchedulerCls(self.dataloader, self.model)


    @rpyc.exposed
    def dataset_load(self, module_name:str, dataset_instance:str):
        pass

    @rpyc.exposed
    def dataset_len(self):
        pass

    @rpyc.exposed
    def dataset_getitem(self, idx: int):
        pass

    @rpyc.exposed
    def getrole(self):
        return self.role
