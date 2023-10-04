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
        dataloader.acknowledge_clients(self.client_connections)
        
        self.dataloader = dataloader

    def prepare_model(self, ModelCls):
        model = ModelCls()
        self.model = model

    def prepare_scheduler(self, SchedulerCls):
        scheduler = SchedulerCls()
        self.scheduler = scheduler

    def on_connect(self, conn):
        self.client_connections[conn.root.getrole()] = conn

    def on_disconnect(self, conn):
        for host in self.client_connections:
            if self.client_connections[host] == conn:
                self.client_connections.pop(host)

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
