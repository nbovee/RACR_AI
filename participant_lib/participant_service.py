import rpyc
from typing import Type

from model_hooked import WrappedModel
from base_lib.runner import BaseRunner
from base_lib.metric_dict import MasterDict


@rpyc.service
class ParticipantService(rpyc.Service):
    """
    The service exposed by all participating nodes regardless of their node role in the test case.
    A test case is defined by mapping node roles to physical devices available on the local
    network, and a node role is defined by passing a DataRetriever, model, and runner to the 
    ZeroDeployedServer instance used to deploy this service. This service expects certain methods
    to be available for each.
    """

    ALIASES = ["PARTICIPANT"]

    runner: BaseRunner
    model: WrappedModel
    master_dict: MasterDict

    def __init__(self,
                 ModelCls: Type[WrappedModel],
                 RunnerCls: Type[BaseRunner]
                 ):
        super().__init__()
        self.active_connections = {}
        self.prepare_model(ModelCls)
        self.prepare_runner(RunnerCls)

    def prepare_model(self, ModelCls):
        self.model = ModelCls(master_dict = self.master_dict)

    def prepare_runner(self, RunnerCls):
        pass

