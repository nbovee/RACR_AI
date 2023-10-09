import rpyc
from typing import Type

from model_hooked import WrappedModel
from base_lib.runner import BaseRunner
from base_lib.node_service import NodeService
from observer_lib.observer_service import ObserverService


@rpyc.service
class ParticipantService(NodeService):
    """
    The service exposed by all participating nodes regardless of their node role in the test case.
    A test case is defined by mapping node roles to physical devices available on the local
    network, and a node role is defined by passing a model and runner to the ZeroDeployedServer
    instance used to deploy this service. This service expects certain methods to be available
    for each.
    """

    ALIASES = ["PARTICIPANT"]

    runner: BaseRunner
    model: WrappedModel

    def __init__(self,
                 ModelCls: Type[WrappedModel],
                 RunnerCls: Type[BaseRunner]
                 ):
        super().__init__()
        self.prepare_model(ModelCls)
        self.prepare_runner(RunnerCls)

    def prepare_model(self, ModelCls):
        observer_svc = self.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        master_dict = observer_svc.get_master_dict()
        self.model = ModelCls(master_dict=master_dict)

    def prepare_runner(self, RunnerCls):
        self.runner = RunnerCls(self)


