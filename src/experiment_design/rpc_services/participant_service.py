import rpyc
import torch.nn as nn
from typing import Type
from queue import PriorityQueue

from tasks.tasks import Task
from models.model_hooked import WrappedModel
from runners.runner import BaseExecutor
from rpc_services.node_service import NodeService
from rpc_services.observer_service import ObserverService


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

    executor: BaseExecutor
    model: WrappedModel
    inbox: PriorityQueue[Task]

    def __init__(self,
                 ModelCls: Type[nn.Module] | None,
                 RunnerCls: Type[BaseExecutor]
                 ):
        super().__init__()
        self.prepare_model(ModelCls)
        self.prepare_runner(RunnerCls)

    def prepare_model(self, ModelCls):
        observer_svc = self.get_connection("OBSERVER")
        assert isinstance(observer_svc, ObserverService)
        master_dict = observer_svc.get_master_dict()
        if not isinstance(ModelCls, nn.Module):
            self.model = WrappedModel(master_dict=master_dict)
        else:
            self.model = WrappedModel(pretrained=ModelCls(), master_dict=master_dict)

    def prepare_runner(self, RunnerCls):
        self.runner = RunnerCls(self)

    @rpyc.exposed
    def give_task(self, task: Task):
        self.inbox.put(task)

    @rpyc.exposed
    def run(self):
        assert self.status == "ready"
        self.executor.start()
