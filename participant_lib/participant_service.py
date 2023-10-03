import rpyc


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

    def __init__(self, DataLoaderCls, ModelCls, SchedulerCls, add_aliases=[]):
        super().__init__()
        ParticipantService.add_aliases(add_aliases)
        self.client_connections = {}
        self.prepare_dataloader(DataLoaderCls)
        self.prepare_model(ModelCls)
        self.prepare_scheduler(SchedulerCls)

    def prepare_dataloader(self, DataLoaderCls):
        dataloader = DataLoaderCls(self.client_connections)
        self.dataloader = dataloader

    def prepare_model(self, ModelCls):
        model = ModelCls()
        self.model = model

    def prepare_scheduler(self, SchedulerCls):
        scheduler = SchedulerCls()
        self.scheduler = scheduler

    def on_connect(self, conn):
        self.client_connections[conn.root.gethost()] = conn

    def on_disconnect(self, conn):
        for host in self.client_connections:
            if self.client_connections[host] == conn:
                self.client_connections.pop(host)
