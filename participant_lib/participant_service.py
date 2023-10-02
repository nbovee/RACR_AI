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

    def __init__(self, dataloader, model, scheduler, add_aliases=[]):
        super().__init__()
        ParticipantService.add_aliases(add_aliases)
        self.client_connections = {}
        self.set_dataloader(dataloader)
        self.set_model(model)
        self.set_scheduler(scheduler)

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def set_model(self, model):
        self.model = model

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def on_connect(self, conn):
        self.client_connections[conn.root.gethost()] = conn

    def on_disconnect(self, conn):
        for host in self.client_connections:
            if self.client_connections[host] == conn:
                self.client_connections.pop(host)
