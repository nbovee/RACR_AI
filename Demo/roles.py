import rpyc


class TestAService(rpyc.Service):

    def __init__(self):
        super().__init__()
        self.poked = False
    
    def exposed_poke_service(self, name):
        if name.upper() in rpyc.list_services():
            connection = rpyc.connect_by_service(name)
            connection.root.set_poked(True)


class TestBService(rpyc.Service):

    def __init__(self):
        super().__init__()
        self.poked = False
    
    def exposed_set_poked(self, val):
        self.poked = val

    def exposed_get_poked(self):
        return self.poked

