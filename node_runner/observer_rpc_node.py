import rpyc
# from rpyc.utils.registry import RegistryServer


class ObserverServer(rpyc.utils.registry.UDPRegistryServer):

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_inference_completed_signal(self, uuid):
        # to be implemented
        pass