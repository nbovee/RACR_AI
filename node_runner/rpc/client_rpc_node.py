import rpyc
import uuid
from rpyc.utils.server import ThreadedServer
from rpyc.utils.helpers import classpartial

class ParticipantService(rpyc.Service):
    def __init__(self, ident, input_dict) -> None:
        self.indent = ident
        self.master_dict = input_dict

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_send_result(self, result, uuid):
        '''sends the parsed result TO the service being called. UUID may or may not have suffix.'''
        pass

    def exposed_get_inference_dict(self, uuid): # this is an exposed method
        uuid = str(uuid) # force cast it if we are being lazy with passing
        return self.master_dict.pop(uuid, [])
    
class ProcessingService(ParticipantService):
    def __init__(self, ident, input_dict) -> None:
        super().__init__(ident, input_dict)
    
    def exposed_send_layer(self, serialized_tensor, parent_uuid, start_layer, end_layer):
        '''sends a tensor TO the service being called. UUID has a suffix.'''
        # add tensor to queue that feeds to local model. 
        pass

# rpc_service = classpartial(ParticipantService) # decoupled early just in case
# node1 = ThreadedServer(rpc_service, port=18861) # rpyc 4.0+ only, single server for all incoming requests to Node to reduce overhead