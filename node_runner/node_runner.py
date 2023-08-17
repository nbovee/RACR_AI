from partitioner.linreg_partitioner import RegressionPartitioner
from model.model_hooked import WrappedModel
from participant_rpc_node import ParticipantService

import uuid
import atexit
import torch
import rpyc
from rpyc import ThreadedServer
import threading
import blosc2
import time

timer = time.perf_counter_ns

# this module reads configs and sets up the modules and local rpc
keepalive = True  

def parse_input(self, input):
    """Checks if the input is appropriate at the given stage of the network. Does not yet check Tensor shapes for intermediate layers."""
    if isinstance(input, Image.Image):
        if input.size != self.base_input_size:
            input = input.resize(self.base_input_size)
        input_tensor = self.preprocess(input)
        input_tensor = input_tensor.unsqueeze(0)
    elif isinstance(input, torch.Tensor):
        input_tensor = input
    if (
        torch.cuda.is_available()
        and self.mode == "cuda"
        and input_tensor.device != self.mode
    ):
        input_tensor = input_tensor.to(self.mode)
    return input_tensor

def parse_output(self, predictions):
    """Take the final output tensor of the wrapped model and map it to its appropriate human readable results."""
    probabilities = torch.nn.functional.softmax(predictions[0], dim=0)
    # Show top categories per image
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    prediction = self.categories[top1_catid]
    return prediction

def safeClose(self):
    keepalive = False

def start_server(server_stub):
    
    def thread_loop():
        server_stub.start()
        while keepalive:
            time.sleep(5)
        server_stub.close()

    thread = threading.Thread(target=thread_loop)
    thread.start()

def callback_with_result(uuid, result):
    global master_dictionary
    parent_uuid = uuid.split(".")[0]
    # master_dictionary[parent_uuid][uuid]["transfer_time"] += timer()
    master_dictionary[parent_uuid]["result"] = result
    print(f"callback for {uuid}")
    # master_dictionary[parent_uuid][uuid]["transfer_time"] += timer()
    #signal observer here
    pass

atexit.register(safeClose)
if __name__ == "__main__":
    global master_dictionary
    master_dictionary = dict()
    m = WrappedModel(dict = master_dictionary, depth = 2)
    Scheduler = RegressionPartitioner(m.splittable_layer_count)
    Scheduler.create_data(m)
    Scheduler.update_regression()
    this_server = ThreadedServer(ParticipantService(), auto_register=True)
    this_server.service.link_dict(master_dictionary)

    cloud_conn = None
    while cloud_conn is None:
        try:
            cloud_conn = rpyc.connect_by_service("cloud")
        except:
            pass

    # start our server after finding cloud, so we don't grab it
    start_server(this_server)
    # map cloud server inference to async
    async_infer = rpyc.async_(cloud_conn.root.complete_inference)
    cloud_estimation = cloud_conn.root.get_scheduler_dict()
    Scheduler.add_server_module(cloud_estimation)

    while keepalive:
        s = Scheduler()
        i = torch.randn(1, *m.base_input_size)
        x_uuid = str(uuid.uuid4())
        print(f"Start inference {x_uuid}")
        x = m(i, inference_id = x_uuid, end=s)
        x = blosc2.pack_tensor(x)
        upload_time = 0 - timer()
        async_infer(x, x_uuid, s, callback_with_result)
        # cloud_conn.root.complete_inference(x, x_uuid, s, callback_with_result)
        upload_time += timer()
        master_dictionary[x_uuid]["transfer_time"] = upload_time
        pass
