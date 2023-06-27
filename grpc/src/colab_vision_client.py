import sys
import logging
import os
import pathlib
import grpc
import pandas as pd
import time
import uuid
import blosc2 as blosc
import numpy as np


sys.path.append(".")
parent = os.path.abspath(".")
sys.path.insert(1, parent)


import alexnet_pytorch_split as alex

from . import colab_vision
from . import colab_vision_pb2
from . import colab_vision_pb2_grpc


client_mode = "cpu"
test_results_dir = pathlib.Path(sys.path[0]) / "test_results"


class FileClient:
    def __init__(self, address, result_output=test_results_dir):
        self.channel = grpc.insecure_channel(address)
        self.stub = colab_vision_pb2_grpc.colab_visionStub(self.channel)
        self.results_dict = {}
        self.result_output = result_output
        logging.basicConfig()
        self.model = alex.Model(mode=client_mode)

    def safeClose(self):
        self.channel.close()
        df = pd.DataFrame(data=self.results_dict)
        current_datetime = time.strftime("%d-%m-%Y_%h:%m:%s")
        df.to_csv(self.result_output / f"test_results__{current_datetime}")

    def initiateInference(self, target):
        messages = self.stub.constantInference(self.inference_generator(target))
        for received_msg in messages:
            self.results_dict[received_msg.id][
                "server_result_class"
            ] = received_msg.results
            self.results_dict[received_msg.id]["client_complete_time"] = time.time()
            for key, val in received_msg.keypairs.items():
                self.results_dict[received_msg.id][key] = val

    def inference_generator_test(self, data_loader):
        for i in range(5):
            yield colab_vision_pb2.Info_Chunk(id="test")

    def inference_generator(self, data_loader):
        tmp = data_loader.next()
        while tmp:
            size_packets = 0
            try:
                [current_obj, exit_layer, filename] = next(tmp)
            except StopIteration:
                return
            message = colab_vision_pb2.Info_Chunk()
            message.id = (
                uuid.uuid4().hex
            )  # uuid4().bytes is utf8 not unicode like grpc wants
            message.layer = exit_layer  # server begins inference 1 layer past edge exit
            self.results_dict[message.id] = {}
            self.results_dict[message.id]["filename"] = filename
            self.results_dict[message.id]["client_mode"] = client_mode
            self.results_dict[message.id]["split_layer"] = exit_layer
            self.results_dict[message.id]["compression_level"] = "9"
            self.results_dict[message.id]["client_start_time"] = time.time()
            # print(f"exit layer: {exit_layer}")
            current_obj = self.model.predict(current_obj, end_layer=exit_layer)
            self.results_dict[message.id]["client_predict_time"] = time.time()
            self.results_dict[message.id]["client_tensor_raw_bytes"] = 32 * np.prod(
                list(current_obj.size())
            )
            if colab_vision.USE_COMPRESSION:
                message.action.append(colab_vision_pb2.ACT_COMPRESSED)
                # Custom compression sizes require we provide tensor shape info to the server
                # current_obj = blosc.compress(current_obj.numpy().to_bytes(), clevel = 9) #force = True if we move to 1.13
                # current_obj = blosc.pack_tensor(current_obj)
                current_obj = blosc.pack_array(current_obj.cpu().numpy())
                self.results_dict[message.id]["client_compression_time"] = time.time()
            # send all pieces
            message.action.append(colab_vision_pb2.ACT_RESET)
            for i, piece in enumerate(colab_vision.get_object_chunks(current_obj)):
                message.chunk.chunk = piece
                if i == 1:
                    message.action.remove(colab_vision_pb2.ACT_RESET)
                yield message  # might be sending twice?
                size_packets += len(message.chunk.chunk)
            message.ClearField("chunk")
            message.chunk.chunk = b""
            # clear RESET from single msg inferences
            if colab_vision_pb2.ACT_RESET in message.action:
                message.action.remove(colab_vision_pb2.ACT_RESET)
            message.action.append(colab_vision_pb2.ACT_INFERENCE)
            yield message  # might be sending twice?
            self.results_dict[message.id]["client_upload_time"] = time.time()
            self.results_dict[message.id]["client_upload_bytes"] = size_packets
