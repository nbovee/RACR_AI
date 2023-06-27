import sys
import logging
import os
from concurrent import futures
import grpc
import time
import blosc2 as blosc
import numpy as np
import torch

sys.path.append(".")
parent = os.path.abspath(".")
sys.path.insert(1, parent)

import alexnet_pytorch_split as alex

from . import colab_vision
from . import colab_vision_pb2
from . import colab_vision_pb2_grpc


server_mode = "cuda"


class FileServer(colab_vision_pb2_grpc.colab_visionServicer):
    def __init__(self):
        class Servicer(colab_vision_pb2_grpc.colab_visionServicer):
            def __init__(self):
                self.tmp_folder = "./temp/"
                self.model = alex.Model(mode=server_mode)

            def constantInference(self, request_iterator, context):
                # unpack msg contents
                current_chunks = []
                last_id = None
                reference_time = 0
                for i, msg in enumerate(request_iterator):
                    m = colab_vision_pb2.Response_Dict(
                        id=msg.id, results=None, actions=msg.action, keypairs=None
                    )
                    if colab_vision_pb2.ACT_END in msg.action:
                        # hard exit, dont do it
                        raise Exception("Hard Exit called")
                    # if new id
                    if colab_vision_pb2.ACT_RESET in msg.action:
                        reference_time = np.float32(time.time())
                        m.keypairs.clear()
                        current_chunks = []
                        last_id = msg.id
                        m.keypairs["server_mode"] = 1 if server_mode == "cuda" else 0
                        m.keypairs["server_reference_float"] = reference_time
                        m.keypairs["server_start_time"] = time.time() - reference_time
                    # rebuild data
                    if (
                        msg.id == last_id
                        and colab_vision_pb2.ACT_INFERENCE not in msg.action
                    ):
                        current_chunks.append(msg.chunk)
                    if colab_vision_pb2.ACT_APPEND in msg.action:
                        raise Exception("Append Unsupported")
                    if colab_vision_pb2.ACT_INFERENCE in msg.action:
                        current_chunks = colab_vision.save_chunks_to_object(
                            current_chunks
                        )
                        m.keypairs["server_assemble_time"] = (
                            time.time() - reference_time
                        )
                        # decompress if needed
                        if colab_vision_pb2.ACT_COMPRESSED in msg.action:
                            current_chunks = blosc.unpack_tensor(current_chunks)
                        m.keypairs["server_decompression_time"] = (
                            time.time() - reference_time
                        )  # not sure if this can even be done on instantiation
                        # start inference
                        if (
                            torch.cuda.is_available()
                            and self.model.mode == "cuda"
                            and current_chunks.device != self.model.mode
                        ):
                            current_chunks = current_chunks.to(self.model.mode)
                        m.keypairs["tensor_mode_convert"] = time.time() - reference_time
                        prediction = self.model.predict(
                            current_chunks, start_layer=msg.layer
                        )
                        m.results = prediction.encode()
                        m.keypairs["server_inference_time"] = (
                            time.time() - reference_time
                        )
                        # clean results
                        print(f"Returning prediction for {msg.id}.")
                    yield m

        logging.basicConfig()
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        colab_vision_pb2_grpc.add_colab_visionServicer_to_server(
            Servicer(), self.server
        )

    def start(self, port):
        self.server.add_insecure_port(f"[::]:{port}")
        self.server.start()
        print("Server started.")
        self.server.wait_for_termination()
