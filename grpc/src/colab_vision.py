import sys
import os
import pickle

sys.path.append(".")
parent = os.path.abspath(".")
sys.path.insert(1, parent)


# this should probably be an independent database that client and server can both
# interact with async
BITRATE = 0.1 * 2**20  # byte/s
USE_COMPRESSION = True
CHUNK_SIZE = 1024 * 1024  # 1MB


def get_object_chunks(object):
    object = pickle.dumps(object)
    for pos in range(0, len(object), CHUNK_SIZE):
        piece = object[pos : pos + CHUNK_SIZE]
        if len(piece) == 0:
            return
        yield piece


def save_chunks_to_object(chunks):
    chunk_byte_list = []
    for c in chunks:
        chunk_byte_list.append(c.chunk)
    obj_bytes = b"".join(chunk_byte_list)
    return pickle.loads(obj_bytes)


def calculate_transfer_speed(data_size_MB, new_bitrate=None):
    if new_bitrate is not None:
        return data_size_MB / new_bitrate
    else:
        return data_size_MB / BITRATE
