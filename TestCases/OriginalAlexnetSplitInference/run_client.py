import src.colab_vision_client as cv
import atexit
from test_data import test_data_loader as data_loader
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 run_client.py <IP> <port>")
        exit(1)
    client = cv.FileClient(sys.argv[1] + ":" + sys.argv[2])
    print("Constructed client.")
    atexit.register(client.safeClose)
    test_data = data_loader()
    print("Initialized data_loader.")
    while test_data.has_next():
        client.initiateInference(test_data)
