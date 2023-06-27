import src.colab_vision_server as cv
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 run_server.py <port>")
        exit(1)
    port = sys.argv[1]

    cv.FileServer().start(port)
