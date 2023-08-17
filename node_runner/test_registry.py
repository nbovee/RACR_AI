#!/usr/bin/env python

# test file that creates local nodes for testing
import threading
import time
import rpyc
from rpyc import ThreadedServer
import atexit
import copy
import yaml

from observer_rpc_node import ObserverServer

participant_servers = []


def create_servers():
    global reg
    reg = ObserverServer(allow_listing=True)
    atexit.register(stop_registry)

def start_registry():
    print(f"Started Registry Server.")
    reg.start()

def stop_registry():
    print(f"Stopping Registry Server.")
    reg.close()

def start_servers():
    threads = [threading.Thread(target=start_registry)]
    for t in threads:
        t.daemon = True # this lets us kill with KeyboardInterrupt
        t.start()

def main():
    create_servers()
    start_servers()
    while True:
        print(f"Active Node Types: {rpyc.list_services()}")
        if len(rpyc.list_services()) > 0:
            for i in rpyc.list_services():
                print(f"Active Nodes of {i}: {rpyc.discover(i)}")
        time.sleep(5)

if __name__ == "__main__":
    main()
