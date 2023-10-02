import atexit
import sys
import threading
import rpyc

from pathlib import Path
from rpyc.utils.registry import UDPRegistryServer

import observer_lib.ssh
import roles
from observer_lib.deploy import ZeroDeployedServer


def run_demo(device_name_a, device_name_b):

    reg = UDPRegistryServer(allow_listing=True)

    dinfo_fp = Path(__file__).parent / "AppData" / "known_devices.yaml"
    dm = observer_lib.ssh.DeviceMgr(dinfo_fp)
    available_devices = dm.get_devices(available_only=True)
    print(f"Available Devices: {available_devices}")

    reg_thread = threading.Thread(target=reg.start)
    reg_thread.daemon = True
    reg_thread.start()
    atexit.register(reg.close)

    server_a, server_b = None, None
    print("Available Devices:")
    for d in available_devices:
        print(d._name)
        if d._name == device_name_a:
            server_a = ZeroDeployedServer(d.as_pb_sshmachine(), "TestAService")
        elif d._name == device_name_b:
            server_b = ZeroDeployedServer(d.as_pb_sshmachine(), "TestBService")

    print(f"\nServer A: {server_a}, Server B: {server_b}")
    if not (server_a and server_b):
        sys.exit()

    print(f"List Services: {rpyc.list_services}")
    conn_a, conn_b = rpyc.connect_by_service("TESTA"), rpyc.connect_by_service("TESTB")
    conn_a.root.poke_service("TESTB")
    b_poked = conn_b.root.get_poked()
    print(f"B Poked: {b_poked}")


if __name__ == "__main__":

    DEVICE_A_NAME = "RACRJnano4gb-1"
    DEVICE_B_NAME = "RACRJnano4gb-2"

    run_demo(DEVICE_A_NAME, DEVICE_B_NAME)
