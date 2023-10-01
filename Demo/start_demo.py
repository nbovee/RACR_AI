import atexit
import sys
import threading
import rpyc

from pathlib import Path
from rpyc.utils.registry import UDPRegistryServer

import onodelib.ssh
import roles
from onodelib.deploy import ZeroDeployedServer




if __name__ == "__main__":

    reg = UDPRegistryServer(allow_listing=True)

    dinfo_fp = Path(__file__).parent / "AppData" / "Store" / "known_devices.yaml"
    dm = onodelib.ssh.DeviceMgr(dinfo_fp)
    available_devices = dm.get_devices(available_only=True)
    print(f"Available Devices: {available_devices}")

    reg_thread = threading.Thread(target=reg.start)
    reg_thread.daemon = True
    reg_thread.start()
    atexit.register(reg.close)

    server_a, server_b = None, None
    for d in available_devices:
        if d._name == "SteveHomePi4":
            server_a = ZeroDeployedServer(d.as_pb_sshmachine(), "TestAService")
        elif d._name == "SteveOldLaptop":
            server_b = ZeroDeployedServer(d.as_pb_sshmachine(), "TestBService")

    print(f"Server A: {server_a}, Server B: {server_b}")
    if not (server_a and server_b):
        sys.exit()

    print(f"List Services: {rpyc.list_services}")
    conn_a, conn_b = rpyc.connect_by_service("TESTA"), rpyc.connect_by_service("TESTB")
    conn_a.root.poke_service("TESTB")
    b_poked = conn_a.root.get_poked()
    print(f"B Poked: {b_poked}")

