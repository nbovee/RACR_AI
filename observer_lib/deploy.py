import rpyc
import sys
from pathlib import Path
from rpyc.utils.zerodeploy import DeployedServer
from rpyc.core.stream import SocketStream
from plumbum import local, CommandNotFound
from plumbum.path import copy
from plumbum.commands.base import BoundCommand


SERVER_SCRIPT = r"""\
import sys
import os
import atexit
import shutil

here = os.path.dirname(__file__)
os.chdir(here)

def rmdir():
    shutil.rmtree(here, ignore_errors = True)
atexit.register(rmdir)

try:
    for dirpath, _, filenames in os.walk(here):
        for fn in filenames:
            if fn == "__pycache__" or (fn.endswith(".pyc") and os.path.exists(fn[:-1])):
                os.remove(os.path.join(dirpath, fn))
except Exception:
    pass

sys.path.insert(0, here)
from $MODULE$ import $SERVER$ as ServerCls
from pnodelib.runner.test import $SERVICE$ as Service

logger = None

server = ServerCls(Service(), port = 18861, reuse_addr = True, logger = logger, auto_register = True)
server.start()
atexit.register(server.close)

try:
    sys.stdin.read()
finally:
    server.close()
"""

class ZeroDeployedServer(DeployedServer):

    def __init__(self, remote_machine, service_class, server_class="rpyc.utils.server.ThreadedServer",
                 python_executable=None):
        self.proc = None
        self.tun = None
        self.remote_machine = remote_machine
        self._tmpdir_ctx = None

        rpyc_root = local.path(rpyc.__file__).up()
        self._tmpdir_ctx = remote_machine.tempdir()
        tmp = self._tmpdir_ctx.__enter__()
        copy(rpyc_root, tmp / "rpyc")

        pnodelib_root = local.path((Path(__file__).parent.parent / "pnodelib").absolute())
        copy(pnodelib_root, tmp/ "pnodelib")

        script = (tmp / "deployed-rpyc.py")
        modname, clsname = server_class.rsplit(".", 1)
        script.write(SERVER_SCRIPT.replace("$MODULE$", modname).replace(
            "$SERVER$", clsname).replace("$SERVICE$", service_class))
        if isinstance(python_executable, BoundCommand):
            cmd = python_executable
        elif python_executable:
            cmd = remote_machine[python_executable]
        else:
            major = sys.version_info[0]
            minor = sys.version_info[1]
            cmd = None
            for opt in [f"python{major}.{minor}", f"python{major}"]:
                try:
                    cmd = remote_machine[opt]
                except CommandNotFound:
                    pass
                else:
                    break
            if not cmd:
                cmd = remote_machine.python

        self.proc = cmd.popen(script, new_session=True)

    def _connect_sock(self):
        return SocketStream._connect(self.remote_machine.host, 18861)

