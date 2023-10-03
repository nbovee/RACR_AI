import rpyc
import sys
from pathlib import Path
from rpyc.utils.zerodeploy import DeployedServer
from rpyc.core.stream import SocketStream
from plumbum import local, CommandNotFound, SshMachine

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
from participant_lib.participant_service import ParticipantService

from user_lib.dataloaders.$DL-MODULE$ import $DL-CLASS$ as DataLoader
from user_lib.models.$MOD-MODULE$ import $MOD-CLASS$ as Model
from user_lib.schedulers.$SCH-MODULE$ import $SCH-CLASS$ as Scheduler

participant_service = ParticipantService(DataLoader, Model, Scheduler)

logger = None

server = ServerCls(participant_service, port = 18861, reuse_addr = True, logger = logger, auto_register = True)
atexit.register(server.close)
server.start()

try:
    sys.stdin.read()
finally:
    server.close()
"""

class ZeroDeployedServer(DeployedServer):

    def __init__(self,
                 remote_machine: SshMachine,
                 dataloader: tuple[str, str],
                 model: tuple[str, str],
                 scheduler: tuple[str, str],
                 server_class="rpyc.utils.server.ThreadedServer",
                 python_executable=None):
        self.proc = None
        self.tun = None
        self.remote_machine = remote_machine
        self._tmpdir_ctx = None

        # Create a temp dir on the remote machine where we make the environment
        self._tmpdir_ctx = remote_machine.tempdir()
        tmp = self._tmpdir_ctx.__enter__()

        # Copy over the rpyc, participant_lib, and user_lib code
        rpyc_root = local.path(rpyc.__file__).up()
        copy(rpyc_root, tmp / "rpyc")

        participant_lib_root = local.path((Path(__file__).parent.parent / "participant_lib").absolute())
        copy(participant_lib_root, tmp/ "participant_lib")

        user_lib_root = local.path((Path(__file__).parent.parent / "user_lib").absolute())
        copy(user_lib_root, tmp/ "user_lib")

        # Substitute placeholders in the remote script and send it over
        script = (tmp / "deployed-rpyc.py")
        modname, clsname = server_class.rsplit(".", 1)
        dl_module, dl_class = dataloader
        m_module, m_class = model
        sch_module, sch_class = scheduler
        script.write(
            SERVER_SCRIPT.replace(
                "$MODULE$", modname
            ).replace(
                "$SERVER$", clsname
            ).replace(
                "$DL-MODULE$", dl_module
            ).replace(
                "$DL-CLASS$", dl_class
            ).replace(
                "$MOD-MODULE$", m_module
            ).replace(
                "$MOD-CLASS$", m_class
            ).replace(
                "$SCH-MODULE$", sch_module
            ).replace(
                "$SCH-CLASS$", sch_class
            )
        )

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

