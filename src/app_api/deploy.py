import rpyc
import sys
from pathlib import Path
from rpyc.utils.zerodeploy import DeployedServer
from rpyc.core.stream import SocketStream
from plumbum import local, CommandNotFound, SshMachine
from plumbum.path import copy
from plumbum.commands.base import BoundCommand

import src.app_api.device_mgmt as dm
import src.app_api.utils as utils


SERVER_SCRIPT = r"""\
import sys
import os
import atexit
import shutil
import logging
import logging.handlers
from importlib import import_module


server_module     = "$SVR-MODULE$"
server_class      = "$SVR-CLASS$"
model_class       = "$MOD-CLASS$"
model_module      = "$MOD-MODULE$"
executor_module   = "$EXR-MODULE$"
executor_class    = "$EXR-CLASS$"
node_name         = "$NODE-NAME$"
participant_host  = "$PRT-HOST$"
observer_ip       = "$OBS-IP$"


def setup_remote_logger(node_name, host, observer_ip):
    logger = logging.getLogger(f"{node_name}_logger")
    formatter = logging.Formatter(f"{node_name}@{host}: %(message)s")

    socket_handler = logging.handlers.SocketHandler(observer_ip, 9000)
    socket_handler.setFormatter(formatter)

    logger.addHandler(socket_handler)
    return logger

logger = setup_remote_logger(node_name, participant_host, observer_ip)

logger.info("Zero deploy sequence started. Removing __pycache__ and *.pyc files from tempdir.")

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

logger.info(f"Importing {server_class} from {server_module} as ServerCls'")
m = import_module(server_module)
ServerCls = getattr(m, server_class)

logger.info("Importing ParticipantService from src.experiment_design.node_behavior.base.")
from src.experiment_design.node_behavior.base import ParticipantService

if model_class and model_module:
    logger.info(f"Importing {model_class} from src.experiment_design.models.{model_module}.")
    m = import_module(f"src.experiment_design.models.{model_module}")
    Model = getattr(m, model_class)
else:
    logger.info("Using default model (AlexNet)")
    Model = None

logger.info(f"Importing {executor_class} from src.experiment_design.runners.{executor_module}.")
m = import_module(f"src.experiment_design.runners.{executor_module}")
Runner = getattr(m, executor_class)

logger.info(f"Defining new class {node_name}Service(ParticipantService) to control formal name.")
class $NODE_NAME$Service(ParticipantService):
    ALIASES = [node_name, "PARTICIPANT"]

logger.info("Constructing participant_service instance.")
participant_service = $NODE_NAME$Service(Model, Runner)

logger.info("Starting RPC server in thread.")
server = ServerCls(participant_service, port=18861, reuse_addr=True, logger=logger, auto_register=True)

def close_server_atexit():
    logger.info("Closing server due to atexit invocation.")
    server.close()
    server_thread.join(2)

def close_server_finally():
    logger.info("Closing server after 'finally' clause was reached in SERVER_SCRIPT.")
    server.close()
    server_thread.join(2)

atexit.register(close_server_atexit)
server_thread = server._start_in_thread()

try:
    sys.stdin.read()
finally:
    close_server_finally()
"""

class ZeroDeployedServer(DeployedServer):

    def __init__(self,
                 device: dm.Device,
                 node_name: str,
                 model: tuple[str, str],
                 executor: tuple[str, str],
                 server_class="rpyc.utils.server.Server",
                 python_executable=None):
        assert device.working_cparams is not None
        self.proc = None
        self.tun = None
        self.remote_machine = device.as_pb_sshmachine()
        self._tmpdir_ctx = None

        # Create a temp dir on the remote machine where we make the environment
        self._tmpdir_ctx = self.remote_machine.tempdir()
        tmp = self._tmpdir_ctx.__enter__()

        # Copy over the rpyc and experiment_design packages
        rpyc_root = local.path(rpyc.__file__).up()
        copy(rpyc_root, tmp / "rpyc")

        src_root = local.path(utils.get_repo_root() / "src")
        copy(src_root, tmp / "src")

        # Substitute placeholders in the remote script and send it over
        script = (tmp / "deployed-rpyc.py")
        modname, clsname = server_class.rsplit(".", 1)
        m_module, m_class = model
        exr_module, exr_class = executor
        observer_ip = utils.get_local_ip()
        participant_host = device.working_cparams.host
        script.write(
            SERVER_SCRIPT.replace(    # type: ignore
                "$SVR-MODULE$", modname
            ).replace(
                "$SVR-CLASS$", clsname
            ).replace(
                "$MOD-MODULE$", m_module
            ).replace(
                "$MOD-CLASS$", m_class
            ).replace(
                "$EXR-MODULE$", exr_module
            ).replace(
                "$EXR-CLASS$", exr_class
            ).replace(
                "$NODE-NAME$", node_name
            ).replace(
                "$OBS-IP$", observer_ip
            ).replace(
                "$PRT-HOST$", participant_host
            )
        )

        if isinstance(python_executable, BoundCommand):
            cmd = python_executable
        elif python_executable:
            cmd = self.remote_machine[python_executable]
        else:
            major = sys.version_info[0]
            minor = sys.version_info[1]
            cmd = None
            for opt in [f"python{major}.{minor}", f"python{major}"]:
                try:
                    cmd = self.remote_machine[opt]
                except CommandNotFound:
                    pass
                else:
                    break
            if not cmd:
                cmd = self.remote_machine.python

        self.proc = cmd.popen(script, new_session=True)

    def _connect_sock(self):
        return SocketStream._connect(self.remote_machine.host, 18861)

