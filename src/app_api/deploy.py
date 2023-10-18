from io import BufferedReader
from time import sleep
import rpyc
import logging
import sys
import subprocess
from rpyc.utils.zerodeploy import DeployedServer
from rpyc.core.stream import SocketStream
from plumbum.machines.remote import RemoteCommand
from plumbum import local, CommandNotFound
from plumbum.path import copy
from plumbum.commands.base import BoundCommand

import src.app_api.device_mgmt as dm
import src.app_api.utils as utils


logger = logging.getLogger("main_logger")


SERVER_SCRIPT = r"""\
import sys
import os
import atexit
import shutil
import logging
import logging.handlers
from importlib import import_module
from threading import Event


server_module     = "$SVR-MODULE$"
server_class      = "$SVR-CLASS$"
model_class       = "$MOD-CLASS$"
model_module      = "$MOD-MODULE$"
ps_module         = "$PS-MODULE$"
ps_class          = "$PS-CLASS$"
node_name         = "$NODE-NAME$"
participant_host  = "$PRT-HOST$"
observer_ip       = "$OBS-IP$"
max_uptime        = $MAX-UPTIME$


def setup_remote_logger(node_name, host, observer_ip):
    logger = logging.getLogger(f"{node_name}_logger")
    logger.setLevel(logging.DEBUG)
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

if model_class and model_module:
    logger.info(f"Importing {model_class} from src.experiment_design.models.{model_module}.")
    m = import_module(f"src.experiment_design.models.{model_module}")
    Model = getattr(m, model_class)
else:
    logger.info("Using default model (AlexNet)")
    Model = None

logger.info(f"Importing {ps_class} from src.experiment_design.node_behavior.{ps_module}.")
m = import_module(f"src.experiment_design.node_behavior.{ps_module}")
CustomParticipantService = getattr(m, ps_class)

logger.info("Constructing participant_service instance.")
participant_service = CustomParticipantService(Model)

done_event = Event()
participant_service.set_done_event(done_event)

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
    done_event.wait(timeout=max_uptime)
finally:
    close_server_finally()
"""

class ZeroDeployedServer(DeployedServer):

    stdout: BufferedReader
    stderr: BufferedReader

    def __init__(self,
                 device: dm.Device,
                 node_name: str,
                 model: tuple[str, str],
                 participant_service: tuple[str, str],
                 server_class="rpyc.utils.server.ThreadedServer",
                 python_executable=None,
                 timeout_s: int | float = 30):
        logger.debug(
            f"Constructing ZeroDeployedServer with params: device={device}, node_name={node_name}, " +
            f"model={model}, participant_service={participant_service}, server_class={server_class}, " +
            f"python_executable={python_executable}"
        )
        assert device.working_cparams is not None
        self.proc = None
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
        ps_module, ps_class = participant_service
        observer_ip = utils.get_local_ip()
        logger.info(utils.get_local_ip())
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
                "$PS-MODULE$", ps_module
            ).replace(
                "$PS-CLASS$", ps_class
            ).replace(
                "$NODE-NAME$", node_name
            ).replace(
                "$OBS-IP$", observer_ip
            ).replace(
                "$PRT-HOST$", participant_host
            ).replace(
                "$MAX-UPTIME$", str(timeout_s)
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

        assert isinstance(cmd, RemoteCommand)
        self.proc = cmd.popen(script, new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.stdout, self.stderr = self.proc.stdout, self.proc.stdout
        print(type(self.stdout), type(self.stderr))

    def _connect_sock(self):
        return SocketStream._connect(self.remote_machine.host, 18861)

