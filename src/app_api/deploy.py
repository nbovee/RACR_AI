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

def setup_remote_logger(node_name, host, observer_ip):
    logger = logging.getLogger(f"{node_name}_logger")
    formatter = logging.Formatter(f"{node_name}@{host}: %(message)s")

    socket_handler = logging.handlers.SocketHandler(observer_ip, 9000)
    socket_handler.setFormatter(formatter)

    logger.addHandler(socket_handler)
    return logger

logger = setup_remote_logger("$NODE_NAME$", "$PARTICIPANT_HOST$", "$OBSERVER_IP$")

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

logger.info("Executing remote script line 'from $MODULE$ import $SERVER$ as ServerCls'")
from $MODULE$ import $SERVER$ as ServerCls

logger.info("Importing ParticipantService in remote script")
from experiment_design.rpc_services.participant_service import ParticipantService

if "$MOD-CLASS$" and "$MOD-MODULE$":
    logger.info("Importing $MOD-CLASS$ from experiment_design.models.$MOD-MODULE$.")
    from experiment_design.models.$MOD-MODULE$ import $MOD-CLASS$ as Model
else:
    Model = None

logger.info("Importing $SCH-CLASS$ from experiment_design.runners.$SCH-MODULE$.")
from experiment_design.runners.$SCH-MODULE$ import $SCH-CLASS$ as Runner

logger.info("Defining new class $NODE_NAME$Service(ParticipantService) to control formal name.")
class $NODE_NAME$Service(ParticipantService):
    ALIASES = ["$NODE_NAME$", "PARTICIPANT"]

logger.info("Constructing participant_service instance.")
participant_service = $NODE_NAME$Service(Model, Runner)

logger.info("Starting RPC server.")
server = ServerCls(participant_service, port=18861, reuse_addr=True, logger=logger, auto_register=True)

def close_server_atexit():
    logger.info("Closing server due to atexit invocation.")
    server.close()

def close_server_finally():
    logger.info("Closing server after 'finally' clause was reached in SERVER_SCRIPT.")
    server.close()

atexit.register(close_server_atexit)
server.start()

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
                 runner: tuple[str, str],
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
        sch_module, sch_class = runner
        observer_ip = utils.get_local_ip()
        participant_host = device.working_cparams.host
        script.write(
            SERVER_SCRIPT.replace(    # type: ignore
                "$MODULE$", modname
            ).replace(
                "$SERVER$", clsname
            ).replace(
                "$MOD-MODULE$", m_module
            ).replace(
                "$MOD-CLASS$", m_class
            ).replace(
                "$SCH-MODULE$", sch_module
            ).replace(
                "$SCH-CLASS$", sch_class
            ).replace(
                "$NODE_NAME$", node_name
            ).replace(
                "$OBSERVER_IP$", observer_ip
            ).replace(
                "$PARTICIPANT_HOST$", participant_host
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

