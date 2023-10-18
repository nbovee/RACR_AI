import atexit
from datetime import datetime
import pathlib
import pickle
import threading
from time import sleep
import rpyc
from rpyc.utils.factory import DiscoveryError
from rpyc.utils.server import ThreadedServer
import yaml
from rpyc.utils.registry import UDPRegistryServer
from socketserver import TCPServer

from src.app_api.deploy import ZeroDeployedServer
from src.experiment_design.node_behavior.base import ObserverService

import src.experiment_design.tasks.tasks as tasks
import src.app_api.utils as utils
import src.app_api.device_mgmt as dm
import src.app_api.log_handling as log


TEST_CASE_DIR: pathlib.Path = utils.get_repo_root() / "MyData" / "TestCases"


class ExperimentManifest:
    """
    A representation of the yaml file used to specify the way an experiment runs.
    """
    
    participant_types: dict[str, dict[str, dict[str, str]]]
    participant_instances: list[dict[str, str]]
    playbook: dict[str, list[tasks.Task]]
    name: str

    def __init__(self, manifest_fp: pathlib.Path):
        p_types, p_instances, playbook_as_dict = self.read_and_parse_file(manifest_fp)
        self.name = manifest_fp.stem
        self.set_ptypes(p_types)
        self.set_p_instances(p_instances)
        self.create_and_set_playbook(playbook_as_dict)

    def read_and_parse_file(
            self,
            manifest_fp: pathlib.Path
        ) -> tuple[dict[str, dict], list[dict[str, str]], dict[str, list]]:
        """
        Reads the given file and returns the three subsections as a tuple:
        `(participant_types, participant_instances, playbook)`
        """
        with open(manifest_fp, 'r') as file:
            manifest_dict = yaml.load(file, yaml.Loader)
        participant_types = manifest_dict["participant_types"]
        participant_instances = manifest_dict["participant_instances"]
        playbook = manifest_dict["playbook"]
        return participant_types, participant_instances, playbook

    def set_ptypes(self, ptypes: dict[str, dict]) -> None:
        self.participant_types = ptypes

    def set_p_instances(self, pinstances: list[dict[str, str]]):
        self.participant_instances = pinstances

    def create_and_set_playbook(
            self, playbook: dict[str, list[dict[str, str | dict[str, str]]]]
        ) -> None:
        new_playbook = {instance_name: [] for instance_name in playbook.keys()}
        for instance_name, tasklist in playbook.items():
            assert len(tasklist)
            task_object = None
            for task_as_dict in tasklist:
                task_object = None
                assert isinstance(task_as_dict["task_type"], str)
                task_type = task_as_dict["task_type"].lower()

                # TODO: handle all task types, or better yet, find a nice way to do this
                if "inf" in task_type and "dataset" in task_type:
                    params = task_as_dict["params"]
                    assert isinstance(params, dict)
                    task_object = tasks.InferOverDatasetTask(
                        params["dataset_module"], params["dataset_instance"]
                    )
                elif "finish" in task_type:
                    task_object = tasks.FinishSignalTask()

            assert task_object is not None
            new_playbook[instance_name].append(task_object)

        self.playbook = new_playbook

    def get_participant_instance_names(self) -> list[str]:
        return [participant["instance_name"] for participant in self.participant_instances]

    def get_zdeploy_params(
            self, available_devices: list[dm.Device]
        ) -> list[tuple[dm.Device, str, tuple[str, str], tuple[str, str]]]:
        result = []
        # instances where device is "any" go last since they don't care
        for instance in sorted(self.participant_instances, key=lambda x: 1 if x["device"] == "any" else 0):
            device = instance["device"]
            for d in available_devices:
                if d._name == device or device.lower() == "any":
                    node_name = instance["instance_name"]
                    model_specs = self.participant_types[instance["node_type"]]["model"]
                    model = tuple([model_specs["module"], model_specs["class"]])
                    if "default" in model:
                        model = tuple(["", ""])
                    service_specs = self.participant_types[instance["node_type"]]["service"]
                    service = tuple([service_specs["module"], service_specs["class"]])
                    param_tuple = tuple([d, node_name, model, service])
                    result.append(param_tuple)
                    available_devices.remove(d)
                    break
            else:
                raise dm.DeviceUnavailableException(
                    f"Experiment manifest specifies device {device} for" +
                    f" {instance['instance_name']}, but it is unavailable."
                )
        return result


class Experiment:
    """
    The interface the application uses to finally run the experiment.
    """

    available_devices: list[dm.Device]
    manifest: ExperimentManifest
    registry_server: UDPRegistryServer = UDPRegistryServer(allow_listing=True)
    remote_log_server: TCPServer = TCPServer(("localhost", 9000), log.LogRecordStreamHandler)
    observer_node: ThreadedServer
    observer_conn: ObserverService
    participant_nodes: list[ZeroDeployedServer] = []
    threads: dict[str, threading.Thread]
    events: dict[str, threading.Event]

    def __init__(self, manifest: ExperimentManifest, available_devices: list[dm.Device]):
        self.available_devices = available_devices
        self.manifest = manifest
        self.threads = {
            "registry_svr": threading.Thread(target=self.start_registry, daemon=True),
            "remote_log_svr": threading.Thread(target=self.start_remote_log_server, daemon=True),
            "observer_svr": threading.Thread(target=self.start_observer_node, daemon=True),
            "participant_svrs": threading.Thread(target=self.start_participant_nodes, daemon=True),
        }
        self.events = {
            "registry_ready": threading.Event(),
            "remote_log_ready": threading.Event(),
            "observer_up": threading.Event(),
        }

    def run(self) -> None:
        """
        Runs the experiment according to the current attributes set. Experiments can be modified
        from their original state by changing self.manifest before calling this method.
        """
        self.threads["registry_svr"].start()
        self.threads["remote_log_svr"].start()

        self.events["registry_ready"].wait()
        self.events["remote_log_ready"].wait()

        self.threads["observer_svr"].start()
        self.events["observer_up"].wait()

        self.threads["participant_svrs"].start()

        self.verify_all_nodes_up()
        self.start_handshake()

        self.wait_for_ready()
        self.send_start_signal_to_observer()
        self.cleanup_after_finished()

    def start_registry(self) -> None:
        atexit.register(self.registry_server.close)
        self.registry_server.start()

    def check_registry_server(self):
        for _ in range(5):
            if utils.registry_server_is_up():
                self.events["registry_ready"].set()
                return
        raise TimeoutError(f"Registry server took too long to become available")

    def start_remote_log_server(self) -> None:
        atexit.register(self.remote_log_server.shutdown)
        self.remote_log_server.serve_forever()
        self.events["remote_log_ready"].set()

    def check_remote_log_server(self) - None:


    def start_observer_node(self) -> None:
        all_node_names = self.manifest.get_participant_instance_names()
        playbook = self.manifest.playbook

        observer_service = ObserverService(all_node_names, playbook)
        self.observer_node = ThreadedServer(observer_service, auto_register=True)

        atexit.register(self.observer_node.close)
        self.observer_node.start()
        self.events["observer_up"].set()

    def start_participant_nodes(self) -> None:
        zdeploy_node_param_list = self.manifest.get_zdeploy_params(self.available_devices)
        for params in zdeploy_node_param_list:
            self.participant_nodes.append(ZeroDeployedServer(*params))

    def verify_all_nodes_up(self):
        service_names = self.manifest.get_participant_instance_names()
        service_names.append("OBSERVER")
        n_attempts = 5
        while n_attempts > 0:
            available_services = rpyc.list_services()
            assert isinstance(available_services, tuple)
            if all([(s in available_services) for s in service_names]):
                return
            n_attempts -= 1
            sleep(2)
        available_services = rpyc.list_services()
        assert isinstance(available_services, tuple)
        straglers = [s for s in service_names if s not in available_services]
        raise TimeoutError(
            f"Waited too long for the following services to register: {straglers}"
        )

    def start_handshake(self):
        self.observer_conn = rpyc.connect_by_service("OBSERVER")
        self.observer_conn.get_ready()

    def wait_for_ready(self) -> None:
        n_attempts = 15
        while n_attempts > 0:
            if self.observer_conn.get_status() == "ready":
                return
            n_attempts -= 1 
            sleep(2)
        raise TimeoutError("Experiment object waited too long for observer to be ready.")

    def send_start_signal_to_observer(self) -> None:
        self.observer_conn.run()

    def cleanup_after_finished(self) -> None:
        while True:
            if self.observer_conn.get_status() == "finished":
                break
            sleep(5)

        master_dict = self.observer_conn.get_master_dict()
        fn = f"{self.manifest.name}__{datetime.now().strftime('%Y-%m-%dT%H%M%S')}.pkl"
        fp = utils.get_repo_root() / "MyData" / "TestResults" / fn
        with open(fp, 'w') as file:
            pickle.dump(master_dict, file)  # type: ignore

        self.observer_node.close()
        self.registry_server.close()
        for p in self.participant_nodes:
            p.close()
        self.remote_log_server.shutdown()

