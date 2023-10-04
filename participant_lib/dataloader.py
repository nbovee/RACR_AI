from __future__ import annotations
import rpyc
import uuid
from rpyc.utils.server import ThreadedServer
from rpyc.utils.helpers import classpartial
from rpyc.utils.zerodeploy import DeployedServer
from rpyc.utils.factory import connect
from rpyc.core.protocol import Connection
from importlib import import_module
import blosc2
import time
import atexit
import threading
import torch
from torch.utils.data import DataLoader, Dataset

from observer_lib.observer_service import ObserverService
from participant_lib.participant_service import ParticipantService


class RemoteDataset(Dataset):
    """
    Wraps the real torch dataset on the other side of the connection
    """
    def __init__(self, conn: Connection, module_name: str, dataset_instance_from_module: str):
        self._conn = conn
        self.get_remote_service().dataset_load(module_name, dataset_instance_from_module)
        self.length = self.get_remote_service().dataset_len()

    def __getitem__(self, index):
        return self.get_remote_service().dataset_getitem(index)

    def __len__(self):
        return self.length

    def get_remote_service(self) -> ObserverService | ParticipantService:
        if self._conn.root is None:
            raise ValueError("RemoteDataset could not access its upstream service")
        return self._conn.root


class BaseDataLoader:
    """
    All user-defined DataLoaders should inherit from this class to ensure compatibility with 
    the ParticipantService they will be loaded into for experiments.
    """

    REMOTE_DATASOURCE_ROLE: str | None = None
    DATASET_MODULE_NAME: str
    DATASET_INSTANCE_FROM_MODULE: str

    conn: Connection | None = None
    dataset: Dataset

    def __iter__(self) -> BaseDataLoader:
        raise NotImplementedError(f"DataLoaders must have an __iter__ method implemented!")

    def __next__(self):
        raise NotImplementedError(f"DataLoaders must have a __next__ method implemented!")

    def init_dataset(self, client_connections: dict[str, Connection]):
        """
        The ParticipantService does not know if its dataloader will need an upstream connection,
        so it will always pass its client connections to this method.
        """
        if self.REMOTE_DATASOURCE_ROLE is not None:
            upstream_datasource_role = self.REMOTE_DATASOURCE_ROLE 
            dataset_module_name = self.DATASET_MODULE_NAME
            dataset_instance_from_module = self.DATASET_INSTANCE_FROM_MODULE

            self.conn = client_connections[upstream_datasource_role]
            self.dataset = RemoteDataset(self.conn, dataset_module_name, dataset_instance_from_module)
        elif self.DATASET_MODULE_NAME is not None and self.DATASET_INSTANCE_FROM_MODULE is not None:
            module = import_module(self.DATASET_MODULE_NAME)
            self.dataset = getattr(module, self.DATASET_INSTANCE_FROM_MODULE)
        else:
            raise ValueError(
                "Children of BaseDataLoader must have DATASET_MODULE_NAME and DATASET_INSTANCE_FROM_MODULE class attributes!"
            )


