from __future__ import annotations
from rpyc.core.protocol import Connection
from importlib import import_module
from torch.utils.data import Dataset

from observer_lib.observer_service import ObserverService
from participant_lib.participant_service import ParticipantService


class BaseDataRetriever:
    """
    All user-defined DataRetrievers should inherit from this class to ensure compatibility with 
    the ParticipantService they will be loaded into for experiments.
    """

    REMOTE_DATASOURCE_ROLE: str | None = None
    DATASET_MODULE_NAME: str
    DATASET_INSTANCE_FROM_MODULE: str

    MAX_ITERATIONS: int | None = None

    conn: Connection | None = None
    dataset: BaseDataset
    index: int

    def __iter__(self) -> BaseDataRetriever:
        if self.dataset is None:
            raise ValueError("Attempted to iterate over a DataRetriever before init_dataset was called!")
        self.index = 0
        if self.dataset.length is not None and self.MAX_ITERATIONS is not None:
            self.length = min((self.dataset.length, self.MAX_ITERATIONS))
        elif self.dataset.length is None:
            raise ValueError("Dataset was not properly initialized")
        else:
            self.length = self.dataset.length
        return self

    def __next__(self):
        if self.length is None:
            raise ValueError("Cannot iterate through DataRetriever before init_dataset is called!")
        if self.index >= self.length:
            raise StopIteration
        result = self.dataset[self.index]
        self.index += 1
        return result 

    def init_dataset(self, client_connections: dict[str, Connection]):
        """
        The ParticipantService does not know if its DataRetriever will need an upstream connection,
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
                "Children of BaseDataRetriever must have DATASET_MODULE_NAME and DATASET_INSTANCE_FROM_MODULE class attributes!"
            )


