from torch.utils.data import Dataset
from rpyc.core.protocol import Connection


class BaseDataset(Dataset):
    """
    Implements basic functionality required for any dataset.
    """

    length: int

    def __getitem__(self, index):
        """
        Ensures elements from the dataset can be retrieved using square bracket notation.
        """
        raise NotImplementedError("Datasets must have a __getitem__ method")

    def __len__(self) -> int:
        return self.length


class RemoteDataset(BaseDataset):
    """
    Wraps the real torch dataset on the other side of the connection
    """

    def __init__(self, conn: Connection, module_name: str, dataset_instance_from_module: str):
        self._conn = conn
        self.get_remote_service().dataset_load(module_name, dataset_instance_from_module)
        length = self.get_remote_service().dataset_len()
        if length is None:
            raise ValueError("Could not get length from remote dataset")
        self.length = length

    def __getitem__(self, index):
        return self.get_remote_service().dataset_getitem(index)

    def get_remote_service(self) -> ObserverService | ParticipantService:
        if self._conn.root is None:
            raise ValueError("RemoteDataset could not access its upstream service")
        return self._conn.root
