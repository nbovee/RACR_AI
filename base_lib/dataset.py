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

