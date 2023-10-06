import threading
import rpyc
import sys
from importlib import import_module
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

from base_lib.dataset import BaseDataset
from base_lib.node_service import NodeService


@rpyc.service
class ObserverService(NodeService):
    """
    The service exposed by the observer device during experiments.
    """

    ALIASES: list[str] = ["OBSERVER"]
    USR_DATASETS_PATH: Path = Path(__file__).parent / "DatasetObjects"

    performance_metrics: dict
    performance_metrics_lock: threading.Lock

    @rpyc.exposed
    class ImageDataset(Dataset):
        def __init__(self, annotations_file: Path, img_dir: Path, transform=None, target_transform=None):
            with open(annotations_file, 'r') as file:
                self.img_labels = "\n".split(file.read())
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform
            self.img_map = {img_class: next(self.img_dir.glob(f"*{img_class}*"))
                for img_class in self.img_labels}

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            label = self.img_labels[idx]
            img_fp = self.img_map[label]
            image = Image.open(img_fp).convert("RGB")

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            # TODO: the original script expects the DataRetriever to return [val, splitlayer, filename]
            return image, label

    def __init__(self):
        super().__init__()
        sys.path.append(str(self.USR_DATASETS_PATH.absolute()))

    @rpyc.exposed
    def get_dataset_reference(self, module_name:str, dataset_instance:str) -> BaseDataset:
        """
        Allows remote nodes to access datasets stored on the observer as if they were local objects.
        """
        module = import_module(module_name)
        dataset = getattr(module, dataset_instance)
        return dataset

    @rpyc.exposed
    def send_log(self):
        pass
