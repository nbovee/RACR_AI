import pathlib

from PIL import Image
from torch.utils.data import Dataset


SOURCE_DIRECTORY = pathlib.Path(__file__).parent.parent.parent/ "MyData" / "Datasets" / "imagenet"
CLASS_TEXTFILE = SOURCE_DIRECTORY / "imagenet_classes.txt"
IMG_DIRECTORY = SOURCE_DIRECTORY / "sample_images"


class ImageDataset(Dataset):
    def __init__(self, annotations_file: pathlib.Path, img_dir: pathlib.Path, transform=None, target_transform=None):
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

        # TODO: the original script expects the dataloader to return [val, splitlayer, filename]
        return image, label

# Here is the instance that should be imported from the module (e.g., from image_datasets import imagenet1000_rgb)
imagenet1000_rgb = ImageDataset(CLASS_TEXTFILE, IMG_DIRECTORY)

