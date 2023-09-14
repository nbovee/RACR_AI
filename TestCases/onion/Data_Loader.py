from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Step 1: Prepare Your Dataset Structure
class CustomDataset(Dataset):
    def __init__(self, root_dir, image_set_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load the list of image file names from the image set file (e.g., train.txt)
        with open(image_set_file, 'r') as file:
            self.image_paths = [line.strip() for line in file.readlines()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'JPEGImages', self.image_paths[idx] + '.jpg')
        image = Image.open(img_name)

        # Load annotations or labels here if needed
        # annotation_file = os.path.join(self.root_dir, 'Annotations', self.image_paths[idx] + '.xml')
        # annotations = parse_annotation(annotation_file)

        if self.transform:
            image = self.transform(image)

        # Return image and corresponding labels/annotations
        return image

# Step 2: Data Transformations (Optional)
data_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.ToTensor()])

# Step 3: Instantiate the Custom Dataset and DataLoader
dataset = CustomDataset(root_dir='onion',
                        image_set_file='onion/ImageSets/Main/train.txt',
                        transform=data_transforms)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


for batch in data_loader:
    # Your training code here
    print(len(batch))

