from pathlib import Path
import os
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_design.models.model_hooked import WrappedModel, HookExitException

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

# Define the path to your dataset
dataset_path = 'onion/testing'

# Custom dataset class to load images
class OnionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [f for f in os.listdir(root) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]

# Custom dataset transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Load your dataset
dataset = OnionDataset(root=dataset_path, transform=transform)

# DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize the YOLO model
yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
m = WrappedModel(config_path=yaml_file_path)
m2 = WrappedModel(config_path=yaml_file_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m.to(device)
m2.to(device)

# Split layer index
split_layer = 5

def forward_with_split(model1, model2, images, split_layer):
    try:
        res = model1(images, end=split_layer)
    except HookExitException as e:
        res = e.result

    # Extract the intermediate output tensor from the result
    intermediate_output = res[split_layer - 1] if isinstance(res, dict) else res

    # Run the second part of the model from the split layer
    final_output = model2(intermediate_output, start=split_layer)

    return final_output

def draw_bounding_boxes(image, boxes, labels, class_names):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        x_min, x_max = np.min([x_min, x_max]), np.max([x_min, x_max])
        y_min, y_max = np.min([y_min, y_max]), np.max([y_min, y_max])
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        if isinstance(label, torch.Tensor):
            label = label.item() if label.numel() == 1 else int(label[0].item())
        elif isinstance(label, np.ndarray):
            label = int(label[0])
        draw.text((x_min, y_min), class_names[label], fill="red")
    return image

class_names = {0: 'with_weeds', 1: 'without_weeds'}

# Run inference on the dataset
m.eval()
m2.eval()
with torch.no_grad():
    for images, image_files in tqdm(data_loader, desc=f"Testing split at layer {split_layer}"):
        images = images.to(device)
        final_output = forward_with_split(m, m2, images, split_layer)
        
        # Convert predictions to bounding boxes
        predictions = final_output[0]
        boxes = predictions[:, :4].cpu().numpy()
        labels = predictions[:, 5].cpu().numpy().astype(int)

        # Load the original image
        original_image = Image.open(os.path.join(dataset_path, image_files[0])).convert("RGB")
        original_image = draw_bounding_boxes(original_image, boxes, labels, class_names)
        
        # Display the image with bounding boxes
        plt.imshow(original_image)
        plt.axis('off')
        plt.show()
        print(f'Processed {image_files[0]} successfully through split model.')
