import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from src.experiment_design.models.model_hooked import WrappedModel

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

yaml_file_path = os.path.join(
    str(Path(__file__).resolve().parents[0]), "model_test.yaml"
)
m = WrappedModel(config_path=yaml_file_path)
m2 = WrappedModel(config_path=yaml_file_path)

# Load and preprocess the image
test_image_path = "./car2.jpg"
test_image = Image.open(test_image_path).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to model's expected input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
])

test_image = preprocess(test_image).unsqueeze(0)  # Add batch dimension

def tensor_to_image(tensor):
    # Assuming the tensor is a normalized image tensor, remove normalization
    unnormalized = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    unnormalized = unnormalized.squeeze().permute(1, 2, 0).detach().numpy()
    return unnormalized

def visualize_tensor(tensor, title):
    if tensor.ndim == 4 and tensor.shape[1] == 3:  # Check if the output is a batch of images with 3 channels
        img = tensor_to_image(tensor[0])
        plt.imshow(img)
        plt.title(title)
        plt.show()

for i in range(19, 23):  # Adjust the range according to the layer count
    logging.info(f"Switch at: {i}")
    res = m(test_image, end=i)
    logging.info("Switched.")
    res2 = m2(res, start=i)
    # Print the output shape for each switch
print("type of result is:", type(res2))
print("result:", res2)
bbox_tensor = res2[0]
print("bbox_tensor:", bbox_tensor)
# Convert the tensor to a numpy array
bbox_array = bbox_tensor.squeeze().numpy()

# Load the original image
image_path = "./car2.jpg"
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# Get image dimensions
image_width, image_height = image.size

# Define a function to draw bounding boxes
def draw_bounding_boxes(image, bbox_array, image_width, image_height):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for bbox in bbox_array:
        x1, y1, x2, y2, conf, cls = bbox[:6]

        # Ensure coordinates are in the correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if conf > 0.5:  # Confidence threshold
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{int(cls)}: {conf:.2f}", fill="red", font=font)

# Convert the tensor to a numpy array and draw bounding boxes
bbox_array = bbox_tensor.squeeze().numpy()

# Draw the bounding boxes on the image
draw_bounding_boxes(image, bbox_array, image_width, image_height)

# Show the image with bounding boxes
image.show()
