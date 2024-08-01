import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from ultralytics import YOLO
from src.tracr.experiment_design.models.model_hooked import WrappedModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tracr_logger")

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to model's expected input size
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Function to extract the tensor at a specific layer in the YOLO model
def extract_tensor_yolo(model, image_tensor, layer_idx):
    outputs = []
    hooks = []

    def hook_fn(module, input, output):
        outputs.append(output)

    # Register hooks for the specified layer
    for idx, layer in enumerate(model.model.model[:layer_idx+1]):
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Run the model to get the output at the specified layer
    model.predict(image_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return outputs[-1]

# Extract tensor from NotDict object
def get_tensor_from_notdict(notdict_obj, layer):
    if hasattr(notdict_obj, 'inner_dict'):
        inner_dict = notdict_obj.inner_dict
        print('inner_dict', inner_dict)
        if layer in inner_dict:
            return inner_dict[layer]
    raise AttributeError("Tensor not found in NotDict object")

# Load the image
image_path = "./car2.jpg"
input_tensor = load_image(image_path)

# Load the original YOLO model
yolo_model = YOLO("yolov8s.pt")

# Extract the tensor at layer 5 from the original YOLO model
layer_idx = 5
original_tensor = extract_tensor_yolo(yolo_model, input_tensor, layer_idx)

# Print the extracted tensor from the original YOLO model
print(f"Tensor from the original YOLO model at layer {layer_idx}:")
print(original_tensor)

# Load split models
yaml_file_path = os.path.join(str(Path(__file__).resolve().parents[0]), "model_test.yaml")
m = WrappedModel(config_path=yaml_file_path)
m2 = WrappedModel(config_path=yaml_file_path)

# Pass the image through the split model layers
logging.info(f"Switch at: {layer_idx}")
res = m(input_tensor, end=layer_idx)
logging.info("Switched.")
res2 = m2(res, start=layer_idx)

# Extract the tensor at layer 5 from the split YOLO model
split_tensor = get_tensor_from_notdict(res, layer_idx)

# Print the extracted tensor from the split YOLO model
print("Tensor from the split YOLO model at layer 5:")
print(split_tensor)

# Ensure both tensors are on the same device before comparison
original_tensor = original_tensor.to('cpu')
split_tensor = split_tensor.to('cpu')

# Compare the tensors
if torch.allclose(original_tensor, split_tensor):
    print("The tensors from the original and split YOLO models are the same.")
else:
    print("The tensors from the original and split YOLO models are different.")

# Debug: Check intermediate outputs layer by layer
def compare_intermediate_outputs(yolo_model, split_model, input_tensor, num_layers):
    yolo_outputs = []
    split_outputs = []

    def yolo_hook_fn(module, input, output):
        yolo_outputs.append(output)

    def split_hook_fn(module, input, output):
        split_outputs.append(output)

    # Hook each layer of the YOLO model
    yolo_hooks = []
    for idx, layer in enumerate(yolo_model.model.model[:num_layers]):
        hook = layer.register_forward_hook(yolo_hook_fn)
        yolo_hooks.append(hook)

    # Run the YOLO model to get outputs
    yolo_model.predict(input_tensor)

    # Remove YOLO hooks
    for hook in yolo_hooks:
        hook.remove()

    # Hook each layer of the split model
    split_hooks = []
    for idx, layer in enumerate(split_model.model.model[:num_layers]):
        hook = layer.register_forward_hook(split_hook_fn)
        split_hooks.append(hook)

    # Run the split model to get outputs
    split_model(input_tensor)

    # Remove split hooks
    for hook in split_hooks:
        hook.remove()

    # Ensure all tensors are on the same device
    yolo_outputs = [output.to('cpu') for output in yolo_outputs]
    split_outputs = [output.to('cpu') for output in split_outputs]

    # Compare outputs
    for i in range(num_layers):
        if not torch.allclose(yolo_outputs[i], split_outputs[i]):
            print(f"Discrepancy found at layer {i}")
            print(f"YOLO output: {yolo_outputs[i]}")
            print(f"Split model output: {split_outputs[i]}")
            break
    else:
        print("All intermediate outputs match.")

# Compare intermediate outputs layer by layer
compare_intermediate_outputs(yolo_model, m, input_tensor, layer_idx + 1)
