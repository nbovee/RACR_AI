from pathlib import Path
import os
import logging

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tracr_logger")

# Define class names (replace with actual class names if different)
class_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light"]

# Dictionary to store the output of the specified layer
layer_outputs = {}

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to model's expected input size
        transforms.ToTensor(),          # Convert image to tensor
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

# Hook function to capture the output
def hook_fn(module, input, output):
    layer_outputs['layer_output'] = output

# Function to run inference and display results
def run_inference(layer_num):
    # Initialize the YOLO model for inference
    model_path = 'yolov8s.pt'  # Adjust the path to your best.pt file
    model = YOLO(model_path)
    
    # Access the specific layer based on the given layer number
    layer = model.model.model[layer_num]
    print(f"Hooking into layer {layer_num}: {layer}")

    # Register the hook to the layer
    hook = layer.register_forward_hook(hook_fn)

    # Load and preprocess the test image
    test_image_path = "./car2.jpg"
    input_tensor, original_image = load_image(test_image_path)
    
    # Normalize input_tensor to range [0, 1]
    input_tensor = input_tensor / input_tensor.max()
    
    # Set model to evaluation mode
    model.model.eval()
    
    # Perform inference using the input tensor
    with torch.no_grad():
        results = model(input_tensor)  # Perform inference directly on the tensor

    # Remove the hook
    hook.remove()

    # Display the output tensor of the specific layer
    if 'layer_output' in layer_outputs:
        print("Output tensor of the specific layer:")
        print(layer_outputs['layer_output'])
    
    # Process the results to draw detections
    detections = results[0].boxes.data.cpu().numpy()  # Adjust as per YOLO result structure
    
    # Debug print the detections
    print("Detections:", detections)
    
    if len(detections) == 0:
        print("No detections were found.")
    else:
        draw = ImageDraw.Draw(original_image)
        font = ImageFont.load_default()

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection[:6]  # Adjust indexing based on the structure
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_name = class_names[int(cls)]  # Map class index to class name
            
            # Ensure bounding boxes are within image dimensions
            x1 = max(0, min(x1, original_image.width))
            y1 = max(0, min(y1, original_image.height))
            x2 = max(0, min(x2, original_image.width))
            y2 = max(0, min(y2, original_image.height))
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f'{cls_name} {conf:.2f}', fill="red", font=font)
        
        # Save or display the image with detections
        original_image.show()  # Display the image
        # original_image.save("output_with_detections.jpg")  # Uncomment to save the image

if __name__ == '__main__':
    # Specify the layer number you want to hook into
    layer_num = 5  # For example, capturing the 5th layer
    run_inference(layer_num)
