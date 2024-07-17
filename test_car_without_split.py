from pathlib import Path
import os
import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("tracr_logger")


# Function to run inference and display results
def run_inference():
    

    # Initialize the YOLO model for inference
    model_path = 'yolov8s.pt'  # Adjust the path to your best.pt file
    model = YOLO(model_path)
    test_image_path = "./car2.jpg"
    test_image = Image.open(test_image_path).convert("RGB")
    results = model.predict(source=test_image,save=True)  # Perform inference

if __name__ == '__main__':
    run_inference()
