import sys

from torchvision import models
from ultralytics import YOLO


def ModelSelector(model_name):
    try:
        if "alexnet" in model_name:
            return models.alexnet(weights="DEFAULT")

        elif "yolo" in model_name:
            return YOLO(str(model_name) + ".pt")
    except Exception as error:
        print("Exception occur due to {0}".format(str(error)))
        sys.exit(1)
