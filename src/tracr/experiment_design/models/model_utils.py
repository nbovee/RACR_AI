"""utils and interface for the model hook wrapper"""

from typing import Any

import logging
import os

from torchvision import models
from ultralytics import YOLO

import numpy as np
import yaml


class NotDict:
    """Wrapper for a dict to circumenvent some of Ultralytics forward pass handling. Uses a class instead of tuple in case additional handling is added later."""

    def __init__(self, passed_dict) -> None:
        self.inner_dict = passed_dict

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.inner_dict


class HookExitException(Exception):
    """Exception to early exit from inference in naive running."""

    def __init__(self, out, *args: object) -> None:
        super().__init__(*args)
        self.result = out


logger = logging.getLogger("tracr_logger")


def read_model_config(path=None, participant_key="client"):
    config_details = __read_yaml_data(path, participant_key)
    model_fixed_details = {}
    with open(
        os.path.join(os.path.dirname(__file__), "model_configs.yaml"),
        encoding="utf8",
    ) as file:
        model_type = config_details["model_name"]
        # handle various versions of yolo at once
        if "yolo" in model_type.lower():
            model_type = "yolo"
        model_fixed_details = yaml.safe_load(file)[model_type]
        config_details.update(model_fixed_details)
    return config_details


def __read_yaml_data(path, participant_key):
    settings = {}
    try:
        with open(path) as file:
            settings = yaml.safe_load(file)["participant_types"][participant_key][
                "model"
            ]
    except Exception:
        logging.warning(
            "No valid configuration provided. Using default settings, behavior could be unexpected."
        )

        # add default entries here just in case
        settings.device = settings.get("device", "cpu")
        settings.mode = settings.get("mode", "eval")
        settings.hook_depth = settings.get("depth", np.inf)
        settings.input_size = settings.get("image_size", (3, 224, 224))
        # settings.dataset_type = kwargs.get("dataset_type", "balanced")
        settings.model_name = settings.get("model_name", "alexnet").lower().strip()

    return settings


"""module to tie in implemented models"""


def model_selector(model_name):
    if "alexnet" in model_name:
        return models.alexnet(weights="DEFAULT")
    elif "yolo" in model_name:
        return YOLO(
            str(model_name) + ".pt"
        ).model  # pop the real model out of their wrapper for now
    else:
        raise NotImplementedError
