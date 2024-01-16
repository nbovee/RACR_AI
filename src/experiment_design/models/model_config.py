import logging

import numpy as np
import yaml

logger = logging.getLogger("tracr_logger")

def read_model_config(path=None, participant_key = 'client'):
    config_details = __read_yaml_data(path, participant_key)
    return config_details

def __read_yaml_data(path, participant_key):
    settings = {}
    try:
        with open(path, "r") as file:
            settings = yaml.safe_load(file)['participant_types'][participant_key]['model']
    except Exception:
        logging.warning("No valid configuration provided. Using default settings, behavior could be unexpected.")

    # add default entries here just in case
        settings.device = settings.get("device", "cpu")
        settings.mode = settings.get("mode", "eval")
        settings.hook_depth = settings.get("depth", np.inf)
        settings.base_input_size = settings.get("image_size", (3, 224, 224))
        # settings.dataset_type = kwargs.get("dataset_type", "balanced")
        settings.model_name = settings.get("model_name", "alexnet").lower().strip()

    return settings
