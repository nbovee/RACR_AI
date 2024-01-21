"""Basic invocation script for model_hooked module"""
from pathlib import Path
import os
import logging

import torch

from src.experiment_design.models.model_hooked import WrappedModel

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

yaml_file_path = os.path.join(
    str(Path(__file__).resolve().parents[0]), "config.yaml"
)
m = WrappedModel(config_path=yaml_file_path)
test_image = torch.randn(1, *m.input_size)

for i in range(1,21):
    logging.info(f"Switch at: {i}")
    res = m(test_image, end = i)
    logging.info("Switched.")
    m(res, start = i)