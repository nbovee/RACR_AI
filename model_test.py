"""Basic invocation script for model_hooked module"""
from pathlib import Path
import os
import logging

from src.experiment_design.models.model_hooked import WrappedModel

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("tracr_logger")

yaml_file_path = os.path.join(
    str(Path(__file__).resolve().parents[0]), "config.yaml"
)
m = WrappedModel(config_path=yaml_file_path)
