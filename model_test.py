from pathlib import Path
import os

from src.experiment_design.models.model_hooked import WrappedModel

yaml_file_path = os.path.join(
    str(Path(__file__).resolve().parents[0]) + "\config.yaml"
)
m = WrappedModel(config_path=yaml_file_path, model_name="yolov5s")
