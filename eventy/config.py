import os
import pathlib
from enum import Enum
from typing import Optional, Union

from pydantic import BaseSettings
from pydantic_loader.yaml_config import load_yaml, save_yaml


class Config(BaseSettings):
    device: str
    epochs: int = 1

    @staticmethod
    def load_file(path: Union[str, pathlib.Path]):
        config = load_yaml(Config, pathlib.Path(path))
        return config

    def save_file(self, path: Union[str, pathlib.Path]):
        os.makedirs(pathlib.Path(path).parent, exist_ok=True)
        config = save_yaml(self, pathlib.Path(path))
        return config
