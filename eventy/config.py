import os
import pathlib
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseSettings, validator
from pydantic_loader.yaml_config import load_yaml, save_yaml


class DatasetConfig(BaseSettings):
    test_split: str
    train_split: str
    validation_split: str


class LossKind(str, Enum):
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"


class ModelConfig(BaseSettings):
    dropout: float


class Config(BaseSettings):
    device: str
    epochs: int = 1
    batch_size: int
    window_size: int
    learning_rate: float
    dataset: DatasetConfig
    loss: List[LossKind]
    model: ModelConfig

    @validator("window_size")
    def odd_window_size(cls, val):
        if val % 2 != 1:
            raise ValueError("Window needs to be an odd number")
        else:
            return val

    @staticmethod
    def load_file(path: Union[str, pathlib.Path]):
        config = load_yaml(Config, pathlib.Path(path))
        return config

    def save_file(self, path: Union[str, pathlib.Path]):
        os.makedirs(pathlib.Path(path).parent, exist_ok=True)
        config = save_yaml(self, pathlib.Path(path))
        return config
