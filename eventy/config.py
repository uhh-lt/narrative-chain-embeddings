import os
import pathlib
from abc import ABC
from enum import Enum
from tkinter.tix import REAL
from typing import List, Optional, Union

from pydantic import BaseSettings, validator
from pydantic_loader.yaml_config import load_yaml, save_yaml


class SamplingSchedule(str, Enum):
    REAL_TO_BALANCED = "real_to_balanced"
    REAL = "real"
    BALANCED = "balanced"
    BALANCED_TO_REAL = "balanced_to_real"


class DatasetConfig(BaseSettings):
    test_split: str
    train_split: str
    validation_split: str
    sampling_schedule: SamplingSchedule


class LossKind(str, Enum):
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"


class ModelConfig(BaseSettings):
    dropout: float


class LoadableConfig(ABC):
    @staticmethod
    def load_file(path: Union[str, pathlib.Path]):
        pass

    def save_file(self, path: Union[str, pathlib.Path]):
        pass


class Config(BaseSettings, LoadableConfig):
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
