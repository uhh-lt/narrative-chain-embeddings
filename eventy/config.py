import os
import pathlib
from abc import ABC
from ast import Dict
from enum import Enum
from tkinter.tix import REAL
from typing import List, Optional, Union

from pydantic import BaseSettings, validator
from pydantic.utils import deep_update
from pydantic_loader.yaml_config import load_yaml, save_yaml

import wandb


class SamplingSchedule(str, Enum):
    REAL_TO_BALANCED = "real_to_balanced"
    REAL = "real"
    BALANCED = "balanced"
    BALANCED_TO_REAL = "balanced_to_real"


class EmbeddingSourceKind(str, Enum):
    FASTTEXT = "fasttext"
    BPEMB = "bpemb"
    MUSE = "muse"


class EmbeddingSource(BaseSettings):
    kind: EmbeddingSourceKind
    name: str


class DatasetConfig(BaseSettings):
    test_split: str
    train_split: str
    validation_split: str
    edge_markers: bool
    sampling_schedule: SamplingSchedule
    vocabulary_file: str
    min_count: int
    lang: str = "en"


class LossKind(str, Enum):
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    EUCLIDEAN = "euclidean"


class ModelKind(str, Enum):
    FFNN = "ffnn"
    TRANSFORMER = "transformer"


class ModelConfig(BaseSettings):
    kind: ModelKind = ModelKind.FFNN
    dropout: float
    model_characters: bool = True
    model_character_names: bool = True


class LoadableConfig(ABC):
    @staticmethod
    def load_file(path: Union[str, pathlib.Path]):
        pass

    def save_file(self, path: Union[str, pathlib.Path]):
        pass


class Config(BaseSettings, LoadableConfig):
    device: str
    epochs: int = 1
    # Wheter to log images of confusion matrices etc
    visualizations: bool = False
    batch_size: int
    window_size: int
    learning_rate: float
    dataset: DatasetConfig
    loss: List[LossKind]
    model: ModelConfig
    embedding_source: EmbeddingSource

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

    def to_one_level_dict(self):
        data = dict(self)

        def unnested(data_in, prefix=""):
            out = {
                prefix + k: v
                for k, v in data_in.items()
                if not isinstance(v, BaseSettings)
            }
            update = [
                unnested(dict(value), prefix=key + ".")
                for key, value in data_in.items()
                if isinstance(value, BaseSettings)
            ]
            for up in update:
                while len(set(up.keys()) & set(out.keys())) > 0:
                    up = {
                        "__duplicate__" + k if k in out.keys() else k: v
                        for k, v in up.items()
                    }
                out.update(up)
            return out

        return unnested(data)

    def update(self, config: wandb.Config):
        d = deep_update(dict(self), config.as_dict())
        return Config(**d)
