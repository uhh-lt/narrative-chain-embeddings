import datetime
import os
from pathlib import Path
from typing import List, Optional

import fasttext
import numpy as np
import torch
import typer
from catalyst import dl
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from torch.utils.data import DataLoader

from eventy.bpemb import BPEmb
from eventy.callbacks.config import ConfigCallback
from eventy.callbacks.embedding import EmbeddingVisualizerCallback
from eventy.callbacks.multiple_choice import MultipleChoiceCallback
from eventy.callbacks.silhouette_score import SilhouetteScoreCallback
from eventy.config import Config, DatasetConfig
from eventy.dataset import ChainBatch, EventWindowDataset
from eventy.model import EventyModel
from eventy.runner import CustomRunner
from eventy.sampler import DynamicImbalancedDatasetSampler
from eventy.visualization import CustomConfusionMatrixCallback

app = typer.Typer()


STOP_EVENTS = ["sagen", "haben", "kommen"]


@app.command()
def test():
    pass


def build_vocabulary():
    vocabulary = []
    for line in open("top_lemmas.txt"):
        count, lemma = line.strip().split(" ")
        if int(count) < 1000:
            break
        new_lemma = lemma[1:-1]
        if new_lemma not in STOP_EVENTS:
            vocabulary.append(lemma[1:-1])  #  Strip the quotes at start and end
    return vocabulary


class EventPredictionSystem:
    def __init__(self, config_path="config.yaml", run_name: Optional[str] = None):
        self.run_name = run_name or str(datetime.datetime.utcnow())
        self.logdir = Path(f"./logs") / self.run_name
        self.config = self.load_config(Path(config_path))
        self.vocabulary = build_vocabulary()
        self.ft = fasttext.load_model(self.config.embedding_source.name)
        self.loaders = get_dataset(
            self.vocabulary,
            window_size=self.config.window_size,
            ft=self.ft,
            dataset_config=self.config.dataset,
            batch_size=self.config.batch_size,
        )
        self.distribution = EventWindowDataset.get_class_distribution(
            self.loaders["train"].dataset
        )
        self.model = self.init_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.runner = CustomRunner(
            class_distribution=self.distribution, losses=self.config.loss
        )
        self.mlflow_logger = dl.MLflowLogger(
            experiment="simple-eventy",
            run=self.run_name,
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        )
        self.wandb_logger = dl.WandbLogger("simple-event-predict", entity="hatzel")

    def get_baselines_results(self) -> str:
        return (
            f"Random chance accuracy: {1 / len(self.vocabulary)}\n"
            f"Majority baseline is: {self.distribution.max().item()}"
        )

    def init_model(self):
        model = EventyModel(
            output_vocab=len(self.vocabulary),
            num_inputs=self.config.window_size,
            class_distribution=self.distribution,
            vocab_embeddings=torch.tensor(
                np.stack([self.ft.get_word_vector(v) for v in self.vocabulary])
            ),
            dropout=self.config.model.dropout,
        )
        model.to(self.config.device)
        return model

    def train(self):
        self.runner.train(
            model=self.model,
            optimizer=self.optimizer,
            loaders=self.loaders,
            num_epochs=self.config.epochs,
            logdir=self.logdir,
            scheduler=torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                epochs=self.config.epochs,
                steps_per_epoch=len(self.loaders["train"]),
                max_lr=self.config.learning_rate,
            ),
            loggers={
                "console": dl.ConsoleLogger(),
                "mlflow": self.mlflow_logger,
                "wandb": self.wandb_logger,
                "tb": dl.TensorboardLogger(logdir=self.logdir),
            },
            engine=dl.GPUEngine(self.config.device)
            if "cuda" in self.config.device
            else dl.CPUEngine,
            callbacks=[
                ConfigCallback(
                    config=self.config, logdir=self.logdir, save_name="config.yaml"
                ),
                dl.EarlyStoppingCallback(
                    patience=5, metric_key="loss", minimize=True, loader_key="valid"
                ),
                EmbeddingVisualizerCallback(
                    label_key="labels",
                    embedding_key="new_embeddings",
                    collect_list=["gehen", "sterben", "feiern", "gewinnen"],
                    collection_frequency=0.1,
                    class_names=self.vocabulary,
                    prefix="embeddings",
                    loader_keys="valid",
                ),
                MultipleChoiceCallback(
                    n_choices=5,
                    input_key="logits",
                    target_key="labels",
                    distribution=self.distribution.to(self.config.device),
                ),
                SilhouetteScoreCallback(
                    input_key="new_embeddings",
                    target_key="labels",
                    sample_size=1000,
                ),
                CustomConfusionMatrixCallback(
                    input_key="logits",
                    target_key="labels",
                    num_classes=len(self.vocabulary),
                    class_names=self.vocabulary,
                    normalize=True,
                    show_numbers=False,
                ),
                *self.build_accuracy_callbacks(),
            ],
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )

    def build_accuracy_callbacks(self):
        return [
            AccuracyCallback(
                input_key="logits_thresholded",
                target_key="labels",
                num_classes=len(self.vocabulary),
                topk=(1, 3, 10),
            ),
            AccuracyCallback(
                input_key="logits",
                target_key="labels",
                num_classes=len(self.vocabulary),
                topk=(1, 3, 10),
                prefix="raw_",
            ),
            AccuracyCallback(
                input_key="cosine_similarities",
                target_key="labels",
                num_classes=len(self.vocabulary),
                topk=(1, 3, 10),
                prefix="cosine_",
            ),
        ]

    def load_config(self, config_path: Path):
        config = Config.load_file(config_path)
        return config

    def save_config(self):
        pass


@app.command()
def main():
    prediciton_system = EventPredictionSystem()
    print(prediciton_system.get_baselines_results())
    prediciton_system.train()


def get_dataset(vocabulary, window_size, ft, dataset_config: DatasetConfig, batch_size):
    dataset_train = EventWindowDataset(
        dataset_config.train_split,
        vocabulary=vocabulary,
        window_size=window_size,
        over_sampling=False,
        fast_text=ft,
    )
    dataset_dev = EventWindowDataset(
        dataset_config.validation_split,
        vocabulary=vocabulary,
        window_size=window_size,
        over_sampling=False,
        fast_text=ft,
    )
    print("Labels", dataset_train.get_label_counts())
    print("Labels relative", dataset_train.get_label_distribution())
    sampler = DynamicImbalancedDatasetSampler(
        dataset_train,
        labels=[item.lemmas[dataset_train.window_size // 2] for item in dataset_train],
        num_steps=100,
        sampling_schedule=dataset_config.sampling_schedule,
    )
    # sampler = BalanceBatchSampler([item.lemmas[dataset_train.window_size // 2] for item in dataset_train], 23, 2500)
    loader_train = DataLoader(
        dataset_train,
        collate_fn=lambda chains: ChainBatch.from_chains(chains, ft),
        batch_size=batch_size,
        sampler=sampler,
    )
    loader_dev = DataLoader(
        dataset_dev,
        collate_fn=lambda chains: ChainBatch.from_chains(chains, ft),
        batch_size=batch_size,
        shuffle=True,
    )
    return {
        "train": loader_train,
        "valid": loader_dev,
    }


# model training


@app.command()
def train():
    pass


if __name__ == "__main__":
    app()
