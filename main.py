import datetime
import os

import fasttext
import numpy as np
import torch
import typer
from catalyst import dl
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from eventy.callbacks.embedding import EmbeddingVisualizerCallback
from eventy.config import Config, DatasetConfig
from eventy.dataset import ChainBatch, EventWindowDataset
from eventy.model import EventyModel
from eventy.runner import CustomRunner

app = typer.Typer()


STOP_EVENTS = ["sagen", "haben", "kommen"]


@app.command()
def test():
    pass


def build_vocabulary():
    vocabulary = []
    for line in open("top_lemmas.txt"):
        count, lemma = line.strip().split(" ")
        if int(count) < 10000:
            break
        new_lemma = lemma[1:-1]
        if new_lemma not in STOP_EVENTS:
            vocabulary.append(lemma[1:-1])  #  Strip the quotes at start and end
    return vocabulary


@app.command()
def main():
    config = Config.load_file("config.yaml")
    run_name = str(datetime.datetime.utcnow())
    mlflow_logger = dl.MLflowLogger(
        experiment="simple-eventy",
        run=run_name,
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
    )
    # vocabulary = ["rennen", "lesen", "hassen", "sehen", "stehen"]
    vocabulary = build_vocabulary()
    print("Random chance accuracy:", 1 / len(vocabulary))
    ft = fasttext.load_model("cc.de.300.bin")
    loaders = get_dataset(
        vocabulary,
        window_size=config.window_size,
        ft=ft,
        dataset_config=config.dataset,
        batch_size=config.batch_size,
    )
    distribution = EventWindowDataset.get_class_distribution(loaders["train"].dataset)
    print("Majority baseline is:", distribution.max().item())
    model = EventyModel(
        output_vocab=len(vocabulary),
        num_inputs=config.window_size,
        class_distribution=distribution,
        vocab_embeddings=torch.tensor(
            np.stack([ft.get_word_vector(v) for v in vocabulary])
        ),
        dropout=config.model.dropout,
    )
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    runner = CustomRunner(class_distribution=distribution, losses=config.loss)
    logdir = f"./logs/{run_name}"
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=config.epochs,
        logdir=logdir,
        hparams=config.dict(),
        loggers={
            "console": dl.ConsoleLogger(),
            "mlflow": mlflow_logger,
            "tb": dl.TensorboardLogger(logdir=logdir),
        },
        engine=dl.GPUEngine(config.device) if "cuda" in config.device else dl.CPUEngine,
        callbacks=[
            dl.EarlyStoppingCallback(
                patience=5, metric_key="loss", minimize=True, loader_key="valid"
            ),
            EmbeddingVisualizerCallback(
                label_key="labels",
                embedding_key="new_embeddings",
                collect_list=["gehen", "sterben", "feiern", "gewinnen"],
                collection_frequency=0.05,
                class_names=vocabulary,
                prefix="embeddings",
            ),
            # this should be accessible on dl. but somehow isn't...
            ConfusionMatrixCallback(
                input_key="logits",
                target_key="labels",
                num_classes=len(vocabulary),
                class_names=vocabulary,
                normalize=True,
            ),
            AccuracyCallback(
                input_key="logits_thresholded",
                target_key="labels",
                num_classes=len(vocabulary),
                topk=(1, 3, 10),
            ),
            AccuracyCallback(
                input_key="logits",
                target_key="labels",
                num_classes=len(vocabulary),
                topk=(1, 3, 10),
                prefix="raw_",
            ),
        ],
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )


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
    sampler = ImbalancedDatasetSampler(
        dataset_train,
        labels=[item.lemmas[dataset_train.window_size // 2] for item in dataset_train],
    )
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
