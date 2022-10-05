import dataclasses
import math
from sys import prefix

import torch
import typer
from catalyst import dl
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

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
    window_size = 7
    # vocabulary = ["rennen", "lesen", "hassen", "sehen", "stehen"]
    vocabulary = build_vocabulary()
    print("Random chance accuracy:", 1 / len(vocabulary))
    loaders = get_dataset(vocabulary, window_size=window_size)
    distribution = EventWindowDataset.get_class_distribution(loaders["train"].dataset)
    print("Majority baseline is:", distribution.max().item())
    model = EventyModel(
        output_vocab=len(vocabulary),
        num_inputs=window_size,
        class_distribution=distribution,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    runner = CustomRunner(class_distribution=distribution)
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=1,
        logdir="./logs",
        callbacks=[
            # this should be accessible on dl. but somehow isn't...
            # ConfusionMatrixCallback(
            #     input_key="logits",
            #     target_key="labels",
            #     num_classes=len(vocabulary),
            #     class_names=vocabulary,
            #     normalize=True,
            # )
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


def get_dataset(vocabulary, window_size):
    dataset_train = EventWindowDataset(
        "data/train_news_sample.jsonlines",
        vocabulary=vocabulary,
        window_size=window_size,
        over_sampling=False,
    )
    dataset_dev = EventWindowDataset(
        "data/dev_news_sample.jsonlines",
        vocabulary=vocabulary,
        window_size=window_size,
        over_sampling=False,
    )
    print("Labels", dataset_train.get_label_counts())
    print("Labels relative", dataset_train.get_label_distribution())
    sampler = ImbalancedDatasetSampler(
        dataset_train,
        labels=[item.lemmas[dataset_train.window_size // 2] for item in dataset_train],
    )
    loader_train = DataLoader(
        dataset_train,
        collate_fn=ChainBatch.from_chains,
        batch_size=8,
        sampler=sampler,
    )
    loader_dev = DataLoader(
        dataset_dev, collate_fn=ChainBatch.from_chains, batch_size=8, shuffle=True
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
