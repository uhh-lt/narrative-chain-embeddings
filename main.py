import dataclasses
import math

import torch
import typer
from catalyst import dl
from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback
from torch.utils.data import DataLoader

from eventy.dataset import ChainBatch, EventWindowDataset
from eventy.model import EventyModel
from eventy.runner import CustomRunner

app = typer.Typer()


@app.command()
def test():
    pass


def build_vocabulary():
    vocabulary = []
    for line in open("top_lemmas.txt"):
        count, lemma = line.strip().split(" ")
        if int(count) < 3000:
            break
        vocabulary.append(lemma[1:-1])  #  Strip the quotes at start and end
    return vocabulary


@app.command()
def main():
    window_size = 5
    vocabulary = ["rennen", "lesen", "hassen", "sehen", "stehen"]
    # vocabulary = build_vocabulary()
    print("Random chance accuracy:", 1 / len(vocabulary))
    model = EventyModel(output_vocab=len(vocabulary), num_inputs=window_size)
    loaders = get_dataset(vocabulary, window_size=window_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    runner = CustomRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=10,
        logdir="./logs",
        callbacks=[
            # this should be accessible on dl. but somehow isn't...
            ConfusionMatrixCallback(
                input_key="logits",
                target_key="labels",
                num_classes=len(vocabulary),
                class_names=vocabulary,
                normalize=True,
            )
        ],
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )


def get_dataset(vocabulary, window_size):
    dataset = EventWindowDataset(
        "/home/hansole/src/chain-extraction/news.jsonlines",
        vocabulary=vocabulary,
        window_size=window_size,
        balanced=True,
    )
    print("Labels", dataset.get_label_counts())
    print("Labels relative", dataset.get_label_distribution())
    train_size = math.floor(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_split, test_split = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    loader_train = DataLoader(
        train_split, collate_fn=ChainBatch.from_chains, batch_size=8, shuffle=True
    )
    loader_test = DataLoader(
        test_split, collate_fn=ChainBatch.from_chains, batch_size=8, shuffle=True
    )
    return {
        "train": loader_train,
        "valid": loader_test,
    }


# model training


@app.command()
def train():
    pass


if __name__ == "__main__":
    app()
