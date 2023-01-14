import datetime
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import fasttext
import numpy as np
import torch
import typer
from catalyst import dl
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import eventy
from eventy.bpemb import BPEmb
from eventy.callbacks.config import ConfigCallback
from eventy.callbacks.embedding import EmbeddingVisualizerCallback
from eventy.callbacks.multiple_choice import MultipleChoiceCallback
from eventy.callbacks.silhouette_score import SilhouetteScoreCallback
from eventy.config import Config, DatasetConfig, EmbeddingSourceKind
from eventy.dataset import ChainBatch, EventWindowDataset, SimilarityDataset
from eventy.model import EventyModel
from eventy.runner import CustomRunner
from eventy.sampler import DynamicImbalancedDatasetSampler
from eventy.visualization import CustomConfusionMatrixCallback

app = typer.Typer()


STOP_EVENTS = [
    "sagen",
    "haben",
    "kommen",
    "have",
    "be",
    "make",
    "get",
    "take",
    "know",
    "give",
    "tell",
    "see",
    "go",
]


def build_vocabulary(vocabulary_file, min_count):
    vocabulary = []
    for line in open(vocabulary_file):
        count, lemma = line.strip().split(" ")
        if int(count) < 1000:
            break
        new_lemma = lemma[1:-1]
        if new_lemma not in STOP_EVENTS:
            vocabulary.append(lemma[1:-1])  #  Strip the quotes at start and end
    return vocabulary


class EventPredictionSystem:
    def __init__(
        self,
        config_path="config.yaml",
        run_name: Optional[str] = None,
        quick_run: bool = False,
        splits: List[str] = ["train", "validation"],
        log: bool = True,
        device_override: Optional[str] = None,
    ):
        self.run_name = run_name or str(datetime.datetime.utcnow())
        self.logdir = Path(f"./logs") / self.run_name
        self.config = EventPredictionSystem.load_config(Path(config_path))
        if device_override is not None:
            self.config.device = device_override
        self.ft = self.get_embedder()
        self.vocabulary = build_vocabulary(
            self.config.dataset.vocabulary_file, self.config.dataset.min_count
        )
        self.loaders = get_dataset(
            self.vocabulary,
            window_size=self.config.window_size,
            ft=self.ft,
            dataset_config=self.config.dataset,
            batch_size=self.config.batch_size,
            size_limit=10000 if quick_run else None,
            splits=splits,
        )
        try:
            self.distribution = EventWindowDataset.get_class_distribution(
                self.loaders["train"].dataset
            )
        except KeyError:
            print(
                "Warning: no label distribution, regular evaluation will fail (fine if you are doing similarity evaluation)!"
            )
            self.distribution = torch.zeros(100)
        self.model = self.init_model()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.runner = CustomRunner(
            class_distribution=self.distribution, losses=self.config.loss
        )
        if log:
            self.wandb_logger = dl.WandbLogger("simple-event-predict", entity="hatzel")
            self.wandb_logger.log_hparams(self.config.to_one_level_dict())

    def get_baselines_results(self) -> str:
        return (
            f"Random chance accuracy: {1 / len(self.vocabulary)}\n"
            f"Majority baseline is: {self.distribution.max().item()}"
        )

    def get_embedder(self):
        if self.config.embedding_source.kind == EmbeddingSourceKind.FASTTEXT:
            return fasttext.load_model(self.config.embedding_source.name)
        elif self.config.embedding_source.kind == EmbeddingSourceKind.BPEMB:
            return BPEmb()
        else:
            raise ValueError(
                "Unkown Embedding source", self.config.embedding_source.kind
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
                "wandb": self.wandb_logger,
            },
            engine=dl.GPUEngine(self.config.device)
            if "cuda" in self.config.device
            else dl.CPUEngine,
            callbacks=[
                ConfigCallback(
                    config=self.config, logdir=self.logdir, save_name="config.yaml"
                ),
                dl.EarlyStoppingCallback(
                    patience=5,
                    metric_key="loss",
                    minimize=True,
                    loader_key="validation",
                ),
                EmbeddingVisualizerCallback(
                    label_key="labels",
                    embedding_key="new_embeddings",
                    collect_list=["die", "walk", "celebrate", "win"]
                    if "en" in self.config.dataset.vocabulary_file
                    else ["lesen", "gehen", "essen", "fahren"],
                    collection_frequency=0.1,
                    class_names=self.vocabulary,
                    prefix="embeddings",
                    loader_keys="validation",
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
            valid_loader="validation",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )

    def test(self):
        self.runner.evaluate_loader(
            loader=self.loaders["validation"],
            model=self.model,
            callbacks=[
                MultipleChoiceCallback(
                    n_choices=5,
                    input_key="logits",
                    target_key="labels",
                    distribution=self.distribution.to(self.config.device),
                ),
                *self.build_accuracy_callbacks(),
            ],
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

    @staticmethod
    def load_config(config_path: Path):
        config = Config.load_file(config_path)
        return config

    def save_config(self):
        pass


@app.command()
def train(quick_run: bool = False):
    prediciton_system = EventPredictionSystem(quick_run=quick_run)
    print(prediciton_system.get_baselines_results())
    prediciton_system.train()


@app.command()
def similarity(
    run_name: str,
    quick_run: bool = False,
    batch_size: int = 256,
    device: str = "cuda:0",
):
    prediciton_system = EventPredictionSystem(
        config_path=Path("logs") / run_name / "config.yaml",
        quick_run=False,
        splits=[],
        log=False,
        device_override=device,
    )
    dataset = SimilarityDataset(
        "data/similarity_chains_de.jsonlines",
        fast_text=prediciton_system.ft,
        window_size=prediciton_system.config.window_size,
        edge_markers=True,
        vocabulary=build_vocabulary(
            prediciton_system.config.dataset.vocabulary_file,
            prediciton_system.config.dataset.min_count,
        ),
    )
    loader = DataLoader(
        dataset,
        collate_fn=lambda chain_and_ids: (
            ChainBatch.from_chains(list(zip(*chain_and_ids))[0], prediciton_system.ft),
            list(zip(*chain_and_ids))[1],
        ),
        batch_size=batch_size,
    )
    embeddings = defaultdict(list)
    for batch, ids in loader:
        prediciton_system.model.eval()
        on_device_batch: ChainBatch = batch.to(device)
        model_output = prediciton_system.model(
            on_device_batch.embeddings,
            on_device_batch.subject_hot_encodings,
            on_device_batch.object_hot_encodings,
            on_device_batch.labels,
            on_device_batch.label_embeddings,
            on_device_batch.object_embeddings,
            on_device_batch.subject_embeddings,
        )
        for doc_id, embedding in zip(ids, model_output.embeddings):
            embeddings[doc_id].append(embedding)
    predicted_sims = []
    all_similarities = defaultdict(list)
    for (doc_a, doc_b), similarities in dataset.similarities.items():
        if len(embeddings[doc_a]) == 0 or len(embeddings[doc_b]) == 0:
            print("skipping")
            continue
        doc_a_embeddings = torch.stack(embeddings[doc_a])
        doc_b_embeddings = torch.stack(embeddings[doc_b])
        sims = eventy.util.cosine_similarity(doc_a_embeddings, doc_b_embeddings)
        predicted_sims.append(sims.max(0).values.mean() + sims.max(1).values.mean())
        for k, v in similarities.items():
            all_similarities[k].append(v)

    for dimension, sim_list in all_similarities.items():
        corr = torch.corrcoef(
            torch.stack([torch.tensor(sim_list), torch.tensor(predicted_sims)])
        )
        print(f"Correlation for {dimension} is {corr[0][1].item():.2f}")


@app.command()
def test(run_name: str, test_set: bool = False):
    prediciton_system = EventPredictionSystem(
        config_path=Path("logs") / run_name / "config.yaml",
        quick_run=True,
        splits=(["test"] if test_set else ["validation"]) + ["train"],
        log=False,
    )
    state_dict = torch.load(Path("logs") / run_name / "checkpoints" / "model.best.pth")
    prediciton_system.model.load_state_dict(state_dict)
    prediciton_system.test()
    # TODO: we need to load the model and the test set here + optionally the dev set instead (actually dev should be default)


@app.command()
def get_stats(config_path: str = "config.yaml"):
    config = EventPredictionSystem.load_config(Path(config_path))
    counter = Counter()
    total_length = 0
    num_chains = 0
    for line in open(config.dataset.train_split):
        data = json.loads(line)
        counter.update([len(data)])
        total_length += len(data)
        num_chains += 1
    plt.bar(range(10), [counter.get(x, 0) for x in range(10)])
    plt.savefig("chain_lengths.pdf")
    print(f"Total number of chains: {num_chains}")
    print(f"Mean chain length: {total_length / num_chains:.2f}")


def get_dataset(
    vocabulary,
    window_size,
    ft,
    dataset_config: DatasetConfig,
    batch_size,
    size_limit: Optional[int] = None,
    splits: List[str] = ["train", "validation"],
):
    loaders = {}
    for split in splits:
        dataset = EventWindowDataset(
            getattr(dataset_config, split + "_split"),
            vocabulary=vocabulary,
            window_size=window_size,
            over_sampling=False,
            edge_markers=dataset_config.edge_markers,
            fast_text=ft,
            size_limit=size_limit,
        )
        if split == "train":
            print("Labels", dataset.get_label_counts())
            print("Labels relative", dataset.get_label_distribution())
        sampler = None
        if split == "train":
            sampler = DynamicImbalancedDatasetSampler(
                dataset,
                labels=[item.lemmas[dataset.window_size // 2] for item in dataset],
                num_steps=100,
                sampling_schedule=dataset_config.sampling_schedule,
            )
        loaders[split] = DataLoader(
            dataset,
            collate_fn=lambda chains: ChainBatch.from_chains(chains, ft),
            batch_size=batch_size,
            sampler=sampler,
        )
    return loaders


if __name__ == "__main__":
    app()
