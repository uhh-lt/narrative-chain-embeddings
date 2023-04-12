import datetime
import json
import os
import pickle
from collections import Counter, defaultdict
from csv import DictWriter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import typer
import yaml
from catalyst import dl
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import eventy
import wandb
from eventy import interactive
from eventy.bpemb import BPEmb
from eventy.callbacks.config import ConfigCallback
from eventy.callbacks.early_stopping import LoggingEarlyStopper
from eventy.callbacks.embedding import EmbeddingVisualizerCallback
from eventy.callbacks.multiple_choice import MultipleChoiceCallback
from eventy.callbacks.silhouette_score import SilhouetteScoreCallback
from eventy.config import Config, DatasetConfig, EmbeddingSourceKind, ModelKind
from eventy.dataset import (
    Chain,
    ChainBatch,
    Event,
    EventWindowDataset,
    SimilarityDataset,
    get_windows,
)
from eventy.fasttext_wrapper import FasttextWrapper
from eventy.model import EventyModel
from eventy.muse import MuseText
from eventy.runner import CustomRunner
from eventy.sampler import DynamicImbalancedDatasetSampler
from eventy.transformer_model import EventyTransformerModel
from eventy.visualization import CustomConfusionMatrixCallback

app = typer.Typer()

WANDB_PROJECT_NAME = "eventy"
WANDB_ENTITY = "hatzel"
global_loader_cache = None


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

DE_STOP_EVENTS = [
    "sagen",
    "haben",
    "geben",
    "kommen",
    "machen",
    "stehen",
    "erklären",
    "sehen",
    "gehen",
    "lassen",
]


def build_vocabulary(vocabulary_file, min_count, lang="en"):
    vocabulary = []
    for line in open(vocabulary_file):
        count, lemma = line.strip().split(" ")
        if int(count) < min_count:
            break
        new_lemma = lemma[1:-1]
        if new_lemma not in [STOP_EVENTS if lang != "de" else DE_STOP_EVENTS]:
            vocabulary.append(lemma[1:-1])  #  Strip the quotes at start and end
    return vocabulary


class EventPredictionSystem:
    def __init__(
        self,
        config_path: Optional[str] = "config.yaml",
        run_name: Optional[str] = None,
        quick_run: bool = False,
        splits: List[str] = ["train", "validation"],
        log: bool = True,
        device_override: Optional[str] = None,
        overrides: Dict[str, any] = {},
        loaders: Dict = None,
        shuffle_chains: bool = False,
        deduplicate: bool = True,
    ):
        self.wandb_logger = None
        if log:
            self.wandb_logger = dl.WandbLogger(WANDB_PROJECT_NAME, entity=WANDB_ENTITY)
        self.run_name = (
            run_name
            or (self.wandb_logger.run.name if self.wandb_logger is not None else None)
            or str(datetime.datetime.utcnow())
        )
        self.logdir = Path(f"./logs") / self.run_name
        if config_path:
            self.config = EventPredictionSystem.load_config(Path(config_path))
        else:
            self.config = Config(**wandb.config)
        if log:
            self.wandb_logger.run.config.update(self.config, allow_val_change=True)
        self.set_overrides(overrides)
        if device_override is not None:
            self.config.device = device_override
        self.ft = self.get_embedder()
        self.vocabulary = build_vocabulary(
            self.config.dataset.vocabulary_file,
            self.config.dataset.min_count,
            self.config.dataset.lang,
        )
        self.loaders = loaders or get_dataset(
            self.vocabulary,
            window_size=self.config.window_size,
            ft=self.ft,
            dataset_config=self.config.dataset,
            batch_size=self.config.batch_size,
            size_limit=100_000 if quick_run else None,
            splits=splits,
            shuffle_chains=shuffle_chains,
            deduplicate=deduplicate,
        )
        if os.path.exists(
            cache_path := self.config.dataset.train_split
            + f"min_conut={self.config.dataset.min_count}_quick_run={quick_run}.cache"
        ):
            self.distribution = pickle.load(open(cache_path, "rb"))
        else:
            try:
                self.distribution = EventWindowDataset.get_class_distribution(
                    self.loaders["train"].dataset
                )
                if not quick_run:
                    pickle.dump(self.distribution, open(cache_path, "wb"))
            except KeyError:
                print(
                    "Warning: no label distribution, regular evaluation will fail (fine if you are doing similarity evaluation)!"
                )
                self.distribution = torch.zeros(100)
        self.model = self.init_model(self.config.model.kind)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.runner = CustomRunner(
            class_distribution=self.distribution, losses=self.config.loss
        )

    def set_overrides(self, overrides: Dict[str, any]):
        for override_path, value in overrides.items():
            elements = override_path.split(".")
            sub_config = self.config
            for el in elements[:-1]:
                sub_config = getattr(sub_config, el)
            setattr(sub_config, elements[-1], ast.literal_eval(value))

    def get_baselines_results(self) -> str:
        return (
            f"Random chance accuracy: {1 / len(self.vocabulary)}\n"
            f"Majority baseline is: {self.distribution.max().item()}"
        )

    def get_embedder(self):
        if self.config.embedding_source.kind == EmbeddingSourceKind.FASTTEXT:
            return FasttextWrapper(self.config.embedding_source.name)
        elif self.config.embedding_source.kind == EmbeddingSourceKind.BPEMB:
            return BPEmb()
        elif self.config.embedding_source.kind == EmbeddingSourceKind.MUSE:
            return MuseText(self.config.embedding_source.name)
        else:
            raise ValueError(
                "Unkown Embedding source", self.config.embedding_source.kind
            )

    def init_model(self, kind):
        model_conf = {k: v for k, v in dict(self.config.model).items() if k != "kind"}
        if kind == ModelKind.TRANSFORMER:
            model = EventyTransformerModel(
                output_vocab=len(self.vocabulary),
                num_inputs=self.config.window_size,
                class_distribution=self.distribution,
                vocab_embeddings=torch.tensor(
                    np.stack([self.ft.get_word_vector(v) for v in self.vocabulary])
                ),
                **model_conf,
            )
        if kind == ModelKind.FFNN:
            model = EventyModel(
                output_vocab=len(self.vocabulary),
                num_inputs=self.config.window_size,
                class_distribution=self.distribution,
                vocab_embeddings=torch.tensor(
                    np.stack([self.ft.get_word_vector(v) for v in self.vocabulary])
                ),
                **model_conf,
            )
        model.to(self.config.device)
        return model

    def train(self):
        early_stopper = LoggingEarlyStopper(
            patience=5,
            metric_key="5_choice_accuracy01",
            minimize=False,
            loader_key="validation",
            logger=self.wandb_logger,
        )
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
                div_factor=20,
                final_div_factor=100,
                pct_start=0.3,
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
                early_stopper,
                MultipleChoiceCallback(
                    n_choices=5,
                    input_key="logits",
                    target_key="labels",
                    distribution=self.distribution.to(self.config.device),
                    loader_keys=["validation", "test"],
                ),
                SilhouetteScoreCallback(
                    input_key="new_embeddings",
                    target_key="labels",
                    sample_size=10_000,
                ),
                *(
                    self.build_visualization_callbacks()
                    if self.config.visualizations
                    else []
                ),
                *self.build_accuracy_callbacks(),
            ],
            valid_loader="validation",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
        return early_stopper.best_score

    def build_visualization_callbacks(self):
        return [
            EmbeddingVisualizerCallback(
                label_key="labels",
                embedding_key="new_embeddings",
                collect_list=["die", "see", "hear", "walk", "run", "sneak"]
                if "gigaword" in self.config.dataset.vocabulary_file
                else ["lesen", "gehen", "essen", "fahren"],
                collection_frequency=1.0,
                class_names=self.vocabulary,
                out_path="embeddings.pdf",
                prefix="embeddings",
                loader_keys=["valid"],
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
        ]

    def test(self, test=False):
        self.runner.evaluate_loader(
            loader=self.loaders["validation"] if not test else self.loaders["test"],
            model=self.model,
            engine=dl.GPUEngine("cuda:0"),
            callbacks=[
                MultipleChoiceCallback(
                    n_choices=5,
                    input_key="logits",
                    target_key="labels",
                    distribution=self.distribution.to(self.config.device),
                    loader_keys=["valid", "test"],
                ),
                *self.build_visualization_callbacks(),
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
    def load_config(config_path: Path) -> Config:
        config = Config.load_file(config_path)
        return config

    def save_config(self):
        pass


@app.command()
def train(quick_run: bool = False, overrides: List[str] = []):
    prediction_system = EventPredictionSystem(
        quick_run=quick_run,
        overrides=dict([o.split("=") for o in overrides]),
    )
    print(prediction_system.get_baselines_results())
    prediction_system.train()


@app.command()
def init_sweep(config_path: str = "sweep.yaml"):
    sweepconfig = yaml.load(open(config_path), Loader=yaml.CLoader)
    wandb.sweep(sweepconfig, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY)


@app.command()
def join_sweep(sweep_id, quick_run: bool = False):
    wandb.agent(sweep_id, project=WANDB_PROJECT_NAME, function=lambda: sweep(quick_run))


def sweep(quick_run):
    global global_loader_cache
    if global_loader_cache is not None:
        prediction_system = EventPredictionSystem(
            config_path=None, loaders=global_loader_cache, quick_run=quick_run
        )
    else:
        prediction_system = EventPredictionSystem(config_path=None, quick_run=quick_run)
        global_loader_cache = prediction_system.loaders
    prediction_system.train()


@app.command("fasttext-chains")
def fasttext_embedding_baseline(
    json_path: str = "data/semeval_eval_en.jsonlines",
    embedding_src: str = "cc.en.300.bin",
):
    # The longer the chain, the closer values are to 0.5 when random
    # long chains are potentially more similary
    torch.manual_seed(42)

    class RandomEmbedder:
        def __init__(self):
            pass

        def get_word_vector(self, _lemma):
            return torch.rand(300, dtype=torch.float)

    embedder = FasttextWrapper(embedding_src)
    # embedder = RandomEmbedder()
    gold_sims = defaultdict(list)
    predicted_sims = []
    predicted_sims_matrix = []
    predicted_sims_words = []
    key = "verb_lemma"
    for line in open(json_path):
        data = json.loads(line)
        embeddings_1 = []
        embeddings_2 = []
        word_embeddings_1 = []
        word_embeddings_2 = []
        if len(data["chains_1"]) == 0 or len(data["chains_2"]) == 0:
            continue
        embedding_1 = torch.zeros(300, dtype=torch.float)
        embedding_2 = torch.zeros(300, dtype=torch.float)
        for chain in data["chains_1"]:
            chain_embedding = torch.stack(
                [embedder.get_word_vector(e[key]) for e in chain]
            ).mean(0)
            word_embeddings_1.extend([embedder.get_word_vector(e[key]) for e in chain])
            embeddings_1.append(chain_embedding)
            embedding_1 += chain_embedding
        for chain in data["chains_2"]:
            chain_embedding = torch.stack(
                [embedder.get_word_vector(e[key]) for e in chain]
            ).mean(0)
            word_embeddings_2.extend([embedder.get_word_vector(e[key]) for e in chain])
            embeddings_2.append(chain_embedding)
            embedding_2 += chain_embedding
        embedding_1 /= len(embeddings_1)
        embedding_2 /= len(embeddings_2)
        sims = eventy.util.cosine_similarity(
            torch.stack(embeddings_1), torch.stack(embeddings_2)
        )
        sims.fill_diagonal_(0)
        predicted_sims_matrix.append(
            1 - (sims.max(0).values.mean() + sims.max(1).values.mean()) / 2
        )
        predicted_sims.append(
            torch.nn.functional.cosine_similarity(
                embedding_1.unsqueeze(0), embedding_2.unsqueeze(0)
            )
        )
        sims_words = eventy.util.cosine_similarity(
            torch.stack(word_embeddings_1), torch.stack(word_embeddings_2)
        )
        sims_words.fill_diagonal_(0)
        predicted_sims_words.append(
            1 - (sims_words.max(0).values.mean() + sims_words.max(1).values.mean()) / 2
        )
        for dim, sim in data["similarities"].items():
            gold_sims[dim].append(sim)
    for dimension, sim_list in gold_sims.items():
        corr_data = torch.stack(
            [torch.tensor(sim_list), torch.tensor(predicted_sims_words)]
        )
        corr = torch.corrcoef(corr_data)
        print(f"Correlation (word matrix) for {dimension} is {corr[0][1].item():.2f}")
    for dimension, sim_list in gold_sims.items():
        corr_data = torch.stack([torch.tensor(sim_list), torch.tensor(predicted_sims)])
        corr = torch.corrcoef(corr_data)
        print(f"Correlation for {dimension} is {corr[0][1].item():.2f}")
    for dimension, sim_list in gold_sims.items():
        corr_data_matrix = torch.stack(
            [torch.tensor(sim_list), torch.tensor(predicted_sims_matrix)]
        )
        corr_matrix = torch.corrcoef(corr_data_matrix)
        print(
            f"Correlation (matrix method) for {dimension} is {corr_matrix[0][1].item():.2f}"
        )


@app.command()
def similarity(
    run_name: str,
    out_file: str,
    quick_run: bool = False,
    batch_size: int = 256,
    device: str = "cuda:0",
    shuffle_chains: bool = False,
    remove_entities: bool = False,
):
    prediction_system = EventPredictionSystem(
        config_path=Path("logs") / run_name / "config.yaml",
        quick_run=quick_run,
        splits=["train"],
        log=False,
        device_override=device,
    )
    state_dict = torch.load(Path("logs") / run_name / "checkpoints" / "model.best.pth")
    prediction_system.model.load_state_dict(state_dict)
    dataset = SimilarityDataset(
        "data/semeval_eval_en.jsonlines",
        fast_text=prediction_system.ft,
        window_size=prediction_system.config.window_size,
        edge_markers=True,
        vocabulary=build_vocabulary(
            prediction_system.config.dataset.vocabulary_file,
            prediction_system.config.dataset.min_count,
        ),
        shuffle_chains=shuffle_chains,
        remove_entities=remove_entities,
    )
    loader = DataLoader(
        dataset,
        collate_fn=lambda chain_and_ids: (
            ChainBatch.from_chains(list(zip(*chain_and_ids))[0], prediction_system.ft),
            list(zip(*chain_and_ids))[1],
        ),
        batch_size=batch_size,
    )
    window_embeddings = {}
    i = 0
    for batch, ids in loader:
        prediction_system.model.eval()
        on_device_batch: ChainBatch = batch.to(device)
        model_output = prediction_system.model(
            on_device_batch.embeddings,
            on_device_batch.subject_hot_encodings,
            on_device_batch.object_hot_encodings,
            on_device_batch.labels,
            on_device_batch.label_embeddings,
            on_device_batch.object_embeddings,
            on_device_batch.subject_embeddings,
            on_device_batch.iobject_embeddings,
        )
        for full_chain_id, embedding in zip(ids, model_output.embeddings):
            window_embeddings[full_chain_id] = window_embeddings.get(
                full_chain_id, []
            ) + [embedding.detach().cpu()]
            i += 1
    per_chain_embeddings = defaultdict(list)
    for k, v in window_embeddings.items():
        doc_id, chain_id = k.split("_")
        per_chain_embeddings[doc_id].append(torch.stack(v).mean(0))
    doc_embeddings = {
        k: torch.stack(chain_embeddings).mean(0)
        for k, chain_embeddings in per_chain_embeddings.items()
    }
    all_similarities = defaultdict(list)
    for (doc_a, doc_b), similarities in dataset.similarities.items():
        for k, v in similarities.items():
            all_similarities[k].append(v)
    predicted_sims = []
    predicted_sims_matrix_match = []
    default_emb = torch.zeros(300)
    for (doc_a, doc_b), _ in dataset.similarities.items():
        predicted_sims.append(
            torch.nn.functional.cosine_similarity(
                doc_embeddings.get(doc_a, default_emb).unsqueeze(0),
                doc_embeddings.get(doc_b, default_emb).unsqueeze(0),
            )
        )
        try:
            sims = eventy.util.cosine_similarity(
                torch.stack(per_chain_embeddings.get(doc_a, [default_emb])),
                torch.stack(per_chain_embeddings.get(doc_b, [default_emb])),
            )
            mean = sims.max(0).values.mean() + sims.max(1).values.mean()
            predicted_sims_matrix_match.append(mean)
        except KeyError:
            predicted_sims_matrix_match.append(torch.tensor(0.0))
    # print(
    #     f"Skipped {skipped} documents out of {total} that's {skipped / total * 100:.2f}%"
    # )
    print([t for t in predicted_sims[:10]])

    writer = DictWriter(
        open(out_file, "w"),
        fieldnames=[
            "strategy",
            "geography",
            "entities",
            "time",
            "narrative",
            "overall",
            "style",
            "tone",
        ],
    )
    correlations = defaultdict(dict)
    writer.writeheader()
    for dimension, sim_list in all_similarities.items():
        corr = torch.corrcoef(
            torch.stack([torch.tensor(sim_list), torch.tensor(predicted_sims)])
        )
        corr_matrix = torch.corrcoef(
            torch.stack(
                [torch.tensor(sim_list), torch.tensor(predicted_sims_matrix_match)]
            )
        )
        correlations["matrix"][dimension] = f"{corr_matrix[0][1].item():.3f}"
        correlations["plain"][dimension] = f"{corr[0][1].item():.3f}"
        print(f"Correlation for {dimension} is {corr[0][1].item():.2f}")
        print(
            f"Matrix-based correlation for {dimension} is {corr_matrix[0][1].item():.2f}"
        )
    for dim, data in correlations.items():
        data.update({"strategy": dim})
        writer.writerow(data)


@app.command()
def test(run_name: str, test_set: bool = False, quick_run: bool = False):
    # prediction_system = EventPredictionSystem(
    #     config_path=Path("logs") / run_name / "config.yaml",
    #     splits=(["test"] if test_set else ["validation"]),
    #     quick_run=quick_run,
    #     log=False,
    # )
    prediction_system = EventPredictionSystem(
        config_path=Path("logs") / run_name / "config.yaml",
        quick_run=quick_run,
        splits=["train"] + (["test"] if test_set else ["validation"]),
        log=False,
        deduplicate=False,
    )
    state_dict = torch.load(Path("logs") / run_name / "checkpoints" / "model.best.pth")
    prediction_system.model.load_state_dict(state_dict)
    prediction_system.test(test_set)


@app.command()
def get_stats(
    config_paths: List[str] = [
        "visualization_configs/config_de.yaml",
        "visualization_configs/config_en.yaml",
    ],
    names: List[str] = ["German", "English-NYT"],
    lower_bound: int = 3,
):
    offset = -0.1
    colors = plt.cm.Set1(range(2))
    upper_bound = 10
    out_csv = open("chain_lengths.csv", "w")
    counts = []
    for config_path, name, color in zip(config_paths, names, colors):
        config = EventPredictionSystem.load_config(Path(config_path))
        counter = Counter()
        total_length = 0
        num_chains = 0
        for line in open(config.dataset.train_split):
            data = json.loads(line)
            if not isinstance(data, list):
                data = data.get("chain", [])
            if len(data) >= lower_bound:
                counter.update([len(data)])
                total_length += len(data)
                num_chains += 1
            # if num_chains > 1000:
            #     break
        normalized_counter = {k: v / sum(counter.values()) for k, v in counter.items()}
        print(f"Total number of chains in {name}: {num_chains}")
        print(f"Mean chain length in {name}: {total_length / num_chains:.2f}")
        plt.bar(
            [x + offset for x in range(lower_bound, upper_bound)],
            [normalized_counter.get(x, 0) for x in range(lower_bound, upper_bound)],
            0.4,
            label=name,
            color=color,
        )
        counts.append(
            [name]
            + [normalized_counter.get(x, 0) for x in range(lower_bound, upper_bound)]
        )
        offset += 0.2
        plt.ylabel("Relative frequency")
        plt.ylabel("Chain length")
    for line in zip(["dataset"] + list(range(lower_bound, upper_bound)), *counts):
        out_csv.write(",".join([str(e) for e in line]) + "\n")
    plt.legend()
    plt.savefig("chain_lengths.pdf")


def get_dataset(
    vocabulary,
    window_size,
    ft,
    dataset_config: DatasetConfig,
    batch_size,
    size_limit: Optional[int] = None,
    splits: List[str] = ["train", "validation"],
    debug_log: bool = True,
    shuffle_chains: bool = False,
    deduplicate: bool = True,
):
    loaders = {}
    for split in splits:
        dataset = EventWindowDataset(
            getattr(dataset_config, split + "_split"),
            vocabulary=vocabulary,
            window_size=window_size,
            over_sampling=False,
            edge_markers=dataset_config.edge_markers if split == "train" else False,
            fast_text=ft,
            min_chain_len=None if split == "train" else 9,
            size_limit=size_limit,
            shuffle_chains=shuffle_chains,
            deduplicate=deduplicate,
        )
        if split == "train" and debug_log:
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
            num_workers=6,
        )
    return loaders


@app.command()
def interactive_chains(run_name: str):
    prediction_system = EventPredictionSystem(
        config_path=Path("logs") / run_name / "config.yaml",
        quick_run=True,
        splits=["train"],
        log=False,
    )
    state_dict = torch.load(Path("logs") / run_name / "checkpoints" / "model.best.pth")
    prediction_system.model.load_state_dict(state_dict)
    input_text = """(A, search, B)
(A, arrest, B)
(B, [find, call, help, die, grow, plead], C)
(D, sentence, B)"""
    prediction_system.model.eval()
    input_chains = interactive.read_editor_input(input_text)
    while True:
        input_chain, input_text = next(input_chains)
        choices = []
        events = []
        window_size = 7
        vocab_set = set(prediction_system.vocabulary) | set(["_"])
        for triple in input_chain:
            subjs, verb, objs = triple
            if isinstance(verb, list):
                choices = verb
                verb = "_"
            events.append(
                Event(
                    subjects=subjs,
                    lemma=verb,
                    objects=objs,
                    iobjs=None,
                    subject_names=subjs,
                    object_names=objs,
                )
            )
        chain = EventWindowDataset.add_edge_markers(events, window_size - 1)
        windows = get_windows(chain, window_size, vocab_set)

        model_chains = []
        for window in windows:
            try:
                to_mask = [i for i, e in enumerate(window) if e.lemma == "_"][0]
            except IndexError:
                continue
            if to_mask == (window_size // 2):
                new_chain = Chain.from_events(
                    windows[0],
                    prediction_system.ft,
                    to_mask,
                    prediction_system.vocabulary,
                )
                model_chains.append(new_chain)
        print("Window:", window)
        try:
            batch = ChainBatch.from_chains(model_chains, prediction_system.ft).to(
                "cuda:0"
            )
        except ValueError as e:
            print("Error", e)
            continue
        model_output = prediction_system.model(
            batch.embeddings,
            batch.subject_hot_encodings,
            batch.object_hot_encodings,
            batch.labels,
            batch.label_embeddings,
            batch.object_embeddings,
            batch.subject_embeddings,
            batch.iobject_embeddings,
        )
        if len(choices) == 0:
            top_k = model_output.logits.topk(5)
            print([prediction_system.vocabulary[k] for k in top_k.indices[0]])
        else:
            indices = [prediction_system.vocabulary.index(v) for v in choices]
            choices_with_score = zip(
                [model_output.logits[0][i].item() for i in indices], choices
            )
            sorted_choices_with_score = list(
                sorted(choices_with_score, key=lambda pair: pair[0], reverse=True)
            )
            print(sorted_choices_with_score)
        input("Press enter to edit text again")


if __name__ == "__main__":
    app()
