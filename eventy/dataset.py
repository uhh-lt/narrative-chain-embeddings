import dataclasses
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import fasttext
import numpy as np
import torch
from torch.utils.data import Dataset

PREDICTABLE_LEMMAS = ["schlagen", "warten", "laufen", "gehen"]


@dataclass
class Event:
    lemma: str
    objects: List[str]
    subjects: List[str]
    object_names: List[str]
    subject_names: List[str]

    @classmethod
    def from_json(cls, data: Dict) -> "Event":
        return Event(
            lemma=data["verb_lemma"],
            objects=data["objects"],
            subjects=data["subjects"],
            subject_names=data.get("subject_names", []),
            object_names=data.get("object_names", []),
        )

    def __hash__(self):
        return hash(self.lemma)


@dataclass
class Chain:
    lemmas: List[str]
    embeddings: List[np.array]
    label: int
    subject_hot_encodings: List[torch.Tensor]
    object_hot_encodings: List[torch.Tensor]
    subject_names: List[str]
    object_names: List[str]
    subject_embeddings: List[np.array]
    object_embeddings: List[np.array]

    def __init__(
        self,
        lemmas,
        embeddings,
        vocabulary,
        subject_hot_encodings,
        object_hot_encodings,
        subject_names,
        object_names,
        subject_embeddings,
        object_embeddings,
    ):
        self.subject_hot_encodings = subject_hot_encodings
        self.object_hot_encodings = object_hot_encodings
        self.lemmas = lemmas
        self.embeddings = embeddings
        self.subject_names = subject_names
        self.object_names = object_names
        self.subject_embeddings = subject_embeddings
        self.object_embeddings = object_embeddings
        self.label = vocabulary.index(self.lemmas[len(self.lemmas) // 2])

    def get_central_lemma(self) -> str:
        if len(self.lemmas) % 2 != 1:
            print("Warning: even length chains may not be properly supported")
        return self.lemmas[len(self.lemmas) // 2]


@dataclass
class ChainBatch:
    # input_embeddings
    embeddings: torch.tensor
    labels: torch.tensor
    # Fasttext embeddings corresponding to the labels
    label_embeddings: torch.Tensor
    subject_hot_encodings: torch.Tensor
    object_hot_encodings: torch.Tensor
    subject_embeddings: torch.Tensor
    object_embeddings: torch.Tensor
    logits: Optional[torch.Tensor]
    logits_thresholded: Optional[torch.Tensor]
    new_embeddings: Optional[torch.Tensor]

    @classmethod
    def from_chains(cls, chains: List[Chain], fast_text: fasttext.FastText._FastText):
        new = cls(
            embeddings=torch.stack(
                [torch.from_numpy(np.stack(chain.embeddings)) for chain in chains]
            ),
            subject_hot_encodings=torch.stack(
                [torch.stack(chain.subject_hot_encodings) for chain in chains]
            ),
            object_hot_encodings=torch.stack(
                [torch.stack(chain.object_hot_encodings) for chain in chains]
            ),
            labels=torch.tensor([chain.label for chain in chains]),
            label_embeddings=torch.tensor(
                np.stack(
                    [
                        fast_text.get_word_vector(chain.get_central_lemma())
                        for chain in chains
                    ]
                )
            ),
            subject_embeddings=torch.stack(
                [
                    torch.from_numpy(np.stack(chain.subject_embeddings))
                    for chain in chains
                ]
            ),
            object_embeddings=torch.stack(
                [
                    torch.from_numpy(np.stack(chain.object_embeddings))
                    for chain in chains
                ]
            ),
            logits=None,
            logits_thresholded=None,
            new_embeddings=None,
        )
        return new

    def to(self, device) -> "ChainBatch":
        return ChainBatch(
            **{
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in dataclasses.asdict(self).items()
            }
        )

    def __getitem__(self, item):
        return getattr(self, item)


def get_predictable_lemmas(input_events: List[Event], vocabulary: List[str]):
    for i, event in enumerate(input_events):
        if event.lemma in vocabulary:
            yield i


def get_windows(events: List[Event], window_size: int, vocabulary: List[str]):
    windows = []
    new_sublists = [events]
    while len(new_sublists) > 0:
        new_windows, new_sublists = _get_windows(
            new_sublists, window_size=window_size, vocabulary=vocabulary
        )
        windows.extend(new_windows)
    return windows


def _get_windows(events: List[List[Event]], window_size: int, vocabulary: List[str]):
    new_sublists = []
    windows = []
    for sublist in events:
        for offset in get_predictable_lemmas(sublist, vocabulary):
            if offset >= window_size // 2 and offset + (window_size // 2) <= len(
                sublist
            ):
                windows.append(
                    tuple(
                        sublist[
                            offset - window_size // 2 : offset + window_size // 2 + 1
                        ]
                    )
                )
                new_sublists.append(sublist[: offset - (window_size // 2)])
                new_sublists.append(sublist[offset + (window_size // 2) :])
                break
        # If we don't find anything we just discard the sublist
    return windows, new_sublists


def str_to_int(in_str):
    out = 0
    for i, c in enumerate(reversed(in_str), start=1):
        out += (ord(c) - ord("A")) * 26**i
    return out


def character_list_to_hot_encoding(input_list: List[str]) -> torch.Tensor:
    encoding = torch.zeros(26)
    for c in input_list:
        if c != "_":
            encoding[str_to_int(c) % 26] = 1
    return encoding


class EventWindowDataset(Dataset):
    """
    Returns windows of configurable size from event json file.

    As the window size increases, fewer documents qualify for being included.
    To not "cheat" by means of including data twice (which may end up in different splits), we only do non-overlapping windows.
    """

    def __init__(
        self,
        file_name: str,
        *args,
        window_size: int = 5,
        vocabulary: List[str] = PREDICTABLE_LEMMAS,
        over_sampling: bool = False,
        edge_markers: bool = False,
        fast_text: Optional[fasttext.FastText._FastText] = None,
        size_limit: Optional[int] = None,
        **kwargs,
    ):
        self.ft = fast_text
        in_file = open(file_name)
        self.chains = []
        self.vocabulary = vocabulary
        self.window_size = window_size
        self.over_sampled = over_sampling
        for i, line in enumerate(in_file):
            chain = [Event.from_json(e) for e in json.loads(line)]
            if edge_markers:
                chain = EventWindowDataset.add_edge_markers(
                    chain, self.window_size // 2
                )
            windows = get_windows(chain, self.window_size, vocabulary)
            for window in windows:
                assert window[len(window) // 2].lemma in self.vocabulary
            self.chains.extend(windows)
            if size_limit and size_limit <= i:
                break
        # remove all duplicates
        self.chains = list(set(self.chains))
        if self.over_sampled:
            lemma_counter = Counter()
            for chain in self.chains:
                lemma_counter.update([chain.get_central_lemma()])
            _least_lemma, least_common_count = list(lemma_counter.most_common())[-1]
            _, most_common_count = list(lemma_counter.most_common())[0]
            num_duplicates_per_item = {
                k: most_common_count / v for k, v in lemma_counter.items()
            }
            balanced_chains = []
            for chain in self.chains:
                for x in range(
                    int(num_duplicates_per_item[chain[window_size // 2].lemma])
                ):
                    balanced_chains.append(chain)
                if (
                    random.random()
                    <= num_duplicates_per_item[chain[window_size // 2].lemma] % 1
                ):
                    balanced_chains.append(chain)
                # if (
                #     balanced_counter.get(chain[window_size // 2].lemma, 0)
                #     < least_common_count
                # ):
                #     balanced_chains.append(chain)
                #     balanced_counter.update([chain[window_size // 2].lemma])
            random.shuffle(balanced_chains)
            self.chains = balanced_chains
        super().__init__(*args, **kwargs)

    @staticmethod
    def add_edge_markers(chain: List[Event], length):
        return (
            [Event(f"<begin{n}>", [], [], [], []) for n in range(length)]
            + chain
            + [Event(f"<end{n}>", [], [], [], []) for n in range(length)]
        )

    @staticmethod
    def get_class_distribution(instance):
        lemma_counter = Counter()
        if isinstance(instance, torch.utils.data.Subset):
            dataset = instance.dataset
        else:
            dataset = instance
        for chain in instance:
            lemma_counter.update([chain.lemmas[dataset.window_size // 2]])
        in_order = torch.tensor(
            [lemma_counter.get(v, 0) for v in dataset.vocabulary],
            dtype=torch.float,
        )
        return torch.nn.functional.normalize(in_order, dim=0, p=1)

    def __getitem__(self, n):
        chain = self.chains[n]
        to_mask_index = self.window_size // 2
        return Chain(
            lemmas=[event.lemma for event in chain],
            embeddings=[
                self.ft.get_word_vector(event.lemma)
                if i != to_mask_index
                else np.zeros(300, dtype=np.float32)
                for i, event in enumerate(chain)
            ],
            subject_hot_encodings=[
                character_list_to_hot_encoding(event.subjects) for event in chain
            ],
            object_hot_encodings=[
                character_list_to_hot_encoding(event.objects) for event in chain
            ],
            vocabulary=self.vocabulary,
            subject_names=[c.subject_names for c in chain],
            object_names=[c.object_names for c in chain],
            subject_embeddings=[
                torch.mean(
                    torch.stack(
                        [self.ft.get_word_vector(subj) for subj in event.subject_names]
                        or [torch.zeros(300)]
                    ),
                    0,
                )
                if i != to_mask_index
                else np.zeros(300, dtype=np.float32)
                for i, event in enumerate(chain)
            ],
            object_embeddings=[
                torch.mean(
                    torch.stack(
                        [self.ft.get_word_vector(subj) for subj in event.subject_names]
                        or [torch.zeros(300)]
                    ),
                    0,
                )
                if i != to_mask_index
                else np.zeros(300, dtype=np.float32)
                for i, event in enumerate(chain)
            ],
        )

    def get_label_counts(self):
        counter = Counter()
        for chain in self:
            counter.update([self.vocabulary[chain.label]])
        return counter

    def get_label_distribution(self):
        counts = self.get_label_counts()
        total = sum(counts.values())
        return {name: count / total for name, count in counts.items()}

    def __len__(self):
        return len(self.chains)


class SimilarityDataset(EventWindowDataset):
    """ """

    def __init__(
        self,
        file_name: str,
        *args,
        window_size: int = 5,
        vocabulary: List[str] = PREDICTABLE_LEMMAS,
        over_sampling: bool = False,
        edge_markers: bool = False,
        fast_text: Optional[fasttext.FastText._FastText] = None,
        **kwargs,
    ):
        self.ft = fast_text
        in_file = open(file_name)
        self.chains = []
        self.vocabulary = vocabulary
        self.window_size = window_size
        self.over_sampled = over_sampling
        self.similarities = {}
        self.doc_ids = []
        self.doc_id_positions = {}
        for i, line in enumerate(in_file):
            data = json.loads(line)
            local_doc_ids = [data[f"doc_id_1"], data["doc_id_2"]]
            for chains_data, doc_id in zip(
                [data["chains_1"], data["chains_2"]], local_doc_ids
            ):
                for chain_data in chains_data:
                    chain = [Event.from_json(e) for e in chain_data]
                    if edge_markers:
                        chain = EventWindowDataset.add_edge_markers(
                            chain, self.window_size // 2
                        )
                    windows = get_windows(chain, self.window_size, vocabulary)
                    for window in windows:
                        assert window[len(window) // 2].lemma in self.vocabulary
                    self.chains.extend(windows)
                    self.doc_ids.extend([doc_id] * len(windows))
                    self.doc_id_positions[doc_id] = self.doc_id_positions.get(
                        doc_id, []
                    ) + list(range(len(self.chains) - len(windows), len(self.chains)))
            self.similarities[tuple(local_doc_ids)] = data["similarities"]

    def __getitem__(self, n):
        return super().__getitem__(n), self.doc_ids[n]
