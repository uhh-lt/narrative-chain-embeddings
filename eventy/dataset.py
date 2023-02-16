import dataclasses
import timeit

from tqdm import tqdm

try:
    import ujson as json
except ImportError:
    import json

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fasttext
import numpy as np
import torch
from torch.utils.data import Dataset

PREDICTABLE_LEMMAS = ["schlagen", "warten", "laufen", "gehen"]


@dataclass(frozen=True, order=True)
class Event:
    lemma: str
    objects: Tuple[str]
    iobjs: Tuple[str]
    subjects: Tuple[str]
    object_names: Tuple[str]
    subject_names: Tuple[str]

    @classmethod
    def from_json(cls, data: Dict) -> "Event":
        iobjs = (
            tuple(o for o in (data["iobject"] or {}).values() or [] if o is not None)
            or None
        )
        event = Event(
            lemma=data["verb_lemma"],
            objects=tuple(data["objects"]),
            subjects=tuple(data["subjects"]),
            iobjs=iobjs,
            subject_names=tuple(
                e for e in data.get("subject_names") if isinstance(e, str)
            ),
            object_names=tuple(
                e for e in data.get("object_names") if isinstance(e, str)
            ),
        )
        return event


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
    iobject_embeddings: List[str]
    iobject_names: List[str]
    masked: int

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
        iobject_embeddings,
        iobject_names,
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
        self.iobject_embeddings = iobject_embeddings
        self.iobject_names = iobject_names

    def get_central_lemma(self) -> str:
        if len(self.lemmas) % 2 != 1:
            print("Warning: even length chains may not be properly supported")
        return self.lemmas[len(self.lemmas) // 2]

    def __hash__(self):
        return hash(
            hash(e)
            for e in zip(
                self.lemmas, self.subject_hot_encodings, self.object_hot_encodings
            )
        )

    def __eq__(self, other):
        return (
            self.lemmas == other.lemmas
            and self.subject_names == other.subject_names
            and self.object_names == other.object_names
            and self.subject_hot_encodings == other.subject_hot_encodings
            and self.object_hot_encodings == other.object_hot_encodings
        )


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
    iobject_embeddings: torch.Tensor
    logits: Optional[torch.Tensor]
    logits_thresholded: Optional[torch.Tensor]
    new_embeddings: Optional[torch.Tensor]

    @classmethod
    def from_chains(cls, chains: List[Chain], fast_text: fasttext.FastText._FastText):
        new = cls(
            embeddings=torch.stack(
                [(torch.stack(chain.embeddings)) for chain in chains]
            ),
            subject_hot_encodings=torch.stack(
                [torch.stack(chain.subject_hot_encodings) for chain in chains]
            ),
            object_hot_encodings=torch.stack(
                [torch.stack(chain.object_hot_encodings) for chain in chains]
            ),
            labels=torch.tensor([chain.label for chain in chains]),
            label_embeddings=torch.stack(
                [
                    fast_text.get_word_vector(chain.get_central_lemma())
                    for chain in chains
                ]
            ),
            subject_embeddings=torch.stack(
                [torch.stack(chain.subject_embeddings) for chain in chains]
            ),
            object_embeddings=torch.stack(
                [torch.stack(chain.object_embeddings) for chain in chains]
            ),
            iobject_embeddings=torch.stack(
                [torch.stack(chain.iobject_embeddings) for chain in chains]
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
            if offset >= window_size // 2 and offset + (window_size // 2) < len(
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
        min_chain_len: Optional[int] = None,
        **kwargs,
    ):
        self.ft = fast_text
        in_file = open(file_name)
        self.chains = []
        self.label_counter = None
        self.vocabulary = vocabulary
        self.vocab_set = set(vocabulary)
        self.window_size = window_size
        self.over_sampled = over_sampling
        for i, line in tqdm(enumerate(in_file), desc="Reading JSON-dataset"):
            data = json.loads(line)
            chain = [
                Event.from_json(e)
                for e in (data["chain"] if isinstance(data, dict) else data)
                if e["verb_lemma"] in self.vocabulary
            ]
            if min_chain_len is not None and len(chain) < min_chain_len:
                continue
            if edge_markers:
                chain = EventWindowDataset.add_edge_markers(
                    chain, self.window_size // 2
                )
            windows = get_windows(chain, self.window_size, self.vocab_set)
            for window in windows:
                assert len(window) == self.window_size
            for window in windows:
                assert window[len(window) // 2].lemma in self.vocabulary
            self.chains.extend(windows)
            if size_limit and size_limit <= i:
                break
        self.chains = list(set(tqdm(self.chains, desc="Deduplicating dataset")))
        print("Number of unique chains: ", len(self.chains))
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
        self.chain_cache = [None for _ in self.chains]
        super().__init__(*args, **kwargs)

    def performance_test(self):
        for x in [10_000, 20_000, 40_000, 80_000, 160_000]:
            print(
                "Dict",
                x,
                timeit.timeit(
                    "list({n : 1 for n in tqdm(self.chains[:" + str(x) + "])}.keys())",
                    globals={"self": self, "tqdm": tqdm},
                    number=1,
                ),
            )
            print(
                "FrozenSet",
                x,
                timeit.timeit(
                    f"list(frozenset(self.chains[:{x}]))",
                    globals={"self": self},
                    number=1,
                ),
            )

    @staticmethod
    def add_edge_markers(chain: List[Event], length):
        return (
            [
                Event(f"<begin{n}>", tuple(), tuple(), tuple(), tuple(), tuple())
                for n in range(length)
            ]
            + chain
            + [
                Event(f"<end{n}>", tuple(), tuple(), tuple(), tuple(), tuple())
                for n in range(length)
            ]
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
        if self.chain_cache[n] is not None:
            return self.chain_cache[n]
        self.chain_cache[n] = Chain(
            lemmas=[event.lemma for event in chain],
            embeddings=[
                self.ft.get_word_vector(event.lemma)
                if i != to_mask_index
                else torch.zeros(300, dtype=torch.float)
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
                else torch.zeros(300, dtype=torch.float32)
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
                else torch.zeros(300, dtype=torch.float32)
                for i, event in enumerate(chain)
            ],
            iobject_names=[
                " ".join(event.iobjs)
                if event.iobjs is not None and len(event.iobjs) > 0
                else None
                for event in chain
            ],
            iobject_embeddings=[
                torch.mean(
                    torch.stack(
                        [self.ft.get_word_vector(subj) for subj in event.iobjs]
                        or [torch.zeros(300)]
                    ),
                    0,
                )
                if event.iobjs is not None and len(event.iobjs) > 0
                else torch.zeros(300)
                for event in chain
            ],
        )
        return self.chain_cache[n]

    def get_label_counts(self):
        if self.label_counter is None:
            self.label_counter = Counter()
            self.label_counter.update(
                tqdm((chain.label for chain in self), desc="Counting label frequencies")
            )
        return {self.vocabulary[k]: count for k, count in self.label_counter.items()}

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
        self.chain_ids = []
        self.doc_id_positions = {}
        for i, line in enumerate(in_file):
            data = json.loads(line)
            if len(data["chains_1"]) == 0 or len(data["chains_2"]) == 0:
                continue
            local_doc_ids = [data["doc_id_1"], data["doc_id_2"]]
            for (chains_data, doc_id) in zip(
                [data["chains_1"], data["chains_2"]], local_doc_ids
            ):
                for i, chain_data in enumerate(chains_data):
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
                    self.chain_ids.extend([str(i)] * len(windows))
                    self.doc_id_positions[doc_id] = self.doc_id_positions.get(
                        doc_id, []
                    ) + list(range(len(self.chains) - len(windows), len(self.chains)))
            self.similarities[tuple(local_doc_ids)] = data["similarities"]
        self.chain_cache = [None for _ in self.chains]

    def __getitem__(self, n):
        return super().__getitem__(n), self.doc_ids[n] + "_" + self.chain_ids[n]
