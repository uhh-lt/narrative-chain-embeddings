"""
Prepare data for t5 pretraining
"""
import dataclasses
import gzip
import math
import random
from typing import List, Optional

import sentencepiece
import torch
import typer
import ujson as json
from tqdm import tqdm
from transformers import (
    AutoModelWithLMHead,
    LogitsProcessor,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from eventy.dataset import Event

app = typer.Typer()


class ForcedSequenceLogitsProcessor(LogitsProcessor):
    """
    Args:
        forced_sequence_ids (`int`):
            The output token_ids for the whole batch, these will be forced in the actual output
    """

    def __init__(self, forced_sequence_ids: torch.tensor):
        self.forced_sequences = forced_sequence_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        at_token = input_ids.shape[-1] - 1
        new_scores = torch.zeros_like(scores)
        new_scores.fill_(-float("inf"))
        forced_indices = self.forced_sequences[:, at_token].unsqueeze(1)
        prob_values = scores.gather(1, forced_indices)
        new_scores.scatter_(1, forced_indices, prob_values)
        return new_scores


@app.command()
def main(dataset_path: str, out_path: str):
    out_file = gzip.open(out_path, "wt")
    for line in tqdm(open(dataset_path, "r")):
        data = json.loads(line)
        chain = [Event.from_json(event) for event in data["chain"]]
        chain_texts = [e.to_text(include_iobj=True, include_names=True) for e in chain]
        out_file.write("".join(chain_texts) + "\n")


def sequence_probabilities(model, tokenizer, input_texts, targets):
    prompt = tokenizer(input_texts, padding=True, return_tensors="pt").to(
        "cuda:0"
    )
    targets_tokenized = tokenizer(targets, padding=True, return_tensors="pt").to(
        "cuda:0"
    )
    outputs = model.generate(
        prompt.input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        logits_processor=[ForcedSequenceLogitsProcessor(targets_tokenized.input_ids)],
        max_length=targets_tokenized.input_ids.shape[-1] + 1,
        do_sample=False,
    )
    scores = torch.stack([s.max(1).values for s in outputs.scores])
    sequence_scores = scores.T.masked_fill(
        targets_tokenized.attention_mask == 0, math.nan
    ).nanmean(1)
    return sequence_scores


def mcnc_event_seq_prob(
    model,
    tokenizer,
    chain: List[Event],
    all_events: List[Event],
    mask: Optional[int] = None,
    only_verbs: bool = False,
    mask_str: str = "<extra_token_0>",
):
    to_text_config = {"include_iobj": True, "include_names": True}
    if mask is None:
        mask = random.randint(0, len(chain) - 1)
    if only_verbs:
        with_original_lemma = dataclasses.asdict(chain[mask])
        original_lemma = with_original_lemma["lemma"]
        del with_original_lemma["lemma"]
        masked_event = Event(**with_original_lemma, lemma=mask_str)
    before_events = [e.to_text(**to_text_config) for e in chain[:mask]]
    after_events = [e.to_text(**to_text_config) for e in chain[mask + 1:]]
    input_texts = ["".join(before_events) + (masked_event.to_text(**to_text_config) if only_verbs else mask_str) + "".join(after_events)]
    if not only_verbs:
        input_texts *= 5
    if only_verbs:
        targets = ["<extra_id_0>" + original_lemma + "<extra_id_1>"]
    else:
        targets = ["<extra_id_0>" + chain[mask].to_text(**to_text_config) + "<extra_id_1>"]
    candidates = random.choices(all_events, k=4)
    for event in candidates:
        if only_verbs:
            random_lemma = random.choice(all_events).lemma
            new_event = Event(**with_original_lemma, lemma=random_lemma)
            input_texts.append("".join(before_events) + new_event.to_text(**to_text_config) + "".join(after_events))
            targets.append("<extra_id_0>" + event.lemma + "<extra_id_1>")
        else:
            targets.append("<extra_id_0>" + event.to_text(**to_text_config) + "<extra_id_1>")
    probs = sequence_probabilities(model, tokenizer, input_texts, targets)
    return probs.argmax() == 0


"""
Strat 1:

Strat 2:

Sequence propability

sent 1: event_1 event_2 event_3 event_4 event_5
sent 2: event_1 event_2 event_random event_4 event_5


How likley is this?
"""

@app.command()
def test(
    model_path: str,
    train_dataset_path: str,
    test_dataset_path: str,
    min_chain_length: Optional[int] = None,
    only_verbs: bool = False,
):
    model = AutoModelWithLMHead.from_pretrained(model_path).to("cuda:0")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    all_events = list()
    all_verbs = list()
    correct = 0
    total = 0
    text_config = {"include_iobj": True, "include_names": True}
    for i, line in enumerate(tqdm(open(train_dataset_path, "r"))):
        data = json.loads(line)
        events = [Event.from_json(event) for event in data["chain"]]
        all_events.extend(events)
        all_verbs.extend([e.lemma for e in events])
        if i > 100000:
            break
    lines = open(test_dataset_path, "r").readlines(750 * 10000)
    # random.shuffle(lines)
    print("Num lines", len(lines))
    pbar = tqdm(enumerate(lines))
    text_events = [e.to_text(**text_config) for e in all_events]
    for i, line in pbar:
        data = json.loads(line)
        elements = [Event.from_json(event) for event in data["chain"]]
        if min_chain_length is not None and len(elements) < min_chain_length:
            continue
        if i > 100000:
            break
        total += 1
        if only_verbs:
            is_correct = mcnc_event_seq_prob(
                model, tokenizer, elements, all_events, only_verbs=True
            )
        else:
            # is_correct = mcnc_event(model, tokenizer, [e.to_text(**text_config) for e in elements], text_events)
            is_correct = mcnc_event_seq_prob(
                model, tokenizer, elements, all_events
            )
        if is_correct == True:
            correct += 1
        pbar.set_postfix({"accuracy": correct / total})
        if i % 100 == 0:
            print("Accuracy", correct / total)
    print("Accuracy", correct / total)


@app.command()
def train_tokenizer(input_path: str, out_path: str):
    lines = gzip.open(input_path, "rt")
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=lines,
        model_prefix=out_path,
        vocab_size=1000,
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
    )


@app.command()
def test_tokenizer(input_path: str):
    tokenizer = T5Tokenizer(input_path)
    print(tokenizer.tokenize("([A], test, [B]"))
    tokenizer.save_pretrained("gigaword-chains")


if __name__ == "__main__":
    app()
