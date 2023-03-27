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
    r"""
    Args:
        bos_token_id (`int`):
            The id of the token to force as the first generated token.
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
        # if forced_indices[0] != forced_indices[1]:
        #     breakpoint()
        return new_scores


@app.command()
def main(dataset_path: str, out_path: str):
    out_file = gzip.open(out_path, "wt")
    for line in tqdm(open(dataset_path, "r")):
        data = json.loads(line)
        chain = [Event.from_json(event) for event in data["chain"]]
        chain_texts = [e.to_text(include_iobj=True, include_names=True) for e in chain]
        out_file.write("".join(chain_texts) + "\n")


def sequence_probabilities(model, tokenizer, input_texts):
    input_seq = tokenizer(
        [t[2:] for t in input_texts], padding=True, return_tensors="pt"
    ).to("cuda:0")
    prompt = tokenizer(["(["] * len(input_texts), padding=True, return_tensors="pt").to(
        "cuda:0"
    )
    outputs = model.generate(
        prompt.input_ids,
        return_dict_in_generate=True,
        output_scores=True,
        logits_processor=[ForcedSequenceLogitsProcessor(input_seq.input_ids)],
        max_length=input_seq.input_ids.shape[-1] + 1,
        do_sample=False,
    )
    scores = torch.stack([s.max(1).values for s in outputs.scores])
    sequence_scores = scores.T.masked_fill(
        input_seq.attention_mask == 0, math.nan
    ).nanmean(1)
    return sequence_scores


def mcnc_event_seq_prob(
    model,
    tokenizer,
    chain: List[str],
    all_events: List[str],
    mask: Optional[int] = None,
):
    input_texts = ["".join(chain)]
    if mask is None:
        mask = random.randint(0, len(chain) - 1)
    candidates = random.choices(all_events, k=4)
    for event in candidates:
        input_texts.append("".join(chain[:mask]) + event + "".join(chain[mask + 1 :]))
    probs = sequence_probabilities(model, tokenizer, input_texts)
    return probs.argmax() == 0


def mcnc_event(
    model,
    tokenizer,
    chain: List[str],
    all_events: List[str],
    mask: Optional[int] = None,
):
    candidates = random.choices(all_events, k=4)
    if mask is None:
        mask = random.randint(0, len(chain) - 1)
    gold = chain[mask]
    chain[mask] = "<extra_token_0>"
    scores = []
    candidate_labels = candidates + [gold]
    for word in candidate_labels:
        input_ids = tokenizer("".join(chain), return_tensors="pt").input_ids.to(
            "cuda:0"
        )
        labels = tokenizer(
            f"<extra_id_0>{word}<extra_id_1>", return_tensors="pt"
        ).input_ids.to("cuda:0")
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        scores.append(loss.item())
    _, predicted_word = min(zip(scores, candidate_labels), key=lambda x: x[0])
    return predicted_word == gold


def mcnc_verb(
    model,
    tokenizer,
    chain: List[Event],
    all_lemmas: List[str],
    mask: Optional[int] = None,
):
    candidates = random.choices(all_lemmas, k=4)
    if mask is None:
        mask = random.randint(0, len(chain) - 1)
    gold = chain[mask]
    old = dataclasses.asdict(gold)
    del old["lemma"]
    chain[mask] = Event(**old, lemma="<extra_token_0>")
    scores = []
    candidate_labels = candidates + [gold]
    input_batch = tokenizer(
        ["".join([e.to_text(include_iobj=True, include_names=True) for e in chain])]
        * 4,
        return_tensors="pt",
        padding=True,
    )
    labels = tokenizer(
        [f"<extra_id_0>{word}<extra_id_1>" for word in candidate_labels],
        return_tensors="pt",
        padding=True,
    ).input_ids
    output = model(
        input_ids=input_batch.input_ids,
        labels=labels,
        attention_mask=input_batch.attention_mask,
    )
    loss = output.loss
    breakpoint()
    for word in candidate_labels:
        scores.append(loss.item())
    _, predicted_word = min(zip(scores, candidate_labels), key=lambda x: x[0])
    return predicted_word == gold


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
    for i, line in enumerate(tqdm(open(train_dataset_path, "r"))):
        data = json.loads(line)
        events = [Event.from_json(event) for event in data["chain"]]
        text_events = [e.to_text(include_iobj=True, include_names=True) for e in events]
        all_events.extend(text_events)
        all_verbs.extend([e.lemma for e in events])
        if i > 100000:
            break
    pbar = tqdm(enumerate(open(test_dataset_path, "r")))
    for i, line in pbar:
        data = json.loads(line)
        elements = [Event.from_json(event) for event in data["chain"]]
        text_elements = [
            e.to_text(include_iobj=True, include_names=True) for e in elements
        ]
        if min_chain_length is not None and len(elements) < 5:
            continue
        if i > 100000:
            break
        total += 1
        if only_verbs:
            is_correct = mcnc_verb(model, tokenizer, elements, all_verbs)
        else:
            # is_correct = mcnc_event(model, tokenizer, text_elements, all_events)
            is_correct = mcnc_event_seq_prob(
                model, tokenizer, text_elements, all_events
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
