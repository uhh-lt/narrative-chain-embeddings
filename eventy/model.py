from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class EventyModel(nn.Module):
    def __init__(
        self,
        *args,
        embedding_size: int = 300,
        num_inputs: int = 5,
        output_vocab: int = 4,
        model_characters: bool = True,
        class_distribution: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_characters = model_characters
        self.class_distribution = (
            class_distribution
            if class_distribution is not None
            else torch.zeros(output_vocab, dtype=torch.float)
        )
        self.hidden_size_1 = 1024
        self.hidden_size_2 = 1024
        self.num_inputs = num_inputs
        self.chain_input_size = embedding_size * num_inputs
        self.test = nn.Linear(embedding_size * num_inputs, self.hidden_size_1)
        self.character_embeddings_size = 10 if model_characters else 0
        self.char_embedding_layer = nn.Sequential(
            nn.Linear(26, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.character_embeddings_size),
            nn.Sigmoid(),
        )
        self.input_layer = nn.Sequential(
            nn.Linear(
                embedding_size * num_inputs
                + self.character_embeddings_size * num_inputs * 2,
                self.hidden_size_1,
            ),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size_2, output_vocab),
        )
        self.loss = nn.CrossEntropyLoss(weight=1 / self.class_distribution)

    def forward(
        self,
        embeddings: torch.Tensor,
        subject_hot_encodings: torch.Tensor,
        object_hot_encodings: torch.Tensor,
        labels: torch.Tensor,
    ):
        reshaped = embeddings.reshape(-1, self.chain_input_size)
        if self.model_characters:
            subject_embeddings = self.char_embedding_layer(
                subject_hot_encodings
            ).reshape(-1, self.character_embeddings_size * self.num_inputs)
            object_embeddings = self.char_embedding_layer(object_hot_encodings).reshape(
                -1, self.character_embeddings_size * self.num_inputs
            )
            with_characters = torch.cat(
                [subject_embeddings, reshaped, object_embeddings], 1
            )
            logits = self.input_layer(with_characters)
        else:
            logits = self.input_layer(reshaped)
        loss = self.loss(logits, labels)
        return BatchOutput(logits, loss)


@dataclass
class BatchOutput:
    logits: torch.Tensor
    loss: torch.Tensor

    def __getitem__(self, item):
        return getattr(self, item)
