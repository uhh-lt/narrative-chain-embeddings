from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class EventyModel(nn.Module):
    def __init__(
        self,
        *args,
        dropout: float,
        embedding_size: int = 300,
        num_inputs: int = 5,
        output_vocab: int = 4,
        vocab_embeddings: Optional[torch.Tensor] = None,
        model_characters: bool = True,
        class_distribution: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dropout_chance = dropout
        self.vocab_embeddings = torch.nn.parameter.Parameter(
            vocab_embeddings, requires_grad=False
        )
        self.model_characters = model_characters
        self.class_distribution = torch.nn.parameter.Parameter(
            class_distribution
            if class_distribution is not None
            else torch.zeros(output_vocab, dtype=torch.float),
            requires_grad=False,
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
            nn.Dropout(self.dropout_chance),
            nn.Linear(128, self.character_embeddings_size),
            nn.Sigmoid(),
            nn.Dropout(self.dropout_chance),
        )
        self.input_layer = nn.Sequential(
            nn.Linear(
                embedding_size * num_inputs
                + self.character_embeddings_size * num_inputs * 2,
                self.hidden_size_1,
            ),
            nn.Sigmoid(),
            nn.Dropout(self.dropout_chance),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.Sigmoid(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size_2, output_vocab),
            nn.Dropout(self.dropout_chance),
        )
        # self.loss = nn.CrossEntropyLoss(weight=1 / self.class_distribution)
        self.loss = nn.CrossEntropyLoss()
        embedding_loss_func = nn.CosineEmbeddingLoss(reduction="none")
        self.embedding_loss = lambda labels, *args: (
            embedding_loss_func(*args)
            # * (1 / self.class_distribution).index_select(0, labels).unsqueeze(0)
        ).mean()

    def forward(
        self,
        embeddings: torch.Tensor,
        subject_hot_encodings: torch.Tensor,
        object_hot_encodings: torch.Tensor,
        labels: torch.Tensor,
        label_embeddings: torch.Tensor,
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
            embeddings = self.input_layer(with_characters)
            logits = self.output_layer(embeddings)
        else:
            embeddings = self.input_layer(reshaped)
            logits = self.output_layer(embeddings)
        ft_embedding_loss = self.embedding_loss(
            labels,
            (self.vocab_embeddings * (logits**3).softmax(-1).unsqueeze(-1)).mean(1),
            label_embeddings,
            torch.ones(logits.shape[0], device=labels.device),
        )
        loss = self.loss(logits, labels)
        return BatchOutput(logits, loss, ft_embedding_loss, embeddings)


@dataclass
class BatchOutput:
    logits: torch.Tensor
    classification_loss: torch.Tensor
    embedding_loss: torch.Tensor
    embeddings: torch.Tensor

    def __getitem__(self, item):
        return getattr(self, item)
