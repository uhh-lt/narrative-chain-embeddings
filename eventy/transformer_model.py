from typing import Optional

import torch
from torch import nn
from transformers import BertConfig, BertModel

from eventy.model import BatchOutput, EventyModel

from .util import cosine_similarity


class EventyTransformerModel(EventyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size_1 = 300
        self.transformer_config = BertConfig(
            hidden_size=self.hidden_size_1,
            num_hidden_layers=3,
            num_attention_heads=4,
            hidden_dropout_prob=self.dropout_chance,
            vocab_size=self.output_vocab,
            max_position_embeddings=self.num_inputs,
        )
        self.reduction_layer_size = self.embedding_size * 2
        if self.model_character_names:
            self.reduction_layer_size += self.embedding_size * 2
        if self.model_characters:
            self.reduction_layer_size += 2 * 10
        self.reduction_layer = nn.Sequential(
            nn.Linear(self.reduction_layer_size, self.hidden_size_1),
            nn.Dropout(self.dropout_chance),
            nn.Tanh(),
        )
        self.classification_layer = nn.Sequential(
            nn.Linear(self.hidden_size_1, self.output_vocab),
            nn.Tanh(),
        )
        self.transformer = BertModel(self.transformer_config)

    def forward(
        self,
        embeddings: torch.Tensor,
        subject_hot_encodings: torch.Tensor,
        object_hot_encodings: torch.Tensor,
        labels: torch.Tensor,
        label_embeddings: torch.Tensor,
        object_text_embeddings: torch.Tensor,
        subject_text_embeddings: torch.Tensor,
        iobject_embeddings: torch.Tensor,
    ):
        if self.model_characters:
            subject_embeddings = self.char_embedding_layer(subject_hot_encodings)
            object_embeddings = self.char_embedding_layer(object_hot_encodings)
            object_embeddings = torch.cat([object_embeddings, iobject_embeddings], 2)
            # object_embeddings = torch.cat([object_embeddings, iobject_embeddings.reshape(-1, self.embedding_size * self.num_inputs)], 1)
            if self.model_character_names:
                object_embeddings = torch.cat(
                    [object_text_embeddings, object_embeddings], 2
                )
                subject_embeddings = torch.cat(
                    [subject_text_embeddings, subject_embeddings], 2
                )
            embeddings = torch.cat(
                [subject_embeddings, embeddings, object_embeddings], 2
            )
        reduced = self.reduction_layer(embeddings)
        out = self.transformer(inputs_embeds=reduced, output_hidden_states=True)
        predict_emb = out.last_hidden_state[:, self.num_inputs // 2, :]
        logits = self.classification_layer(predict_emb)
        embedding_mixes = torch.nn.functional.normalize(
            (self.vocab_embeddings * (logits**2).softmax(-1).unsqueeze(-1)).mean(1)
        )
        ft_embedding_loss = self.embedding_loss(
            labels,
            embedding_mixes,
            label_embeddings,
            torch.ones(logits.shape[0], device=labels.device),
        )
        ft_euclidean_loss = self.euclidean_loss(
            labels,
            embedding_mixes,
            label_embeddings,
        )
        loss = self.loss(logits, labels)
        similarities = 1 - cosine_similarity(embedding_mixes, self.vocab_embeddings)
        return BatchOutput(
            logits,
            similarities,
            loss,
            ft_embedding_loss,
            ft_euclidean_loss,
            predict_emb,
        )
