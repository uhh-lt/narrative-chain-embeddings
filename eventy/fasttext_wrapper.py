from functools import lru_cache

import fasttext
import torch


class FasttextWrapper:
    def __init__(self, embedding_source):
        self.inner = fasttext.load_model(embedding_source)

    @lru_cache(2_000_000)
    def get_word_vector(self, word):
        return torch.tensor(self.inner.get_word_vector(word))
