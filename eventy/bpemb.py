from functools import lru_cache

import bpemb
import torch


class BPEmb:
    """
    Wrapper to make bpemb API compatible with fasttext.
    """

    def __init__(self):
        self.inner = bpemb.BPEmb(
            model_file="multi.wiki.bpe.vs1000000.model",
            emb_file="multi.wiki.bpe.vs1000000.d300.w2v.bin",
            dim=300,
        )

    @lru_cache(2_000_000)
    def get_word_vector(self, word):
        return torch.from_numpy(self.inner.vectors[self.inner.encode_ids(word)].mean(0))
