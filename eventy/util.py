import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def token_list(tokens) -> str:
    return "[" + ", ".join(tokens) + "]"


def matrix_based_similarity(embeddings_a, embeddings_b):
    """
    SBert style row and column based max similarities.

    This solves an assignment problem of sorts.
    """
    sims = cosine_similarity(
        embeddings_a,
        embeddings_b,
    )
    mean = sims.max(0).values.mean() + sims.max(1).values.mean()
    return mean
