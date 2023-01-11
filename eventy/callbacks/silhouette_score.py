from random import sample
from typing import Dict, Optional

import sklearn.metrics
import torch
from catalyst.callbacks import LoaderMetricCallback
from catalyst.metrics._metric import ICallbackLoaderMetric


class SilhouetteMetric(ICallbackLoaderMetric):
    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        sample_size: Optional[int] = None,
    ):
        """Init."""
        super().__init__(compute_on_call=compute_on_call, suffix=suffix, prefix=prefix)
        self.prefix = prefix or ""
        self.suffix = suffix or ""
        self.sample_size = sample_size
        self.metric_name = f"{prefix}silhouette{suffix}"

    def reset(self, num_batches: int, num_samples: int) -> None:
        self.embeddings = []
        self.labels = []

    def update(self, embeddings: torch.Tensor, targets: torch.Tensor) -> None:
        self.embeddings.append(embeddings.detach().cpu())
        self.labels.append(targets)

    def compute(self) -> float:
        return sklearn.metrics.silhouette_score(
            torch.cat(self.embeddings).detach().cpu(),
            torch.cat(self.labels).detach().cpu(),
            sample_size=self.sample_size,
        )

    def compute_key_value(self) -> Dict[str, float]:
        """Computes the metric based on it's accumulated state.

        By default, this is called at the end of each loader
        (`on_loader_end` event).

        Returns:
            Dict: computed value in key-value format.  # noqa: DAR202
        """
        return {"silhouette": self.compute()}


class SilhouetteScoreCallback(LoaderMetricCallback):
    def __init__(
        self,
        *args,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        sample_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            metric=SilhouetteMetric(
                prefix=prefix, suffix=suffix, sample_size=sample_size
            ),
            **kwargs,
        )
