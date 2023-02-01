from typing import Iterable, List, Optional, Sequence

import torch
from catalyst.callbacks import BatchMetricCallback
from catalyst.metrics._topk_metric import TopKMetric


class MultipleChoiceCallback(BatchMetricCallback):
    def __init__(
        self,
        input_key: str,
        target_key: str,
        n_choices: int,
        num_classes: int = None,
        log_on_batch: bool = True,
        topk: Sequence[int] = (1,),
        prefix: str = None,
        suffix: str = None,
        distribution: Optional[torch.Tensor] = None,
        loader_keys: List[str] = [],
    ):
        """
        Init.

        Unlike the parent class we have a laoder_keys argument in which you can specify which splits to evaluate this on.
        This is necessary as the operation is very expensive and actually takes about half the time when running on a big GPU server.
        """
        self.loader_keys = loader_keys
        super().__init__(
            metric=MultipleChoiceMetric(
                topk=topk,
                num_classes=num_classes,
                prefix=prefix,
                num_choices=n_choices,
                suffix=suffix,
                distribution=distribution,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )

    def on_loader_start(self, runner: "IRunner") -> None:
        if runner.loader_key in self.loader_keys:
            super().on_loader_start(runner)

    def on_batch_end(self, runner: "IRunner") -> None:
        if runner.loader_key in self.loader_keys:
            super().on_batch_end(runner)

    def on_loader_end(self, runner: "IRunner") -> None:
        if runner.loader_key in self.loader_keys:
            super().on_loader_end(runner)


def multiple_choice(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    topk: Sequence[int] = (1,),
    num_options: int = 5,
    distribution: Optional[torch.Tensor] = None,
) -> Sequence[torch.Tensor]:
    max_k = max(topk)
    batch_size = targets.size(0)
    num_classes = outputs.size(1)

    if distribution is None:
        distribution = torch.ones(num_classes, device=outputs.device)
    random_options = torch.multinomial(
        distribution, (num_options - 1) * batch_size, replacement=True
    ).reshape(batch_size, num_options - 1)
    # torch.multinomial(distribution, (batch_size * num_options - 1), replacement=True)
    # random_options = torch.randint(0, num_classes - 1, (batch_size, num_options - 1), device=outputs.device)
    options = torch.cat((random_options, targets.reshape(-1, 1)), dim=1)
    mc_outputs = outputs.gather(1, options)
    mc_targets = torch.ones((batch_size, 1), device=outputs.device) * (num_options - 1)

    if len(mc_outputs.shape) == 1 or mc_outputs.shape[1] == 1:
        # binary accuracy
        pred = mc_outputs.t()
    else:
        # multiclass accuracy
        _, pred = mc_outputs.topk(max_k, 1, True, True)
        pred = pred.t()
    correct = pred.eq(mc_targets.long().view(1, -1).expand_as(pred))

    output = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        output.append(correct_k.mul_(1.0 / batch_size))
    return output


class MultipleChoiceMetric(TopKMetric):
    """ """

    def __init__(
        self,
        num_choices: int,
        topk: Iterable[int] = None,
        num_classes: int = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
        distribution: Optional[torch.Tensor] = None,
    ):
        """Init AccuracyMetric"""
        self.topk = topk or get_default_topk(num_classes)
        self.num_choices = num_choices
        self.distribution = distribution

        def mc(*args, **kwargs):
            return multiple_choice(*args, **kwargs, distribution=distribution)

        super().__init__(
            metric_name=f"{self.num_choices}_choice_accuracy",
            metric_function=mc,
            topk=self.topk,
            compute_on_call=compute_on_call,
            prefix=prefix,
            suffix=suffix,
        )
