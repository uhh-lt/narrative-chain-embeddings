import dataclasses
from collections import defaultdict
from dataclasses import dataclass

import torch
from catalyst import dl, metrics

from eventy.dataset import ChainBatch


class CustomRunner(dl.Runner):
    def __init__(self, *args, class_distribution=None, **kwargs):
        self.class_distribution = class_distribution
        super().__init__(self, *args, **kwargs)

    def predict_batch(self, batch):
        on_device_batch = batch.to(self.engine.device)
        # model inference step
        return self.model(
            on_device_batch.embeddings,
            on_device_batch.subject_hot_encodings,
            on_device_batch.object_hot_encodings,
            on_device_batch.labels,
        )

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False) for key in ["loss"]
        }

    def on_batch_start(self, runner):
        if dataclasses.is_dataclass(self.batch):
            self.batch_size = len(next(iter(dataclasses.asdict(self.batch).values())))
        elif isinstance(self.batch, dict):
            self.batch_size = len(next(iter(self.batch.values())))
        else:
            self.batch_size = len(self.batch[0])

        # we have an batch per each worker...
        self.batch_step += self.engine.num_processes
        self.loader_batch_step += self.engine.num_processes
        self.sample_step += self.batch_size * self.engine.num_processes
        self.loader_sample_step += self.batch_size * self.engine.num_processes
        self.batch_metrics: Dict = defaultdict(None)

    def handle_batch(self, batch):
        # run model forward pass
        on_device_batch: ChainBatch = batch.to(self.engine.device)
        model_output = self.model(
            on_device_batch.embeddings,
            on_device_batch.subject_hot_encodings,
            on_device_batch.object_hot_encodings,
            on_device_batch.labels,
        )
        self.batch.logits = model_output.logits
        self.batch.logits_thresholded = model_output.logits / self.class_distribution
        # log metrics
        self.batch_metrics.update({"loss": model_output.loss})
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        # run model backward pass
        if self.is_train_loader:
            self.engine.backward(model_output.loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

    def on_loader_end(self, runner):
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)
