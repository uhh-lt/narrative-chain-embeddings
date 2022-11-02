"""
This file is based on imbalanced-dataset-sampler

MIT License

Copyright (c) 2018 Ming adaptations 2022 by Hans Ole Hatzel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Callable, Optional

import pandas as pd
import torch
import torch.utils.data
import torchvision

from eventy.config import SamplingSchedule


class DynamicImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
        num_steps: Optional[int] = None,
        sampling_schedule: SamplingSchedule = SamplingSchedule.BALANCED_TO_REAL,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())
        self.current_weights = (
            self.weights
            if sampling_schedule == SamplingSchedule.BALANCED_TO_REAL
            else torch.ones_like(self.weights)
        )
        self.num_steps = num_steps
        self.sampling_schedule = sampling_schedule
        self.step = 0

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def _update(self):
        self.step += 1
        if self.step > self.num_steps:
            print("WARNING: exceeding planed steps in sampler")
        if self.num_steps is not None:
            delta = (torch.ones_like(self.weights) - self.weights) / self.num_steps
        else:
            delta = torch.zeros_like(self.weights)
        if self.sampling_schedule == SamplingSchedule.BALANCED_TO_REAL:
            self.current_weights = self.weights + delta * self.step
        elif self.sampling_schedule == SamplingSchedule.REAL_TO_BALANCED:
            self.current_weights = torch.ones_like(self.weights) - delta * self.step
        elif self.sampling_schedule == SamplingSchedule.REAL:
            self.current_weights = torch.ones_like(self.weights)
        elif self.sampling_schedule == SamplingSchedule.BALANCED:
            self.current_weights = self.weights
        else:
            raise ValueError("Invalid sampling schedule")

    def __iter__(self):
        samples = (
            self.indices[i]
            for i in torch.multinomial(
                self.current_weights, self.num_samples, replacement=True
            )
        )
        self._update()
        return samples

    def __len__(self):
        return self.num_samples
