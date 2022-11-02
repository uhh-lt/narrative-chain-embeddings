import pathlib
import random
from collections import defaultdict
from typing import Callable, List, Optional, Union

import numpy
from catalyst import dl
from catalyst.callbacks import Callback
from catalyst.contrib.utils.visualization import render_figure_to_array
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class EmbeddingVisualizerCallback(Callback):
    def __init__(
        self,
        label_key: str,
        embedding_key: str,
        class_names: List[str] = [],
        collect_list: Optional[List[str]] = None,
        collection_frequency: Union[float, List[float]] = 1.0,
        dimensionality_reducer: Optional[
            Callable[[numpy.ndarray], numpy.ndarray]
        ] = None,
        prefix: Optional[str] = None,
        loader_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            label_key: key to use from ``runner.batch``, specifies our labels
            embedding_key: key to use from ``runner.batch``, specifies our embeddings
            class_names: list of names for the classes indexing using ``label_key`` should yield the corresponding class name
            collection_list: list of class names to collect embeddings for
            collection_frequency: propotion of examples matching the collection list to include in the visualization, may also be a list of the same lenght as collection_list
            dimensionality_reducer: callable to reduce the stacked embeddings dimension to two
            prefix: plot name for monitoring tools
            loader_keys: List of loaders from which to collect embeddings, will collect from all if None
        """
        super().__init__(120)
        self.embeddings = []
        self.labels = []
        self.loader_keys = loader_keys
        self.class_names = class_names
        self.label_key = label_key
        self.embedding_key = embedding_key
        self.dimensionality_reducer = dimensionality_reducer
        self.collect_list = collect_list
        if isinstance(collection_frequency, list):
            if len(collect_list) != len(collection_frequency):
                raise ValueError(
                    "`collection_frequency` and `collection_list` must be of the same length if the former is a list"
                )
            self.collection_frequency_index = {
                class_names.index(class_name): freq
                for class_name, freq in zip(collect_list, collection_frequency)
            }
        else:
            self.collection_frequency_index = defaultdict(lambda: collection_frequency)
        self.prefix = prefix
        # Let's not interfer with the default randomizer
        self.randomizer = random.Random()

    def on_loader_start(self, runner: "IRunner") -> None:
        self.embeddings = []
        self.labels = []

    def on_loader_end(self, runner: "IRunner") -> None:
        if self.loader_keys is not None and runner.loader_key not in self.loader_keys:
            return
        self.show_embeddings(runner)
        self.embeddings = []
        self.labels = []

    def on_batch_end(self, runner):
        if self.loader_keys is not None and runner.loader_key not in self.loader_keys:
            return
        for class_, emb in zip(
            getattr(runner.batch, self.label_key),
            getattr(runner.batch, self.embedding_key),
        ):
            if (
                self.class_names[class_] in self.collect_list
                or self.collect_list is None
            ):
                collection_chance = self.collection_frequency_index[class_.item()]
                if (
                    collection_chance == 1.0
                    or self.randomizer.random() <= collection_chance
                ):
                    self.labels.append(class_)
                    self.embeddings.append(emb.detach().cpu().numpy())

    def tsne(self, embeddings: numpy.ndarray):
        reducer = TSNE(n_components=2, init="random", perplexity=len(embeddings) // 2)
        reduced = PCA(n_components=min(50, len(embeddings) // 2)).fit_transform(
            numpy.stack(embeddings)
        )
        return reducer.fit_transform(reduced)

    def show_embeddings(
        self, runner: "Runner", out_path: Optional[pathlib.Path] = None
    ):
        if self.dimensionality_reducer is not None:
            reduced_data = self.dimensionality_reducer(self.embeddings)
        else:
            reduced_data = self.tsne(self.embeddings)
        fig, ax = plt.subplots()
        data_per_class_x = defaultdict(list)
        data_per_class_y = defaultdict(list)
        lookup = {}
        for data, class_ in zip(reduced_data, self.labels):
            data_per_class_x[class_.item()].append(data[0])
            data_per_class_y[class_.item()].append(data[1])
        if self.collect_list is not None and len(self.collect_list):
            for i, class_ in enumerate(sorted(set(data_per_class_x.keys()))):
                lookup[class_] = get_cmap("Dark2").colors[
                    i % len(get_cmap("Dark2").colors)
                ]
        for class_, color in lookup.items():
            ax.scatter(
                data_per_class_x[class_],
                data_per_class_y[class_],
                color=color,
            )
        plt.legend(
            labels=[self.class_names[n] for n in set(lookup.keys())]
            if len(self.class_names) > 0
            else set(lookup.keys())
        )
        ax.grid(True)
        if out_path is not None:
            plt.savefig(out_path)
        image = render_figure_to_array(fig)
        runner.log_image(tag=self.prefix, image=image, scope="loader")
