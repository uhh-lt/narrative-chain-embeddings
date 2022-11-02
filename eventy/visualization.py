import itertools

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback
from catalyst.contrib.utils.visualization import render_figure_to_array


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names=None,
    normalize=False,
    title="confusion matrix",
    fname=None,
    show=True,
    figsize=12,
    fontsize=32,
    colormap="Blues",
    show_numbers=True,
):
    """Render the confusion matrix and return matplotlib"s figure with it.
    Normalization can be applied by setting `normalize=True`.

    Args:
        cm: numpy confusion matrix
        class_names: class names
        normalize: boolean flag to normalize confusion matrix
        title: title
        fname: filename to save confusion matrix
        show: boolean flag for preview
        figsize: matplotlib figure size
        fontsize: matplotlib font size
        colormap: matplotlib color map
        show_numbers: boolean to indicate if each cell should have a number

    Returns:
        matplotlib figure
    """
    plt.ioff()

    cmap = plt.cm.__dict__[colormap]

    if class_names is None:
        class_names = [str(i) for i in range(len(np.diag(cm)))]

    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]

    plt.rcParams.update({"font.size": int(fontsize / np.log2(len(class_names)))})

    figure = plt.figure(figsize=(figsize, figsize))
    plt.title(title)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")

    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    if show_numbers:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    if fname is not None:
        plt.savefig(fname=fname)

    if show:
        plt.show()

    plt.ion()
    return figure


class CustomConfusionMatrixCallback(ConfusionMatrixCallback):
    def __init__(self, *args, show_numbers: bool = True, **kwargs):
        self.show_numbers = show_numbers
        super().__init__(*args, **kwargs)

    def on_loader_end(self, runner: "IRunner"):
        """Loader end hook.

        Args:
            runner: current runner
        """
        confusion_matrix = self.confusion_matrix.compute()
        fig = plot_confusion_matrix(
            confusion_matrix,
            class_names=self.class_names,
            normalize=self.normalize,
            show=False,
            show_numbers=False,
            **self._plot_params,
        )
        image = render_figure_to_array(fig)
        runner.log_image(tag=self.prefix, image=image, scope="loader")
