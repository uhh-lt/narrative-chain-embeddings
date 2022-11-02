from pathlib import Path

from catalyst.callbacks import Callback

from eventy.config import LoadableConfig


class ConfigCallback(Callback):
    def __init__(
        self, *args, config: LoadableConfig, logdir: Path, save_name: str, **kwargs
    ):
        self.config = config
        self.logdir = logdir
        super().__init__(10, *args, **kwargs)

    def on_experiment_start(self, runner: "IRunner") -> None:
        runner._hparams = self.config.dict()
        self.config.save_file(self.logdir / "config.yaml")
        return super().on_experiment_start(runner)
