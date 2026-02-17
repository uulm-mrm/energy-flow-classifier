from typing import Any, Literal
from dataclasses import dataclass

from data import ExportConfig


@dataclass
class DataConfig:
    dataset: Literal["nuimages-v1.0", "COCO"]
    version: str

    batch_size: int
    shuffle: bool
    skip_last: bool

    @property
    def export_cfg(self):
        return ExportConfig(self.dataset, self.version)


@dataclass
class TrainConfig:
    learn_rate: float
    epochs: int

    # which losses to run with
    losses: list[Literal["convergence", "divergence", "eikonal", "prototype"]]
    lambdas: list[float]  # lambdas for losses in order
    losses_kwargs: dict[str, Any]  # any kwargs for the any specific losses
