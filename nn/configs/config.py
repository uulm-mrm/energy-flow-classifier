from typing import Any, Literal
from dataclasses import dataclass, field

from data import ExportConfig


@dataclass
class DataConfig:
    dataset: Literal["nuimages-v1.0", "COCO"]
    version: str

    batch_size: int
    shuffle: bool
    skip_last: bool

    data_cfg: ExportConfig = field(init=False)

    def __post_init__(self):
        self.data_cfg = ExportConfig(self.dataset, self.version)


@dataclass
class TrainConfig:
    learn_rate: float
    epochs: int

    # which losses to run with
    losses: list[Literal["convergence", "divergence", "eikonal", "prototype"]]
    lambdas: list[float]  # lambdas for losses in order
    losses_kwargs: dict[str, Any]  # any kwargs for the any specific losses
