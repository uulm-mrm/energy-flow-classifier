import os
from typing import Literal
from dataclasses import dataclass
from datetime import datetime, timezone

from data import ExportConfig


@dataclass
class DataConfig:
    dataset: Literal["nuimages-v1.0", "COCO"]
    version: str

    batch_size: int
    shuffle: bool
    skip_last: bool

    def get_export_cfg(self):
        return ExportConfig(self.dataset, self.version)


@dataclass
class TrainConfig:
    learn_rate: float
    epochs: int


@dataclass
class EvalConfig:
    run_name: str
    checkpoint: int

    # eval dataset
    dataset: Literal["nuimages-v1.0", "COCO"]
    version: str
    batch_size: int

    def get_model_weights(self) -> str:
        return os.path.join(
            "runs", self.run_name, "checkpoints", f"checkpoint_{self.checkpoint}.pt"
        )

    def get_export_cfg(self) -> ExportConfig:
        return ExportConfig(self.dataset, self.version)

    def get_eval_dir(self, name: str | None = None) -> str:
        name = name if name else datetime.now(timezone.utc).isoformat()
        return os.path.join("runs", self.run_name, "evals", name)
