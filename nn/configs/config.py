import os
from typing import Any, Literal
from dataclasses import dataclass
from datetime import datetime, timezone

from data import ExportConfig

from nn.configs.yaml_utils import load_yaml


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

    # which losses to run with
    losses: list[Literal["convergence", "divergence", "eikonal", "prototype"]]

    lambdas: list[tuple[float, float]]  # start and end lambdas for losses in order
    warmups: list[tuple[int, int]]  # warmup epochs for each lambda for annealing

    losses_kwargs: dict[str, Any]  # any kwargs specific to a loss


@dataclass
class EvalConfig:
    run_name: str
    checkpoint: int

    # eval dataset
    dataset: Literal["nuimages-v1.0", "COCO"]
    version: str
    batch_size: int

    ode_steps: int

    def get_model_cfg(self) -> dict:
        cfg_path = os.path.join("runs", self.run_name, "configs", "model.cfg.yaml")
        return load_yaml(cfg_path)

    def get_model_weights(self) -> str:
        return os.path.join(
            "runs", self.run_name, "checkpoints", f"checkpoint_{self.checkpoint}.pt"
        )

    def get_export_cfg(self) -> ExportConfig:
        return ExportConfig(self.dataset, self.version)

    def get_prototypes(self) -> str:
        data_cfg_path = os.path.join("runs", self.run_name, "configs", "data.cfg.yaml")
        data_cfg = DataConfig(**load_yaml(data_cfg_path)["data_config"])

        return data_cfg.get_export_cfg().prototypes

    def get_eval_dir(self, name: str | None = None) -> str:
        name = name if name else datetime.now(timezone.utc).isoformat()
        return os.path.join("runs", self.run_name, "evals", name)
