import os
import shutil
import logging
from datetime import datetime, timezone
from typing import Optional

import yaml

import matplotlib.pyplot as plt

from torch import Tensor

from nn.configs.config import DataConfig, TrainConfig


def _load_yaml(path: str) -> dict:
    with open(path, "r+", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


def _copy_yaml(src: str, dest_dir: str) -> None:
    dest = os.path.join(dest_dir, src.split("/")[-1])
    shutil.copy(src, dest)


class Run:
    def __init__(
        self,
        model_cfg_path: str,
        data_cfg_path: str,
        train_cfg_path: str,
        name: Optional[str],
    ) -> None:
        # set name
        self.name = name if name else datetime.now(timezone.utc).isoformat()

        # load configs
        self.model_config: dict = _load_yaml(model_cfg_path)
        self.data_cfg = DataConfig(**_load_yaml(data_cfg_path)["data_config"])
        self.train_cfg = TrainConfig(**_load_yaml(train_cfg_path)["train_config"])

        # set run dirs
        self.run_dir = os.path.join("runs", self.name)
        self.plot_dir = os.path.join(self.run_dir, "plots")
        self.cfg_dir = os.path.join(self.run_dir, "configs")

        # make dirs for run
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.cfg_dir, exist_ok=True)

        # copy configs to run
        _copy_yaml(model_cfg_path, self.cfg_dir)
        _copy_yaml(data_cfg_path, self.cfg_dir)
        _copy_yaml(train_cfg_path, self.cfg_dir)

        # set up loss tracking batch/epoch
        self.batch_losses = {loss_name: 0.0 for loss_name in self.train_cfg.losses}
        self.batch_losses["total"] = 0.0

        self.epoch_losses = {loss_name: [] for loss_name in self.train_cfg.losses}
        self.epoch_losses["total"] = []

        # make logger
        logging.basicConfig(
            filename=os.path.join(self.run_dir, "run_log.log"),
            filemode="w",
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        # set up pt filename
        self.model_fname = os.path.join(self.run_dir, "model.pt")

    def update_batch_loss(self, loss_dict: dict[str, Tensor]) -> None:
        total = 0

        for loss, loss_val in loss_dict.items():
            loss_val = loss_val.item()

            self.batch_losses[loss] += loss_val
            total += loss_val

        self.batch_losses["total"] += total

    def update_epoch_loss(self, num_batches: int) -> None:
        for loss, loss_val in self.batch_losses.items():
            self.epoch_losses[loss].append(loss_val / num_batches)

            self.batch_losses[loss] = 0.0  # reset loss

    def log_state(self, epoch: int) -> None:
        loss_str = " | ".join(
            [f"{name}: {values[-1]:.4f}" for name, values in self.epoch_losses.items()]
        )

        logging.info(f"Epoch {epoch:04d} | {loss_str}")  # pylint: disable=W1203

    def plot_losses(self) -> None:
        for loss, loss_vals in self.epoch_losses.items():
            plt.plot(loss_vals, label=loss)

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, "losses.pdf"), format="pdf")
