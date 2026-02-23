import os
import shutil
import logging
from datetime import datetime, timezone
from typing import Optional

import matplotlib.pyplot as plt

from torch import Tensor

from faster_rcnn.configs import DataConfig, TrainConfig, load_yaml


class Run:
    def __init__(
        self,
        data_cfg_path: str,
        train_cfg_path: str,
        name: Optional[str],
        train: bool,
    ) -> None:
        # set name and train state
        self.name = name if name else datetime.now(timezone.utc).isoformat()

        # load configs
        self.data_cfg = DataConfig(**load_yaml(data_cfg_path)["data_config"])
        self.train_cfg = TrainConfig(**load_yaml(train_cfg_path)["train_config"])

        # set run dirs
        self.run_dir = os.path.join("runs", self.name)
        self.plot_dir = os.path.join(self.run_dir, "plots")
        self.cfg_dir = os.path.join(self.run_dir, "configs")
        self.eval_dir = os.path.join(self.run_dir, "evals")
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")

        # set up checkpoints filenames
        self.checkpoint_fname = os.path.join(self.checkpoints_dir, "checkpoint_{}.pt")

        if train:
            self.__train_init(data_cfg_path, train_cfg_path)

    @staticmethod
    def init_from_name(name: str) -> "Run":
        run_configs_dir = os.path.join("runs", name, "configs")

        return Run(
            data_cfg_path=os.path.join(run_configs_dir, "data.cfg.yaml"),
            train_cfg_path=os.path.join(run_configs_dir, "train.cfg.yaml"),
            name=name,
            train=False,
        )

    def __train_init(self, data_cfg_path: str, train_cfg_path: str) -> None:
        # make dirs for run
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.cfg_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # copy configs to run
        shutil.copy(data_cfg_path, os.path.join(self.cfg_dir, "data.cfg.yaml"))
        shutil.copy(train_cfg_path, os.path.join(self.cfg_dir, "train.cfg.yaml"))

        # make logger
        logging.basicConfig(
            filename=os.path.join(self.run_dir, "run_log.log"),
            filemode="w",
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        # set up loss tracking batch/epoch
        self.batch_losses = {"total": 0.0}
        self.epoch_losses = {"total": []}

    def update_batch_loss(self, loss: Tensor) -> None:
        self.batch_losses["total"] += loss.item()

    def update_epoch_loss(self, num_batches: int) -> None:
        self.epoch_losses["total"].append(self.batch_losses["total"] / num_batches)

        self.batch_losses["total"] = 0.0  # reset loss

    def log_state(self, epoch: int) -> None:
        loss_str = " | ".join(
            [f"{name}: {values[-1]:.6f}" for name, values in self.epoch_losses.items()]
        )

        logging.info(f"Epoch {epoch:04d} | {loss_str}")  # pylint: disable=W1203

    def plot_losses(self) -> None:
        for loss, loss_vals in self.epoch_losses.items():
            plt.plot(loss_vals, label=loss)

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, "losses.pdf"), format="pdf")
