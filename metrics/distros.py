import os

import torch
from torch import Tensor
from torch.distributions import Gamma, Normal


class DataDistributions:
    def __init__(self, run: str, device: torch.device | str) -> None:
        self.run = run
        self.device = device

        post_train = torch.load(os.path.join("runs", self.run, "post_train.pt"))
        self.post_train = self.__read_post_train(post_train)

        self.dist_gamma = Gamma(
            concentration=self.post_train["dist_shape"],
            rate=1.0 / self.post_train["dist_scale"],
        )
        self.pot_normal = Normal(
            loc=self.post_train["pot_loc"], scale=self.post_train["pot_sd"]
        )

    def __read_post_train(self, post_train: dict) -> dict[str, Tensor]:
        shape = (len(post_train),)
        compressed = {
            "dist_shape": torch.empty(shape, dtype=torch.float32, device=self.device),
            "dist_loc": torch.empty(shape, dtype=torch.float32, device=self.device),
            "dist_scale": torch.empty(shape, dtype=torch.float32, device=self.device),
            "pot_loc": torch.empty(shape, dtype=torch.float32, device=self.device),
            "pot_sd": torch.empty(shape, dtype=torch.float32, device=self.device),
        }

        for cat in post_train:
            compressed["dist_shape"][cat] = post_train[cat]["dist_shape"]
            compressed["dist_loc"][cat] = post_train[cat]["dist_loc"]
            compressed["dist_scale"][cat] = post_train[cat]["dist_scale"]
            compressed["pot_loc"][cat] = post_train[cat]["pot_loc"]
            compressed["pot_sd"][cat] = post_train[cat]["pot_sd"]

        return compressed

    def get_dist_likelihoods(self, x: Tensor) -> Tensor:
        # subtract loc from x to center it back to 0, because torch assumes 0 center
        # clamp the minimum distance to close to 0 because gamma starts at p(0) = inf
        x = torch.clamp_min(x - self.post_train["dist_loc"], 1e-6)

        return self.dist_gamma.log_prob(x)

    def get_pot_likelihoods(self, x: Tensor) -> Tensor:
        return self.pot_normal.log_prob(x)
