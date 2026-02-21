import os

import torch
from torch import Tensor
from torch.distributions import Gamma


class MultiGamma:
    def __init__(self, run: str, device: torch.device | str) -> None:
        self.run = run
        self.device = device

        post_train = torch.load(os.path.join("runs", self.run, "post_train.pt"))
        self.shape_loc_scale = self.__read_post_train(post_train)

        self.gamma = Gamma(
            concentration=self.shape_loc_scale["shape"],
            rate=1.0 / self.shape_loc_scale["scale"],
        )

    def __read_post_train(self, post_train: dict) -> dict[str, Tensor]:
        shape = (len(post_train),)
        compressed = {
            "shape": torch.empty(shape, dtype=torch.float32, device=self.device),
            "loc": torch.empty(shape, dtype=torch.float32, device=self.device),
            "scale": torch.empty(shape, dtype=torch.float32, device=self.device),
        }

        for cat in post_train:
            compressed["shape"][cat] = post_train[cat]["shape"]
            compressed["loc"][cat] = post_train[cat]["loc"]
            compressed["scale"][cat] = post_train[cat]["scale"]

        return compressed

    def get_likelihoods(self, x: Tensor) -> Tensor:
        # subtract loc from x to center it back to 0, because torch assumes 0 center
        # clamp the minimum distance to close to 0 because gamma starts at p(0) = inf
        x = torch.clamp_min(x - self.shape_loc_scale["loc"], 1e-6)

        return self.gamma.log_prob(x)
