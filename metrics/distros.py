import os

import numpy as np

import torch
from torch import Tensor
from torch.distributions import Normal


class DataDistributions:
    def __init__(self, run: str, device: torch.device | str) -> None:
        self.run = run
        self.device = device

        post_train = torch.load(
            os.path.join("runs", self.run, "post_train.pt"), weights_only=False
        )
        self.post_train = self.__read_post_train(post_train)

        self.dist_kdes = self.post_train["dist_kde"]
        self.dist_quants = self.post_train["dist_quant"]
        self.pot_normal = Normal(
            loc=self.post_train["pot_loc"], scale=self.post_train["pot_sd"]
        )

    def __read_post_train(self, post_train: dict) -> dict:
        shape = (len(post_train),)
        compressed = {
            "dist_kde": [],
            "dist_quant": torch.empty(shape, dtype=torch.float32, device=self.device),
            "pot_loc": torch.empty(shape, dtype=torch.float32, device=self.device),
            "pot_sd": torch.empty(shape, dtype=torch.float32, device=self.device),
        }

        for cat in post_train:
            compressed["dist_kde"].append(post_train[cat]["dist_kde"])
            compressed["dist_quant"][cat] = post_train[cat]["dist_quant"]
            compressed["pot_loc"][cat] = post_train[cat]["pot_loc"]
            compressed["pot_sd"][cat] = post_train[cat]["pot_sd"]

        return compressed

    def get_dist_nll(self, x: Tensor, argmins: Tensor) -> Tensor:
        """Computes the negative log likelihood of a distance belonging to a certain
        distance KDE

        Args:
            x (Tensor): distance tensor (B, c)
            argmins (Tensor): argmins to know which KDE to aim for (B,)

        Returns:
            Tensor: negative log likelihood of a distance belonging to a KDE
        """

        # make output
        likelihoods = torch.zeros_like(argmins, dtype=torch.float32)

        # get argmin'd distances
        dists = x[torch.arange(x.shape[0]), argmins].cpu().numpy()
        argmins_np = argmins.cpu().numpy()

        # get unique argmins for easy indexing
        unique_cats = np.unique(argmins_np)

        # go over each cat in unique cats and update likelhoods
        for cat in unique_cats:
            mask = argmins_np == cat

            # eval dist likelihoods
            cat_likelihoods = self.dist_kdes[cat].evaluate(dists[mask])
            likelihoods[mask] = torch.from_numpy(cat_likelihoods).float().to(x.device)

        return -torch.log(likelihoods + 1e-8)

    def is_anomaly(self, preds: Tensor, nlls: Tensor) -> Tensor:
        """Computes which predictions are OOD based on their NLL

        Args:
            preds (Tensor): class predictions (B,)
            nlls (Tensor): negative log likelihoods (B,)

        Returns:
            Tensor: OOD predictions masked out with -1's
        """
        # get anomaly masks
        thresholds = self.dist_quants[preds]
        anomaly_mask = nlls > thresholds

        # mask out anomalies
        new_preds = preds.clone()
        new_preds[anomaly_mask] = -1.0

        return new_preds

    def get_pot_likelihoods(self, x: Tensor) -> Tensor:
        return self.pot_normal.log_prob(x)


if __name__ == "__main__":
    dd = DataDistributions("all_data_conv_1k", "cuda")
    print(dd.post_train)

    x = torch.rand((10, 5), device="cuda", dtype=torch.float32) + 20
    print(x)

    preds = x.argmin(dim=-1)
    print(preds)

    nll = dd.get_dist_nll(x, preds)
    print(nll)

    preds = dd.is_anomaly(preds, nll)
    print(preds)
