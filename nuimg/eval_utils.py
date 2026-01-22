# pylint: disable=C0103

import torch
from torch import Tensor
import torch.nn.functional as F


def cosine_similarity(sols: Tensor, deltas: Tensor) -> Tensor:
    """Returns cosine similarity between process solutions and dirac deltas
    that the solutions target

    Args:
        sols (Tensor): solutions tensor size [B, D...]
        deltas (Tensor): dirac delta points size [n, D...]

    Returns:
        Tensor: Cosine similarity for each solution w.r.t each delta size [B, n]
    """

    # normalizes using L2 along feature dims so that only matmul is needed for cos sim
    sols = F.normalize(sols.view(sols.shape[0], -1), p=2, dim=-1)
    deltas = F.normalize(deltas.view(deltas.shape[0], -1), p=2, dim=-1)

    return sols @ deltas.T  # (B, n)


def norm_decay(sols: Tensor, r: float = 1.0, alpha: float = 1.0) -> Tensor:
    """Decays the norm of the solutions agains a sphere of radius r

    Args:
        sols (Tensor): solutions tensor size [B, D...]
        r (float, optional): sphere radius. Defaults to 1.0.
        alpha (float, optional): decay factor. Defaults to 1.0.

    Returns:
        Tensor: decay result in (0, 1] size (B,)
    """

    # exponential decay nice and smooth and somewhat slow
    # good middleground between rational and gaussian
    sols = torch.norm(sols.view(sols.shape[0], -1), p=2, dim=1)
    return torch.exp(-alpha * torch.abs(sols - r))


def credal_measures(
    measure: Tensor, quality: Tensor, W: float = 1.0
) -> tuple[Tensor, Tensor]:
    """
    Calculates belief and vacuity as defined by Dirichlet for Credal Sets,
    with evidence being defined as measure * quality

    Args:
        measure (Tensor): >2-monotone positive measure of size (B, n)
        quality (Tensor): quality of the measurement of size (B,)
        W (float, optional): prior strenght as per Dirichlet, usually number of classes.
            Defaults to 1.0.

    Returns:
        tuple[Tensor, Tensor]: belief and vacuity tensors size (B, n) and (B,)
    """

    evidence = measure * quality.unsqueeze(1)
    denom = evidence.sum(dim=-1) + W

    return evidence / denom.unsqueeze(1), W / denom


def main():
    sols = torch.randn((10, 3))
    deltas = torch.randn((2, 3))

    meas = cosine_similarity(sols, deltas)
    qual = norm_decay(sols)

    print(credal_measures((meas + 1) * 0.5, qual, W=3))


if __name__ == "__main__":
    main()
