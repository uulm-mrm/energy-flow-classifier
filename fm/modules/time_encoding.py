import math

import torch
from torch import Tensor, nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int, max_period: float = 10_000.0) -> None:
        """
        Args:
            emb_dim (int): dims into which to embed time
            max_period (float, optional): up to which period to embed. Defaults to 10_000.0.
        """
        super().__init__()

        assert emb_dim % 2 == 0, "Embedding dimension needs to be divisible by 2"

        self.emb_dim = emb_dim

        # precompute freq consts
        half_dim = emb_dim // 2
        exp = -math.log(max_period) * torch.arange(0, half_dim) / half_dim
        self.register_buffer("freqs", torch.exp(exp))

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t (Tensor): tensor in size (B,) or ()

        Returns:
            Tensor: time embedded to (B, D)
        """

        t = t.reshape(-1, 1)

        # sin and cos emb
        angles = t * self.freqs  # type: ignore
        t_emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return t_emb
