import torch
from torch import Tensor


class AffineScheduler:
    def alpha(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def sigma(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def d_alpha(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def d_sigma(self, t: Tensor) -> Tensor:
        raise NotImplementedError


class OTScheduler(AffineScheduler):
    def alpha(self, t: Tensor) -> Tensor:
        return t

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - t

    def d_alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def d_sigma(self, t: Tensor) -> Tensor:
        return -torch.ones_like(t)


class PolyScheduler(AffineScheduler):
    def __init__(self, n: float) -> None:
        super().__init__()

        self.n = n

    def alpha(self, t: Tensor) -> Tensor:
        return torch.pow(t, self.n)

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - torch.pow(t, self.n)

    def d_alpha(self, t: Tensor) -> Tensor:
        return self.n * torch.pow(t, self.n - 1)

    def d_sigma(self, t: Tensor) -> Tensor:
        return -self.n * torch.pow(t, self.n - 1)
