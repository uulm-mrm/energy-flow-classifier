from abc import abstractmethod

from torch import nn, Tensor


class TimeDependentModule(nn.Module):
    """Base nn.Module class extended to have time as 2nd forward input"""

    @abstractmethod
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Regular input + time embedding"""


class TimeDependentSequential(nn.Sequential, TimeDependentModule):  # type: ignore
    """Sequential that can take time as input as well"""

    def forward(self, x: Tensor, t: Tensor):  # type: ignore pylint: disable=W0221
        for layer in self:
            if isinstance(layer, TimeDependentModule):
                x = layer.forward(x, t)
            else:
                x = layer.forward(x)

        return x
