from dataclasses import dataclass, field

from torch import Tensor

from .scheduler import AffineScheduler


def broadcast_to(x: Tensor, y: Tensor) -> Tensor:
    """Broadcasts x to y's dimensions

    Args:
        x (Tensor): the source to be broadcast, size (B,)
        y (Tensor): the target to broadcast to, size (B, N, M, ...)

    Returns:
        Tensor: broadcasted x from (B,) to (B, 1...) to match y
    """

    expand_to = len(y.shape) - len(x.shape)

    x = x.view(*x.shape, *([1] * expand_to))

    return x


@dataclass
class PathSample:
    """Provides x at t, speed of x at t, and t broadcast to their dimensions"""

    xt: Tensor = field(metadata={"help": "x at time t based on scheduler"})
    dxt: Tensor = field(
        metadata={"help": "what the speed of x should be at time t based on scheduler"}
    )
    t: Tensor = field(metadata={"help": "t broadcast to xt dimensions"})


class AffinePath:
    """Samples the probability path at different times for the vector field"""

    def __init__(self, sched: AffineScheduler) -> None:
        self.sched = sched

    def sample(self, x0: Tensor, x1: Tensor, t: Tensor) -> PathSample:
        """Samples the probability path to return x at time t and speed of x at t

        Args:
            x0 (Tensor): initial x at time 0, size (B, ...)
            x1 (Tensor): target x at time 1, size (B, ...)
            t (Tensor): time to sample between, size (B,)

        Returns:
            PathSample: dataclass with xt, dxt, t
        """

        t = broadcast_to(t, x0)
        xt = self.sched.alpha(t) * x1 + self.sched.sigma(t) * x0
        dxt = self.sched.d_alpha(t) * x1 + self.sched.d_sigma(t) * x0

        return PathSample(xt, dxt, t)
