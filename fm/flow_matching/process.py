import torch
from torch import Tensor, nn

from .integrator import Integrator


def gradient(y: Tensor, x: Tensor) -> Tensor:
    """Calculate the gradient of y with respec to x
    Wrapper function for torch.autograd.grad for ease of use

    Initiates grad_outputs as ones, and doesn't create the graph by default

    Args:
        y (Tensor): the output of a function to take grad from
        x (Tensor): the input to the function to take grad w.r.t

    Returns:
        Tensor: grad of y w.r.t x
    """

    grad_outputs = torch.ones_like(y).detach()

    grad = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=False)[0]

    return grad


class PotentialProcess:
    """
    Process solver for the gradient based potential manifold,
    that can sample points in the field using the potential
    """

    def __init__(self, potential_manifold: nn.Module, integrator: Integrator) -> None:
        self.potential_manifold = potential_manifold
        self.integrator = integrator

    def sample(
        self, x_init: Tensor, ints: Tensor, steps: int, **pm_extras
    ) -> tuple[Tensor, Tensor]:
        """Integrates the point in vector space using the potential field along the probability path

        Args:
            x_init (Tensor): initial condition of the Process
            ints (Tensor): start and end time points for each point in x_init
            steps (int): number of steps for the Process Integrator

        Returns:
            tuple[Tensor, Tensor]: (steps+1, B, 1) of time
            and (steps+1, B, D...) of solution trajectories
        """

        def diff_eq(t: Tensor, x: list[Tensor]) -> list[Tensor]:
            _x = x[0]

            # to get the velocity we need to allow _x to have grads
            _x.requires_grad_(True)

            with torch.set_grad_enabled(True):
                potential: Tensor = self.potential_manifold.forward(_x, t, **pm_extras)
                velocity = -gradient(potential.sum(), _x)

            return [velocity]

        with torch.no_grad():
            t_traj, x_traj = self.integrator.integrate(
                diff_eq, [x_init], ints, steps=steps
            )

        return t_traj, x_traj[0]
