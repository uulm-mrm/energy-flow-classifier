import torch
from torch import Tensor

from fm.modules import TimeDependentModule

__all__ = ["LOSS_DICT"]


def __gradient(y: Tensor, x: Tensor, create_graph: bool = True) -> Tensor:
    """Calculate the gradient of y with respec to x
    Wrapper function for torch.autograd.grad for ease of use

    Initiates grad_outputs as ones, and creates the graph by default

    Args:
        y (Tensor): the output of a function to take grad from
        x (Tensor): the input to the function to take grad w.r.t
        create_graph (Tensor): whether to create graph or not

    Returns:
        Tensor: grad of y w.r.t x
    """

    grad_outputs = torch.ones_like(y).detach()

    grad = torch.autograd.grad(
        y, x, grad_outputs=grad_outputs, create_graph=create_graph
    )[0]

    return grad


def convergence_loss(
    xt: Tensor, dxt: Tensor, t: Tensor, net: TimeDependentModule
) -> Tensor:
    """Loss for in distribution data flowing towards their respective
    class prototypes

    Args:
        xt (Tensor): input sampled over probability path, shape (B, C, H, W)
        dxt (Tensor): velocity induced by the probability path, shape (B, C, H, W)
        t (Tensor): time from when input is sampled, shape (B,)
        net (TimeDependentModule): network to predict point potential

    Returns:
        Tensor: Convergence loss for in distribution points
    """

    # sample small noise around xt
    sigma = 0.05
    eps = torch.randn_like(xt) * sigma

    # jitter xt by eps to have wider paths
    xt_jittered = xt + eps
    xt_jittered = xt_jittered.detach().requires_grad_(True)

    # get potential
    potential = net.forward(xt_jittered, t.unsqueeze(1))

    # get speed as the negative gradient of the potential
    dxt_hat = -__gradient(potential.sum(), xt_jittered, create_graph=True)

    # get difference between speeds
    return (dxt_hat - (dxt - eps)).square().mean()


def divergence_loss(
    xt: Tensor, t: Tensor, net: TimeDependentModule, margin: float = 1.0
) -> Tensor:
    """Loss for OOD points. These should have high potential bariers to get into
    respective class prototypes, thus hindering stuff OOD to flow inwards

    Args:
        xt (Tensor): input sampled over probability path, shape (B, C, H, W)
        t (Tensor): time from when input is sampled, shape (B,)
        net (TimeDependentModule): network to predict point potential
        margin (float, optional): potential barrier height which to aim for. Defaults to 1.0.

    Returns:
        Tensor: _description_
    """

    # sample points off-path with much more noise
    sigma = 1.0
    xt_noise = xt + torch.randn_like(xt) * sigma

    # get potentials for both, we want noise to have higher potential than data
    data_potential = net.forward(xt, t.view(-1, 1))
    noise_potential = net.forward(xt_noise, t.view(-1, 1))

    # hinge loss for potential forcing it to be higher
    return torch.relu(margin - (noise_potential - data_potential)).mean()


def eikonal_loss(
    xt: Tensor, t: Tensor, net: TimeDependentModule, blanket: tuple[float, float]
) -> Tensor:
    """Calculates eikonal loss over a blanket to keep velocities low on whole domain

    Args:
        xt (Tensor): input sampled over probability path, to copy information from,
            shape (B, C, H, W)
        t (Tensor): time from when input is sampled, shape (B,)
        net (TimeDependentModule): network to predict point potential
        blanket (tuple[float, float]): a square area from which to sample points,
            should be around where the data actually lies, if not sure just cast a wide blanket

    Returns:
        Tensor: eikonal loss for noise
    """

    # sample a random blanket around data and say it's "sampled" over time
    xt_blanket = torch.empty_like(xt).uniform_(*blanket)
    xt_blanket = xt_blanket.detach().requires_grad_(True)

    # get velocity
    blanket_potential = net.forward(xt_blanket, t)
    blanket_velocity = __gradient(
        blanket_potential.sum(), xt_blanket, create_graph=True
    )

    # calc norm
    velocity_norm = blanket_velocity.flatten(start_dim=1).norm(p=2, dim=-1)

    # calc loss
    return (velocity_norm - 1).square().mean()


def prototype_loss(
    prototypes: Tensor, net: TimeDependentModule, margin: float = -5.0
) -> Tensor:
    """Aims to keep energy of prototype points around the passed margin

    Args:
        prototypes (Tensor): class prototypes from the dataset
        net (TimeDependentModule): network to predict point potential
        margin (float, optional): potential which to aim for. Defaults to -5.0.

    Returns:
        Tensor: Hinge loss with margin as bound
    """
    t = torch.ones(
        (prototypes.shape[0], 1), dtype=prototypes.dtype, device=prototypes.device
    )

    prototype_potential = net.forward(prototypes, t)

    return torch.relu(prototype_potential - margin).mean()


LOSS_DICT = {
    "convergence": convergence_loss,
    "divergence": divergence_loss,
    "eikonal": eikonal_loss,
    "prototype": prototype_loss,
}
