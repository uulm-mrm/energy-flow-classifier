# pylint: disable=W0613

import math
from typing import Callable

import torch
from torch import Tensor

from fm.flow_matching import AffinePath
from fm.modules import TimeDependentModule

__all__ = ["LOSS_DICT", "apply_losses", "anneal_lambda"]


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
    xt: Tensor,
    dxt: Tensor,
    t: Tensor,
    net: TimeDependentModule,
    conv_sigma: float = 0.01,
    **kwargs
) -> Tensor:
    """Loss for in distribution data flowing towards their respective
    class prototypes

    Args:
        xt (Tensor): input sampled over probability path, shape (B, C, H, W)
        dxt (Tensor): velocity induced by the probability path, shape (B, C, H, W)
        t (Tensor): time from when input is sampled, shape (B,)
        net (TimeDependentModule): network to predict point potential
        conv_sigma (float): standard deviation for the noise to apply to data. Defaults to 0.01.

    Returns:
        Tensor: Convergence loss for in distribution points
    """

    # jitter xt with noise to have wider paths
    xt_jittered = xt + torch.randn_like(xt) * conv_sigma
    xt_jittered = xt_jittered.detach().requires_grad_(True)

    # get potential
    potential = net.forward(xt_jittered, t.unsqueeze(1))

    # get speed as the negative gradient of the potential
    dxt_hat = -1.0 * __gradient(potential.sum(), xt_jittered, create_graph=True)

    # get difference between speeds
    return (dxt_hat - dxt).square().mean()


def divergence_loss(
    xt: Tensor,
    t: Tensor,
    net: TimeDependentModule,
    barrier: float = 1.0,
    div_sigma: float = 0.1,
    **kwargs
) -> Tensor:
    """Loss for OOD points. These should have high potential bariers to get into
    respective class prototypes, thus hindering stuff OOD to flow inwards

    Args:
        xt (Tensor): input sampled over probability path, shape (B, C, H, W)
        t (Tensor): time from when input is sampled, shape (B,)
        net (TimeDependentModule): network to predict point potential
        barrier (float, optional): potential barrier height which to aim for. Defaults to 1.0.
        div_sigma (float): standard deviation for the noise to apply to data. Defaults to 0.1.

    Returns:
        Tensor: loss pushing the energy landscape to have barriers around in distribution data
    """

    # sample points "off-path"
    xt_noise = xt + torch.randn_like(xt) * div_sigma

    # get potentials for both, we want noise to have higher potential than data
    data_potential = net.forward(xt, t.view(-1, 1))
    noise_potential = net.forward(xt_noise, t.view(-1, 1))

    # hinge loss for potential forcing it to be higher
    return torch.relu(barrier - (noise_potential - data_potential)).mean()


def eikonal_loss(
    xt: Tensor,
    dxt: Tensor,
    t: Tensor,
    net: TimeDependentModule,
    eik_sigma: float = 0.05,
    **kwargs
) -> Tensor:
    """Calculates eikonal loss around data to keep velocities low on whole domain

    Args:
        xt (Tensor): input sampled over probability path, to copy information from,
            shape (B, C, H, W)
        dxt (Tensor): velocity induced by the probability path, shape (B, C, H, W)
        t (Tensor): time from when input is sampled, shape (B,)
        net (TimeDependentModule): network to predict point potential
        eik_sigma (float): deviation for noise around input. Defaults to 0.05.

    Returns:
        Tensor: eikonal loss for noisy data
    """

    # sample around data
    xt_proximal = xt + torch.randn_like(xt) * eik_sigma
    xt_proximal = xt_proximal.detach().requires_grad_(True)

    # get velocity
    proximal_potential = net.forward(xt_proximal, t)
    grad_phi = __gradient(proximal_potential.sum(), xt_proximal, create_graph=True)

    # calc norm
    grad_norm = grad_phi.flatten(start_dim=1).norm(p=2, dim=-1)
    target_norm = dxt.flatten(start_dim=1).norm(p=2, dim=-1).detach()

    # calc loss as the difference between the expected velocity for the batch
    # vs the computed loss for the batch
    return (grad_norm - target_norm).square().mean()


def prototype_loss(
    prototypes: Tensor, net: TimeDependentModule, sink: float = -1.0, **kwargs
) -> Tensor:
    """Aims to keep energy of prototype points around the passed sink

    Args:
        prototypes (Tensor): class prototypes from the dataset
        net (TimeDependentModule): network to predict point potential
        sink (float, optional): potential which to aim for. Defaults to -5.0.

    Returns:
        Tensor: Hinge loss with sink as bound
    """
    t = torch.ones(
        (prototypes.shape[0], 1), dtype=prototypes.dtype, device=prototypes.device
    )

    prototype_potential = net.forward(prototypes, t)

    return (prototype_potential - sink).square().mean()


def apply_losses(
    x: Tensor,
    y: Tensor,
    prototypes: Tensor,
    path: AffinePath,
    net: TimeDependentModule,
    losses: dict[str, Callable[..., Tensor]],
    lambdas: list[float],
    **losses_kwargs
) -> dict[str, Tensor]:
    """Applies all losses specified in the list multiplied by their lambdas

    Args:
        x (Tensor): input at t=0, shape (B, C, H, W)
        y (Tensor): class ptototypes at t=1, shape (B, C, H, W)
        prototypes (Tensor): unique prototypes from dataset, shape (num class, C, H, W)
        path (AffinePath): probability path which to sample
        net (TimeDependentModule): network that computes potentials
        losses (dict[str, Callable[..., Tensor]]): a list of losses to compute
        labmdas (list[float]): scaling coefficients for those losses

    Returns:
        dict[str, Tensor]: loss for each loss in losses
    """

    t = torch.rand((x.shape[0],), dtype=x.dtype, device=x.device)
    path_sample = path.sample(x, y, t)

    retval = {}
    for i, (loss, loss_fn) in enumerate(losses.items()):
        retval[loss] = lambdas[i] * loss_fn(
            xt=path_sample.xt,
            dxt=path_sample.dxt,
            t=t,
            net=net,
            prototypes=prototypes,
            **losses_kwargs
        )

    return retval


LOSS_DICT = {
    "convergence": convergence_loss,
    "divergence": divergence_loss,
    "eikonal": eikonal_loss,
    "prototype": prototype_loss,
}


def anneal_lambda(l: float, e: int, warmup: int) -> float:
    """Anneals loss coefficient lambda w.r.t epoch and warmup.
    Assumes start lambda is 0 by default, and that l is the end value after warmup

    Args:
        l (float): lambda to anneal
        e (int): current epoch
        warmup (int): warmup epochs for lambda

    Returns:
        float: lambda if e >= warmup, otherwise an s curve rampup
    """
    if e >= warmup:
        return l

    # sigmoid type warmup, nice for gradients and loss
    return l / (1 + math.exp(-10 * (e / warmup - 0.5)))
