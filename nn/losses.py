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
    reg_weight: float = 1e-4,
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
    # make sigma narrower as time goes on
    sigma = conv_sigma * (1 - t).view(-1, 1, 1, 1)

    # jitter xt with noise to have wider paths
    xt_jittered = xt + torch.randn_like(xt) * sigma
    xt_jittered = xt_jittered.detach().requires_grad_(True)

    # get potential
    potential = net.forward(xt_jittered, t.unsqueeze(1))

    # potential regularization, to keep the +C part near 0
    reg = (potential - potential.mean()).square().mean() * reg_weight

    # get speed as the negative gradient of the potential
    dxt_hat = -1.0 * __gradient(potential.sum(), xt_jittered, create_graph=True)

    # mse velocity loss
    mse = (dxt_hat - dxt).square().mean()

    # get difference between speeds
    return mse + reg


def divergence_loss(
    xt: Tensor,
    t: Tensor,
    net: TimeDependentModule,
    barrier: float = 1.0,
    div_sigma: float = 0.1,
    tmax: float = 1.0,
    **kwargs
) -> Tensor:
    """Loss for OOD points. These should have high potential bariers to get into
    respective class prototypes, thus hindering stuff OOD to flow inwards

    Args:
        xt (Tensor): input sampled over probability path, shape (B, C, H, W)
        t (Tensor): time from when input is sampled, shape (B,)
        net (TimeDependentModule): network to predict point potential
        rel_margin (float, optional): relative difference between the potentials. Defaults to 0.1.
        div_sigma (float): standard deviation for the noise to apply to data. Defaults to 0.1.

    Returns:
        Tensor: loss pushing the energy landscape to have barriers around in distribution data
    """
    # make sigma narrower as time goes on, but cap it
    sigma = torch.where(t <= tmax, div_sigma * (1 - t), div_sigma * (1 - tmax))

    # sample points "off-path"
    xt_noise = xt + torch.randn_like(xt) * sigma.view(-1, 1, 1, 1)

    # get potentials for both, we want noise to have higher potential than data
    data_potential = net.forward(xt, t.view(-1, 1))
    noise_potential = net.forward(xt_noise, t.view(-1, 1))

    # hinge loss for potential forcing it to be higher
    diff = noise_potential - data_potential
    return torch.relu(barrier - diff).mean()


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
        # no need to always compute all losses
        # if it's lambda is 0 then it means that it won't contribute this epoch
        # still needs grad though, to be able to do total_loss.backward()
        if lambdas[i] == 0.0:
            retval[loss] = torch.tensor(0.0, requires_grad=True, device=x.device)
            continue

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
}


def anneal_lambda(
    lambda_int: tuple[float, float], warmup_int: tuple[int, int], current_epoch: int
) -> float:
    """Anneals lambda based on it's start and end intervals,
    w.r.t warmup start and end, and current epoch

    Args:
        lambda_int (tuple[float, float]): start and end lambda values
        warmup_int (tuple[int, int]): start epoch and end epoch of warmup
        current_epoch (int): current training epoch

    Returns:
        float: a smooth S curve for lambda rampup starting with lambda_int[0] form warmup_int[0]
        and capping at lambda_int[1] from warmup_int[1]
    """
    start_l, end_l = lambda_int
    start_e, end_e = warmup_int

    # handle boundaries
    if current_epoch < start_e:
        return 0.0
    if current_epoch >= end_e:
        return end_l

    # compute lambda weight [0, 1]
    progress = (current_epoch - start_e) / (end_e - start_e)

    # apply sigmoid transformation
    # map progress to a sigmoid input range [x_min, x_max].
    # -6 to 6 covers the majority of the S-curve transition.
    x_min, x_max = -6, 6
    x = x_min + (x_max - x_min) * progress

    sigmoid_val = 1 / (1 + math.exp(-x))

    # scale and shift to match lambda_int
    return start_l + (end_l - start_l) * sigmoid_val
