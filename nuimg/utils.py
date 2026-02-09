# pylint: disable=C0103, E1102

import torch
from torch import Tensor
import torch.nn.functional as F

from flow_matching.flow_matching import AffinePath
from flow_matching.modules.utils import TimeDependentModule


def gradient(y: Tensor, x: Tensor, create_graph: bool = False) -> Tensor:
    """Calculate the gradient of y with respec to x
    Wrapper function for torch.autograd.grad for ease of use

    Initiates grad_outputs as ones, and doesn't create the graph by default

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


def get_data_loss(
    x: Tensor, y: Tensor, net: TimeDependentModule, path: AffinePath
) -> Tensor:
    """The ordinary loss for learning the vector field

    Args:
        x (Tensor): _description_
        y (Tensor): _description_
        net (TimeDependentModule): _description_
        path (AffinePath): _description_

    Returns:
        Tensor: _description_
    """

    # sample t
    t = torch.rand((x.shape[0],), dtype=x.dtype, device=x.device)

    # sample path for xt
    path_sample = path.sample(x, y, t)
    xt = path_sample.xt.detach().requires_grad_(True)

    # get potential
    potential = net.forward(xt, t.unsqueeze(1))

    # get speed as the negative gradient of the potential
    dxt_hat = gradient(potential.sum(), xt, create_graph=True)

    # get difference between speeds
    return (dxt_hat - path_sample.dxt).square().mean()


def get_noise_loss(
    x: Tensor,
    y: Tensor,
    net: TimeDependentModule,
    path: AffinePath,
    blanket: tuple[float, float] = (-10.0, 10.0),
) -> Tensor:
    """
    cos sim + grad norm

    grad norm to push them outward slowly and not mega quickly
    and also for it to be easily overridden by data loss


    Args:
        x (Tensor): _description_
        y (Tensor): _description_
        net (TimeDependentModule): _description_
        path (AffinePath): _description_
        blanket (tuple[float, float], optional): _description_. Defaults to (-10.0, 10.0).

    Returns:
        Tensor: _description_
    """

    # sample x noise from U[a, b]
    x_noise = torch.empty_like(x).uniform_(*blanket)

    # sample time for it
    t = torch.rand((x.shape[0],), dtype=x.dtype, device=x.device)

    # sample noise path
    noise_path_sample = path.sample(x_noise, y, t)
    xt_noise = noise_path_sample.xt.detach().requires_grad_(True)

    # get velocity of noise
    potential = net.forward(xt_noise, t.unsqueeze(1))
    dxt_noise = -gradient(potential.sum(), xt_noise, create_graph=True)

    grad_norm = dxt_noise.norm(2, dim=-1)

    # grad loss is E[(||Fi|| - 1)^2] try to minimize this
    grad_loss = (grad_norm - 1.0).square().mean()

    # try to also minimize cos sim for noise, so it flows "away" from target
    # since cossim is [-1, 1] minimizing it means "flow away"
    cos_sim = F.cosine_similarity(dxt_noise, noise_path_sample.dxt, dim=-1).mean()

    # return the total loss of noise
    grad_lambda = 0.1
    return cos_sim + grad_lambda * grad_loss


def cosine_similarity(sols: Tensor, deltas: Tensor, signal_dims: int) -> Tensor:
    """Returns cosine similarity between process solutions and dirac deltas
    that the solutions target

    Args:
        sols (Tensor): solutions tensor size [B, D...]
        deltas (Tensor): dirac delta points size [n, D...]
        signal_dims (int): number of dimensions in which the signal lies

    Returns:
        Tensor: Cosine similarity for each solution w.r.t each delta size [B, n]
    """

    # flatten
    sols = sols.view(sols.shape[0], -1)
    deltas = deltas.view(deltas.shape[0], -1)

    sols = sols[:, :signal_dims]
    deltas = deltas[:, :signal_dims]

    # normalizes using L2 along feature dims so that only matmul is needed for cos sim
    sols = F.normalize(sols, p=2, dim=-1)
    deltas = F.normalize(deltas, p=2, dim=-1)

    return sols @ deltas.T  # (B, n)


def norm_decay(
    sols: Tensor, signal_dims: int, r: float = 1.0, alpha: float = 1.0
) -> Tensor:
    """Decays the norm of the solutions agains a sphere of radius r

    Args:
        sols (Tensor): solutions tensor size [B, D...]
        signal_dims (int): number of dimensions in which the signal lies
        r (float, optional): sphere radius. Defaults to 1.0.
        alpha (float, optional): decay factor. Defaults to 1.0.

    Returns:
        Tensor: decay result in (0, 1] size (B,)
    """
    sols = sols.view(sols.shape[0], -1)
    sols = sols[:, :signal_dims]

    # exponential decay nice and smooth and somewhat slow
    # good middleground between rational and gaussian
    sols = torch.norm(sols, p=2, dim=1)
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

    meas = cosine_similarity(sols, deltas, 2)
    qual = norm_decay(sols, 2)

    print(credal_measures((meas + 1) * 0.5, qual, W=3))


if __name__ == "__main__":
    main()
