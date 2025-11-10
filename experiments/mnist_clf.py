from tqdm import tqdm

import matplotlib.pyplot as plt

import torch

from flow_matching.flow_matching import (
    AffineMultiPath,
    AffinePath,
    ODEProcess,
    RungeKuttaIntegrator,
)
from flow_matching.flow_matching.distributions import MultiIndependentNormal
from flow_matching.flow_matching.integrator_utils import RK4_TABLEAU
from flow_matching.flow_matching.scheduler import CosineScheduler
from flow_matching.modules.utils import EMA

from models.unet import UNet

from data.mnist import get_mnist, MNISTSampler


def main():
    torch.manual_seed(42)

    # consts
    device = "cuda:0"
    mnist_shape = (1, 32, 32)
    classes = tuple(range(3))

    num_classes = len(classes)
    r = 5.0
    sigma = 1.0

    batch_size = 512
    t_dims = 256
    lr = 1e-3
    epochs = 1024

    # dataset
    mnist = get_mnist("train")
    mnist_sampler = MNISTSampler(
        mnist, classes=classes, batch_size=batch_size, device=device, skip_last=True
    )

    multi_normal = MultiIndependentNormal(
        c=num_classes, shape=mnist_shape, r=r, sigma=sigma, device=device
    )

    # model stuff
    net = UNet(in_c=1, out_c=1, features=[32, 64, 128], t_dims=t_dims).to(device)
    ema = EMA(net, rate=0.999)
    path = AffineMultiPath(AffinePath(CosineScheduler()), num_classes)

    optim = torch.optim.AdamW(net.parameters(), lr=lr)

    # training
    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x in mnist_sampler:
            optim.zero_grad()

            x1 = multi_normal.sample(x[0].shape[0])
            x0 = torch.stack(x, dim=0)
            t = torch.rand((x[0].shape[0],), dtype=torch.float32, device=device)

            path_sample = path.sample(x0, x1, t)
            dxt_hat = net.forward(path_sample.xt, path_sample.t)

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            optim.step()

            ema.update_ema_t()

            epoch_loss = epoch_loss + loss

        pbar.set_description(f"Loss: {(epoch_loss / mnist_sampler.batches):.3f}")

    ema.to_model()
    net = net.eval()

    proc = ODEProcess(net, RungeKuttaIntegrator(RK4_TABLEAU, device=device))
    ode_steps = 100

    _, x_traj = proc.sample(
        multi_normal.means,
        ints=torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device),
        steps=ode_steps,
    )

    sols = x_traj[-1].detach().cpu().numpy()

    for i, sol in enumerate(sols):
        plt.imshow(sol[0], "gray")
        plt.savefig(f"./plots/sol{i}.jpg")


if __name__ == "__main__":
    main()
