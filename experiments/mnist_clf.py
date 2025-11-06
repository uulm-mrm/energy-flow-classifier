import os

from tqdm import tqdm

import matplotlib.pyplot as plt

import torch

from flow_matching.flow_matching import (
    AffineMultiPath,
    AffinePath,
    ODEProcess,
    RungeKuttaIntegrator,
    tableaus,
)
from flow_matching.flow_matching.distributions import MultiIndependentNormal
from flow_matching.flow_matching.scheduler import CosineScheduler
from flow_matching.modules.utils import EMA

from models.unet import UNet

from data.mnist import get_mnist, MNISTSampler


# set cublas to be deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(mode=True)

    # consts
    device = "cuda:0"
    mnist_shape = (1, 32, 32)
    classes = (0,)

    num_classes = len(classes)
    r = 5.0
    sigma = 1.0

    batch_size = 1024
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
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=epochs // 10, gamma=0.1)

    # training
    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x in mnist_sampler:
            optim.zero_grad()

            x0 = multi_normal.sample(x[0].shape[0])
            x1 = torch.stack(x, dim=0)
            t = torch.rand((x[0].shape[0],), dtype=torch.float32, device=device)

            path_sample = path.sample(x0, x1, t)
            dxt_hat = net.forward(path_sample.xt, path_sample.t)

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            optim.step()

            ema.update_ema_t()

            epoch_loss = epoch_loss + loss

        sched.step()
        pbar.set_description(f"Loss: {(epoch_loss / mnist_sampler.batches):.3f}")

    ema.to_model()
    net = net.eval()

    proc = ODEProcess(net, RungeKuttaIntegrator(tableaus.RK4_TABLEAU, device=device))
    ode_steps = 200

    # plot a couple of trajectories
    intervals = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device).expand(
        multi_normal.means.shape[0], 2
    )

    _, x_traj = proc.sample(multi_normal.means, intervals, steps=ode_steps)

    sols = x_traj[-1].detach().cpu().numpy()

    for sol in sols:
        plt.imshow(sol[0], "gray")
        plt.show()

    # classify a couple of samples
    # indices = torch.cat(
    #     [mnist_sampler.indices[0][:1], mnist_sampler.indices[1][:1]], dim=0
    # )
    # x = mnist_sampler.data[indices]

    # intervals = torch.tensor([[1.0, 0.0]])


if __name__ == "__main__":
    main()
