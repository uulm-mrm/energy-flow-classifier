import math

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


def train():
    torch.manual_seed(42)

    # consts
    device = "cuda:0"
    mnist_shape = (1, 32, 32)
    mnist_dims = math.prod(mnist_shape)
    classes = tuple(range(3))

    num_classes = len(classes)

    sigma = 1.0
    k = 3.0
    r = k * sigma * mnist_dims**0.5

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
    net = UNet(in_c=1, out_c=1, features=[64, 128, 256], t_dims=t_dims).to(device)
    ema = EMA(net, rate=0.999)
    path = AffineMultiPath(AffinePath(CosineScheduler()), num_classes)

    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    losses = []

    # training
    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x in mnist_sampler:
            optim.zero_grad()

            # flow from images to noise. should be better than reverse
            x0 = torch.stack(x, dim=0)
            x1 = multi_normal.sample(x[0].shape[0])
            t = torch.rand((x[0].shape[0],), dtype=torch.float32, device=device)

            path_sample = path.sample(x0, x1, t)
            dxt_hat = net.forward(path_sample.xt, path_sample.t)

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)

            optim.step()

            ema.update_ema_t()

            epoch_loss = epoch_loss + loss.item()

        losses.append(epoch_loss / mnist_sampler.batches)
        pbar.set_description(f"Loss: {(epoch_loss / mnist_sampler.batches):.3f}")

    ema.to_model()
    net = net.eval()

    torch.save(net.state_dict(), f"trained/mnist_{num_classes}.pt")

    plt.plot(losses)
    plt.show()


def test():
    torch.manual_seed(42)

    # consts
    device = "cuda:0"
    mnist_shape = (1, 32, 32)
    mnist_dims = math.prod(mnist_shape)
    classes = tuple(range(3))

    num_classes = len(classes)

    sigma = 1.0
    k = 3.0
    r = k * sigma * mnist_dims**0.5

    batch_size = 512
    t_dims = 256

    # dataset
    mnist = get_mnist("test")
    mnist_sampler = MNISTSampler(
        mnist, classes=classes, batch_size=batch_size, device=device, skip_last=True
    )

    multi_normal = MultiIndependentNormal(
        c=num_classes, shape=mnist_shape, r=r, sigma=sigma, device=device
    )

    net = UNet(in_c=1, out_c=1, features=[64, 128, 256], t_dims=t_dims).to(device)
    net.load_state_dict(torch.load(f"trained/mnist_{num_classes}.pt"))
    net = net.eval()

    # process to integrate
    proc = ODEProcess(net, RungeKuttaIntegrator(RK4_TABLEAU, device=device))
    ode_steps = 100

    # generate
    x_noise = multi_normal.means
    intervals = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device).expand(
        x_noise.shape[0], 2
    )
    _, x_traj = proc.sample(x_noise, ints=intervals, steps=ode_steps)
    sols = x_traj[-1].detach().cpu().numpy()

    for sol in sols:
        plt.imshow(sol[0], "gray")
        plt.show()

    # predict
    for c in classes:
        x_pred = mnist_sampler.data[mnist_sampler.indices[c]].to(device)
        intervals = torch.tensor(
            [[0.0, 1.0]], dtype=torch.float32, device=device
        ).expand(x_pred.shape[0], 2)

        _, x_traj = proc.sample(x_pred, intervals, steps=ode_steps)
        sols = x_traj[-1]
        probs = multi_normal.log_likelihood(sols)
        print(f"Class {c} elements: {x_pred.shape[0]}")
        print(f"Correctly classified: {torch.sum(probs.argmax(dim=1) == c).item()}")


if __name__ == "__main__":
    # train()
    test()
