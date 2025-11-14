from tqdm import tqdm

import matplotlib.pyplot as plt

import torch

from flow_matching.flow_matching import (
    AffineMultiPath,
    AffinePath,
)
from flow_matching.flow_matching.distributions import MultiIndependentNormal
from flow_matching.flow_matching.scheduler import CosineScheduler
from flow_matching.modules.utils import EMA

from models.unet import UNet

from data.mnist import get_mnist, MNISTSampler

from experiments.mnist.consts import CONFIG


def train():
    torch.manual_seed(42)

    # dataset
    mnist = get_mnist("train")
    mnist_sampler = MNISTSampler(
        mnist,
        classes=CONFIG.classes,
        batch_size=CONFIG.batch_size,
        device=CONFIG.device,
        skip_last=True,
    )

    multi_normal = MultiIndependentNormal(
        c=CONFIG.num_classes,
        shape=CONFIG.shape,
        r=CONFIG.r,
        sigma=CONFIG.sigma,
        device=CONFIG.device,
    )

    # model stuff
    net = UNet(in_c=1, out_c=1, features=CONFIG.features, t_dims=CONFIG.t_dims).to(
        CONFIG.device
    )
    ema = EMA(net, rate=0.999)
    path = AffineMultiPath(AffinePath(CosineScheduler()), CONFIG.num_classes)

    optim = torch.optim.AdamW(net.parameters(), lr=CONFIG.lr)
    losses = []

    # training
    for _ in (pbar := tqdm(range(CONFIG.epochs))):
        epoch_loss = 0.0

        for x in mnist_sampler:
            optim.zero_grad()

            x_data = torch.stack(x, dim=0)
            x_noise = multi_normal.sample(x[0].shape[0])
            t = torch.rand((x[0].shape[0],), dtype=torch.float32, device=CONFIG.device)

            # path_sample = path.sample(x_data, x_noise, t)
            path_sample = path.sample(x_noise, x_data, t)
            dxt_hat = net.forward(path_sample.xt, path_sample.t)

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)

            optim.step()

            ema.update_ema_t()

            epoch_loss = epoch_loss + loss.item()

        losses.append(epoch_loss / mnist_sampler.batches)
        pbar.set_description(f"Loss: {(epoch_loss / mnist_sampler.batches):.3f}")

    ema.to_model()
    net = net.eval()

    torch.save(net.state_dict(), f"trained/mnist_{CONFIG.num_classes}.pt")

    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    train()
