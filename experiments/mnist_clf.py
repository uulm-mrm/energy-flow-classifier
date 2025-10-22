from pprint import pprint

from tqdm import tqdm

import torch

from flow_matching.flow_matching import (
    Path,
    ODEProcess,
    MidpointIntegrator,
    GoldenSectionSeeker,
)
from flow_matching.flow_matching.scheduler import OTScheduler
from flow_matching.flow_matching.distributions import GaussianMixture
from flow_matching.flow_matching.utils import push_forward_all
from flow_matching.modules.utils import EMA

from unet import TimeCondUnet
from utils import closest_anchor, anchors_to_class

from data.mnist import get_mnist, MNISTSampler


def main():
    torch.manual_seed(42)

    # consts
    device = "cuda:0"

    classes = (0, 1, 2)
    anchor_times = (0.0, 0.33, 0.66, 1.0)  # manually or with linspace
    time_class_map = dict(zip(anchor_times, (-1, *classes)))

    batch_size = 1024

    t_dims = 64
    lr = 1e-3
    epochs = 200

    mnist = get_mnist("train")
    x_sampler = MNISTSampler(
        mnist, classes=classes, batch_size=batch_size, device=device
    )

    # mnist image dims are 1, 28, 28
    x0_sampler = GaussianMixture(
        n=8, shape=(1, 28, 28), sigma=0.5, r=1.0, device=device
    )

    # model stuff
    unet = TimeCondUnet(in_c=1, filters=32, heads=4, t_dims=t_dims).to(device)
    ema = EMA(unet, rate=0.999).to(device)
    path = Path(OTScheduler())

    optim = torch.optim.AdamW(ema.model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

    # training
    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x in x_sampler:
            optim.zero_grad()

            x0 = x0_sampler.sample(x[0].shape[0])
            anchor_x = (x0, *x)

            loss = push_forward_all(
                anc_x=anchor_x, anc_t=anchor_times, path=path, vf=ema.model
            )

            loss.backward()
            optim.step()

            ema.update_ema_t()

            epoch_loss = epoch_loss + loss

        sched.step()
        pbar.set_description(f"Loss: {(epoch_loss / x_sampler.batches):.3f}")

    ema.to_model()
    unet = ema.model.eval()

    # classification
    integrator = ODEProcess(unet, MidpointIntegrator())
    seeker = GoldenSectionSeeker(max_evals=10)
    ode_steps = 10
    log_p0 = x0_sampler.log_likelihood

    # classify digits 0
    x1 = x_sampler.data[x_sampler.indices[0][:1000]]

    min_t, _ = integrator.classify(
        seeker, x1.to(device), log_p0, steps=ode_steps, est_steps=1, eps=1e-4
    )

    preds = anchors_to_class(closest_anchor(min_t, anchor_times), time_class_map)  # type: ignore
    print(sum(preds == 0))


if __name__ == "__main__":
    main()
