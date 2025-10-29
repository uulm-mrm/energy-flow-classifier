import math
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
from torch.distributions import Independent, Normal

from flow_matching.flow_matching import (
    MultiPath,
    ODEProcess,
    MidpointIntegrator,
    NaiveMidpoints,
)
from flow_matching.flow_matching.scheduler import CosineMultiScheduler
from flow_matching.modules.utils import EMA

from models.unet import UNet
from utils import closest_anchor, anchors_to_class

from data.mnist import get_mnist, MNISTSampler


def main():
    torch.manual_seed(42)

    # consts
    device = "cuda:0"
    mnist_shape = (1, 32, 32)

    classes = (0, 1)
    anchor_times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32, device=device)
    path_width = 0.5

    # might mess up classification
    time_class_map = dict(zip(anchor_times.cpu().numpy(), (-1, *classes)))

    batch_size = 1024

    t_dims = 256
    lr = 1e-3
    epochs = 1000

    mnist = get_mnist("train")
    x_sampler = MNISTSampler(
        mnist, classes=classes, batch_size=batch_size, device=device, skip_last=True
    )

    # model stuff
    net = UNet(in_c=1, out_c=1, features=[32, 64, 128], t_dims=t_dims).to(device)
    ema = EMA(net, rate=0.999)
    path = MultiPath(CosineMultiScheduler(k=path_width))

    optim = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=epochs // 10, gamma=0.5)

    # training
    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x in x_sampler:
            optim.zero_grad()

            x0 = torch.randn_like(x[0])
            t = torch.rand((x0.shape[0],), dtype=torch.float32, device=device)

            path_sample = path.sample(torch.stack([x0, *x], dim=0), anchor_times, t)

            dxt_hat = net.forward(path_sample.xt, t)

            loss = (dxt_hat - path_sample.dxt).square().mean()

            loss.backward()
            optim.step()

            ema.update_ema_t()

            epoch_loss = epoch_loss + loss

        sched.step()
        pbar.set_description(f"Loss: {(epoch_loss / x_sampler.batches):.3f}")

    ema.to_model()
    net = net.eval()

    # probability stuff
    integrator = ODEProcess(net, MidpointIntegrator())
    seeker = NaiveMidpoints(max_evals=30, iters=3)
    ode_steps = 100
    log_p0 = log_p0 = Independent(
        Normal(
            torch.zeros(mnist_shape, device=device),
            torch.ones(mnist_shape, device=device),
        ),
        reinterpreted_batch_ndims=3,
    ).log_prob

    # plot path
    t_traj, x_traj = integrator.sample(
        x_init=torch.zeros((1, *mnist_shape), dtype=torch.float32, device=device),
        ints=torch.tensor(
            [[anchor_times[0].item(), anchor_times[-1].item()]],
            dtype=torch.float32,
            device=device,
        ),
        steps=ode_steps,
    )

    # plot path
    sols = x_traj.detach().cpu().numpy()

    ax_cols = math.ceil(sols.shape[0] ** 0.5)
    ax_rows = math.ceil(sols.shape[0] / ax_cols)
    fig, axs = plt.subplots(ax_rows, ax_cols, figsize=(ax_cols * 4, ax_rows * 4))

    # yes you can flatten axes they are a np.array
    axs = axs.flatten()  # type: ignore
    for i, (time, sol) in enumerate(zip(t_traj, sols)):
        sol = sol.reshape(*mnist_shape[1:])
        time = time.reshape(1).item()

        axs[i].imshow(sol, cmap="gray")
        axs[i].set_title(f"Time: {time:.3f}")
        axs[i].set_aspect("equal")

    for i in range(sols.shape[0], len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

    # plot prob
    ode_steps = 20  # lower steps here cuz we just estimating stuff
    for c in classes:
        indices = x_sampler.indices[classes[c]][:50]
        batch = len(indices)
        t_steps = 20

        t = torch.linspace(0.0, anchor_times[-1], steps=t_steps, device=device).repeat(
            batch
        )
        intervals = torch.zeros(
            (t_steps * batch, 2), dtype=torch.float32, device=device
        )
        intervals[:, 0] = t

        x = x_sampler.data[indices].repeat_interleave(t_steps, dim=0).to(device)
        _, probs = integrator.compute_likelihood(
            x, intervals, log_p0, steps=ode_steps, est_steps=10
        )

        plt.gca().invert_xaxis()
        for prob, interval in zip(probs.chunk(batch), intervals.chunk(batch)):
            plt.plot(interval[:, 0].cpu().numpy(), prob.cpu().numpy())
        plt.show()

    return

    # classify digits
    indices = torch.cat([x_sampler.indices[c][:5] for c in classes], dim=0)
    x = x_sampler.data[indices]
    print(x_sampler.labels[indices])

    min_t, _ = integrator.classify(
        seeker,
        x.to(device),
        log_p0,
        interval=(anchor_times[0].item(), anchor_times[-1].item()),
        steps=ode_steps,
        est_steps=1,
        eps=1e-8,
    )
    print(min_t)

    preds = anchors_to_class(closest_anchor(min_t, anchor_times), time_class_map)  # type: ignore
    print(preds)


if __name__ == "__main__":
    main()
