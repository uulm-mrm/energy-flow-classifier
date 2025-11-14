import matplotlib.pyplot as plt

import torch

from flow_matching.flow_matching import (
    ODEProcess,
    RungeKuttaIntegrator,
)
from flow_matching.flow_matching.distributions import MultiIndependentNormal
from flow_matching.flow_matching.integrator_utils import RK4_TABLEAU

from models.unet import UNet

from data.mnist import get_mnist, get_fashion_mnist, MNISTSampler

from experiments.mnist.consts import CONFIG


def generate(noise_sampler: MultiIndependentNormal, proc: ODEProcess, ode_steps: int):
    x_noises = noise_sampler.sample(3)
    for x_noise in x_noises:
        intervals = torch.tensor(
            [[0.0, 1.0]], dtype=torch.float32, device=CONFIG.device
        ).expand(x_noise.shape[0], 2)

        _, x_traj = proc.sample(x_noise, ints=intervals, steps=ode_steps)
        sols = x_traj[-1].detach().cpu().numpy()

        for sol in sols:
            plt.imshow(sol[0], "gray")
            plt.show()

    return


def accuracy(
    data_sampler: MNISTSampler,
    noise_sampler: MultiIndependentNormal,
    proc: ODEProcess,
    ode_steps: int,
):
    argmax_cls = {c: i for i, c in enumerate(CONFIG.classes)}

    for c in CONFIG.classes:
        x_pred = data_sampler.data[data_sampler.indices[c]].to(CONFIG.device)
        intervals = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float32, device=CONFIG.device
        ).expand(x_pred.shape[0], 2)

        _, x_traj = proc.sample(x_pred, intervals, steps=ode_steps)
        sols = x_traj[-1]
        probs = noise_sampler.log_likelihood(sols)
        print(f"Class {c} elements: {x_pred.shape[0]}")
        print(
            f"Correctly classified: {torch.sum(probs.argmax(dim=-1) == argmax_cls[c]).item()}"
        )

    return


def random_from_dataset(
    data_sampler: MNISTSampler,
    noise_sampler: MultiIndependentNormal,
    proc: ODEProcess,
    ode_steps: int,
):
    x_pred = data_sampler.data[:10].to(CONFIG.device)
    intervals = torch.tensor(
        [[1.0, 0.0]], dtype=torch.float32, device=CONFIG.device
    ).expand(x_pred.shape[0], 2)

    _, x_traj = proc.sample(x_pred, intervals, steps=ode_steps)
    sols = x_traj[-1]
    probs = noise_sampler.log_likelihood(sols)

    for img, prob in zip(x_pred, probs):
        print(prob)

        plt.imshow(img[0].detach().cpu(), "gray")
        plt.show()


def random_noise(
    noise_sampler: MultiIndependentNormal, proc: ODEProcess, ode_steps: int
):
    x_pred = torch.randn(
        size=(10, *CONFIG.shape), dtype=torch.float32, device=CONFIG.device
    )
    intervals = torch.tensor(
        [[1.0, 0.0]], dtype=torch.float32, device=CONFIG.device
    ).expand(x_pred.shape[0], 2)

    _, x_traj = proc.sample(x_pred, intervals, steps=ode_steps)
    sols = x_traj[-1]
    probs = noise_sampler.log_likelihood(sols)
    print(probs)


def test():
    torch.manual_seed(42)

    # dataset
    mnist = get_mnist("test")
    mnist_sampler = MNISTSampler(
        mnist,
        classes=CONFIG.classes,
        batch_size=CONFIG.batch_size,
        device=CONFIG.device,
        skip_last=True,
    )

    fashion_mnist = get_fashion_mnist("test")
    fashion_sampler = MNISTSampler(
        fashion_mnist,
        classes=CONFIG.classes,
        batch_size=CONFIG.batch_size,
        device=CONFIG.device,
        skip_last=True,
    )

    multi_normal = MultiIndependentNormal(
        c=CONFIG.num_classes, shape=CONFIG.shape, k=CONFIG.k, device=CONFIG.device
    )
    print(multi_normal.means)
    print(multi_normal.sigma)

    net = UNet(in_c=1, out_c=1, features=CONFIG.features, t_dims=CONFIG.t_dims).to(
        CONFIG.device
    )
    net.load_state_dict(torch.load(f"trained/mnist_{CONFIG.num_classes}.pt"))
    net = net.eval()

    # process to integrate
    proc = ODEProcess(net, RungeKuttaIntegrator(RK4_TABLEAU, device=CONFIG.device))
    ode_steps = 100

    # generate(multi_normal, proc, ode_steps)
    # accuracy(mnist_sampler, multi_normal, proc, ode_steps)
    random_from_dataset(mnist_sampler, multi_normal, proc, ode_steps)
    random_from_dataset(fashion_sampler, multi_normal, proc, ode_steps)
    # random_noise(multi_normal, proc, ode_steps)


if __name__ == "__main__":
    test()
