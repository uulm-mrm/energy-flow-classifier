from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from data.mnist import MNISTSampler, get_mnist
from flow_matching.flow_matching.integrator import RungeKuttaIntegrator
from flow_matching.flow_matching.integrator_utils import RK4_TABLEAU
from flow_matching.flow_matching.path import AffinePath
from flow_matching.flow_matching.process import ODEProcess
from flow_matching.flow_matching.scheduler import CosineScheduler
from flow_matching.modules.utils.ema import EMA
from models.unet import UNet


def main():
    torch.manual_seed(42)

    device = "cuda:0"
    mnist_shape = (1, 32, 32)
    classes = (0,)

    batch_size = 1024
    t_dims = 256
    lr = 1e-3
    epochs = 1024

    mnist = get_mnist("train")
    mnist_sampler = MNISTSampler(
        mnist, classes=classes, batch_size=batch_size, device=device, skip_last=True
    )

    net = UNet(in_c=1, out_c=1, features=[32, 64, 128], t_dims=t_dims).to(device)
    ema = EMA(net, rate=0.999)

    path = AffinePath(CosineScheduler())

    optim = torch.optim.AdamW(net.parameters(), lr=lr)

    for _ in (pbar := tqdm(range(epochs))):
        epoch_loss = 0.0

        for x in mnist_sampler:
            optim.zero_grad()

            x0 = torch.randn_like(x[0])
            x1 = x[0]
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
        torch.zeros((1, *mnist_shape), dtype=torch.float32, device=device),
        ints=torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device),
        steps=ode_steps,
    )

    sols = x_traj[-1].detach().cpu().numpy()
    plt.imshow(sols[0][0], "gray")
    plt.show()


if __name__ == "__main__":
    main()
