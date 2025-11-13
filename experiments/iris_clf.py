from tqdm import tqdm

import torch
from torch import optim

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

from data.iris import get_iris

from models.mlp import VectorField, MLP


def main():
    torch.manual_seed(42)
    device = "cuda:0"

    num_class = 3
    in_dims = 4
    emb_f = [8, 16, 32]
    h_dims = 512
    epochs = 5_000
    batch_size = 50

    sigma = 1.0
    r = 5.0

    # noise sampler
    multi_normal = MultiIndependentNormal(
        c=num_class, shape=(emb_f[-1],), r=r, sigma=sigma, device=device
    )

    # dataset
    x1, x2, x3 = get_iris(device=device)
    x = torch.cat([x1, x2, x3], dim=0)

    # models
    emb = MLP(in_features=in_dims, emb_features=emb_f).to(device)
    vf = VectorField(emb_f[-1], h_dims, t_d=1).to(device)
    ema = EMA(vf, rate=0.999)

    path = AffineMultiPath(AffinePath(CosineScheduler()), num_paths=num_class)

    # optimizers and schedulers
    emb_lr = 1e-1
    vf_lr = 1e-2
    target_lr = 1e-3

    emb_gamma = (target_lr / emb_lr) ** (1.0 / epochs)
    vf_gamma = (target_lr / vf_lr) ** (1.0 / epochs)

    emb_optim = optim.AdamW(emb.parameters(), lr=emb_lr)
    emb_sched = optim.lr_scheduler.ExponentialLR(emb_optim, gamma=emb_gamma)

    vf_optim = optim.AdamW(vf.parameters(), lr=vf_lr)
    vf_sched = optim.lr_scheduler.ExponentialLR(vf_optim, gamma=vf_gamma)

    for _ in (pbar := tqdm(range(epochs))):
        emb_optim.zero_grad()
        vf_optim.zero_grad()

        x_noise = multi_normal.sample(batch_size)
        t = torch.rand((batch_size,), dtype=torch.float32, device=device)

        # project
        emb_x = emb.forward(x)
        emb_x = emb_x.unflatten(0, (num_class, batch_size))

        path_sample = path.sample(emb_x, x_noise, t)
        dxt_hat = vf.forward(path_sample.xt, path_sample.t)

        loss = (dxt_hat - path_sample.dxt).square().mean()

        loss.backward()
        vf_optim.step()
        emb_optim.step()

        ema.update_ema_t()

        emb_sched.step()
        vf_sched.step()

        pbar.set_description(f"Loss: {loss.item():.3f}")

    # eval all models
    emb = emb.eval()

    ema.to_model()
    vf = vf.eval()

    # evaluate accuracy
    proc = ODEProcess(vf, RungeKuttaIntegrator(RK4_TABLEAU, device=device))

    x_init = torch.cat([x, torch.rand_like(x1)], dim=0)
    intervals = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device)
    intervals = intervals.expand(x_init.shape[0], 2)

    # first embed
    x_init = emb.forward(x_init)

    # then integrate
    _, x_traj = proc.sample(x_init, intervals, steps=100)
    sols = x_traj[-1]
    probs = multi_normal.log_likelihood(sols)
    print(probs)

    for i, c in enumerate(probs.argmax(dim=1).chunk(4)[:-1]):
        print((c == i).sum())


if __name__ == "__main__":
    main()
