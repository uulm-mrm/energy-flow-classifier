import os
from collections import defaultdict
from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch import Tensor

from fm.flow_matching import PotentialProcess, RungeKuttaIntegrator, tableaus

from data import RoIFeatureDataset, RoIFeatureDataLoader

from nn.models import TimeCNN
from nn.run import Run

parser = ArgumentParser()
parser.add_argument("--run", type=str, default="debug_run")
parser.add_argument("--checkpoint", type=int, default=9)


def post_train():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)

    # load run
    args = parser.parse_args()
    run = Run.init_from_name(args.run)

    # dataset
    ds = RoIFeatureDataset(run.data_cfg.get_export_cfg())

    # dataloader
    dl = RoIFeatureDataLoader(
        ds, batch_size=1024, shuffle=False, skip_last=False, device=device
    )

    # get prototypes
    assert (
        dl.prototypes is not None
    ), "Prototypes are None in training script. This is bad."
    prototypes = dl.prototypes.to(device)
    prototypes_flat = prototypes.view(prototypes.shape[0], -1)

    # model
    net = TimeCNN(run.model_config["model"]).to(device)
    net.load_state_dict(torch.load(run.checkpoint_fname.format(args.checkpoint)))
    net.eval()

    # process setup
    proc = PotentialProcess(
        net, RungeKuttaIntegrator(tableaus.RK4_TABLEAU, device=device)  # type: ignore
    )

    # go over data
    batch_stats = defaultdict(lambda: {"pot_sum": 0.0, "dist_sum": 0.0, "count": 0})
    for x, _, y in tqdm(dl, desc="Processing Batches"):
        x: Tensor = x.to(device)
        y: Tensor = y.to(device)

        # get the categories present in batch
        cats = torch.unique(y)

        # compute potential
        t = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        potential = net.forward(x, t)

        # solve process to get distances from prototypes
        intervals = torch.tensor([[0.0, 1.0]], dtype=x.dtype, device=x.device)
        intervals = intervals.expand(x.shape[0], 2)

        _, x_traj = proc.sample(x, intervals, steps=10)
        sols = x_traj[-1]
        sols_flat = sols.view(sols.shape[0], -1)
        dist_measure = torch.cdist(sols_flat, prototypes_flat)

        # update batch stats dict
        for cat in cats:
            mask = y == cat
            cat = cat.item()

            batch_stats[cat]["pot_sum"] += potential[mask].detach().sum().item()
            batch_stats[cat]["dist_sum"] += dist_measure[mask, cat].detach().sum().item()  # type: ignore
            batch_stats[cat]["count"] += mask.detach().sum().item()

    # make post_train dict
    post_train_res = {
        cat: {
            "mean_potential": data["pot_sum"] / data["count"],
            "mean_dist": data["dist_sum"] / data["count"],
        }
        for cat, data in batch_stats.items()
    }

    # save res to pt
    torch.save(post_train_res, os.path.join(run.run_dir, "post_train.pt"))


if __name__ == "__main__":
    post_train()
