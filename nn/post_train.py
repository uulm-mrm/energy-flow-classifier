import os
from collections import defaultdict
from argparse import ArgumentParser

from tqdm import tqdm

from scipy import stats

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
    batch_stats = defaultdict(lambda: {"dist": [], "pot": []})
    for x, _, y in tqdm(dl, desc="Processing Batches"):
        x: Tensor = x.to(device)
        y: Tensor = y.to(device)

        # get the categories present in batch
        cats = torch.unique(y)

        # compute potential
        t = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        with torch.no_grad():
            potential = net.forward(x, t).detach()

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

            batch_stats[cat]["dist"].append(dist_measure[mask, cat].detach().cpu())
            batch_stats[cat]["pot"].append(potential[mask].detach().cpu())

    # make post_train dict
    post_train_res = {
        cat: {
            "dist_shape": torch.empty(size=(0,)),
            "dist_loc": torch.empty(size=(0,)),
            "dist_scale": torch.empty(size=(0,)),
            "pot_loc": torch.empty(size=(0,)),
            "pot_sd": torch.empty(size=(0,)),
        }
        for cat in batch_stats
    }
    for cat, data in batch_stats.items():
        # make tensors
        dist = torch.cat(data["dist"], dim=0).numpy()
        pot = torch.cat(data["pot"], dim=0)

        # get distance gamma estimation
        shape, loc, scale = stats.gamma.fit(dist)

        post_train_res[cat]["dist_shape"] = torch.tensor(shape)
        post_train_res[cat]["dist_loc"] = torch.tensor(loc)
        post_train_res[cat]["dist_scale"] = torch.tensor(scale)

        # estimate mean and sd for potential
        loc, dev = torch.std_mean(pot)
        post_train_res[cat]["pot_loc"] = loc
        post_train_res[cat]["pot_sd"] = dev

    # save res to pt
    torch.save(post_train_res, os.path.join(run.run_dir, "post_train.pt"))


if __name__ == "__main__":
    post_train()
