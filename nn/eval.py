import os
import shutil
from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch import Tensor

from fm.flow_matching import PotentialProcess, RungeKuttaIntegrator, tableaus

from data import RoIFeatureDataset, RoIFeatureDataLoader

from nn.models import TimeCNN
from nn.configs import EvalConfig, load_yaml

parser = ArgumentParser()
parser.add_argument(
    "--eval_config",
    type=str,
    default=os.path.join("nn", "configs", "eval.cfg.yaml"),
)
parser.add_argument("--name", type=str, required=False)


def evaluate():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # get config
    args = parser.parse_args()
    cfg = EvalConfig(**load_yaml(args.eval_config)["eval_config"])

    # make cfg dir
    eval_dir = cfg.get_eval_dir(args.name)
    os.makedirs(eval_dir, exist_ok=True)

    # copy config
    shutil.copy(args.eval_config, os.path.join(eval_dir, "eval.cfg.yaml"))

    # dataset
    ds = RoIFeatureDataset(cfg.get_export_cfg())

    # dataloader
    dl = RoIFeatureDataLoader(
        ds, cfg.batch_size, shuffle=False, skip_last=False, device=device, train=False
    )

    # prototypes from run data config
    prototypes: Tensor = torch.load(cfg.get_prototypes())
    prototypes = prototypes.to(device)
    prototypes_flat = prototypes.view(prototypes.shape[0], -1)

    # model
    net = TimeCNN(cfg.get_model_cfg()["model"]).to(device)
    net.load_state_dict(torch.load(cfg.get_model_weights()))
    net.eval()

    # process setup
    proc = PotentialProcess(
        net, RungeKuttaIntegrator(tableaus.RK4_TABLEAU, device=device)  # type: ignore
    )

    # go over data
    for batch, (x, _, y) in tqdm(enumerate(dl), desc="Processing Batches"):
        state = {}

        x: Tensor = x.to(device)
        y: Tensor = y.to(device)
        y = y.view(-1, 1)

        # compute potential
        t = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        potential = net.forward(x, t)

        # solve process to get distances from prototypes
        intervals = torch.tensor([[0.0, 1.0]], dtype=x.dtype, device=x.device)
        intervals = intervals.expand(x.shape[0], 2)

        _, x_traj = proc.sample(x, intervals, steps=cfg.ode_steps)
        sols = x_traj[-1]
        sols_flat = sols.view(sols.shape[0], -1)
        dist_measure = torch.cdist(sols_flat, prototypes_flat)

        # save state as pt
        state["meta"] = ["labels", "potential", "distance"]
        state["sizes"] = [y.shape[1], potential.shape[1], dist_measure.shape[1]]
        state["data"] = torch.cat([y, potential, dist_measure], dim=1)

        fname = os.path.join(eval_dir, f"batch_{batch}.pt")
        torch.save(state, fname)


if __name__ == "__main__":
    evaluate()
