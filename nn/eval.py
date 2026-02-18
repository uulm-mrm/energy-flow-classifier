import os
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


def evaluate():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # get config
    args = parser.parse_args()
    cfg = EvalConfig(**load_yaml(args.eval_config)["eval_config"])

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
    for x, _, y in tqdm(dl, desc="Processing Batches"):
        x: Tensor = x.to(device)
        y: Tensor = y.to(device)
        print(y)

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
        print(dist_measure)


if __name__ == "__main__":
    evaluate()
