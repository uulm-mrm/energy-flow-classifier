import os
from argparse import ArgumentParser
from typing import Callable

import torch

from nn.run import Run

import experiments

parser = ArgumentParser()
parser.add_argument("--run", type=str, default="debug_run")
parser.add_argument("--eval", type=str)
parser.add_argument("--metrics", type=str, nargs="+", default=["accuracy"])

METRICS: dict[str, Callable[[Run, str, torch.device | str], float]] = {
    "accuracy": experiments.accuracy.metric
}


def main():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # load run config
    args = parser.parse_args()
    run = Run.init_from_name(args.run)

    # get eval dir
    eval_dir = os.path.join(run.eval_dir, args.eval)

    # set up metric runs
    retvals = {}
    for metric in args.metrics:
        func = METRICS[metric]
        retvals[metric] = func(run, eval_dir, device)

    print(retvals)


if __name__ == "__main__":
    main()
