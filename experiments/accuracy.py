import os
from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch import Tensor

from nn.run import Run

parser = ArgumentParser()
parser.add_argument("--run", type=str, default="debug_run")
parser.add_argument("--eval", type=str)


def metric():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # load run config
    args = parser.parse_args()
    run = Run.init_from_name(args.run)

    # get eval dir
    eval_dir = os.path.join(run.eval_dir, args.eval)

    true_positives = 0
    count = 0

    for pt in (pbar := tqdm(os.listdir(eval_dir))):
        if not pt.endswith(".pt"):
            continue

        batch_data: dict[str, Tensor] = torch.load(os.path.join(eval_dir, pt))
        dist = batch_data["data"][:, 2:].to(device)
        labels = batch_data["data"][:, 0].to(device).view(-1)

        preds = dist.argmin(dim=-1)

        true_positives += (labels == preds).sum().item()
        count += labels.numel()

        pbar.set_description(f"Running Accuracy: {(true_positives / count):.4f}")

    print(f"Total Accuracy: {(true_positives / count):.4f}")


if __name__ == "__main__":
    metric()
