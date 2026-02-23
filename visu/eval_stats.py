import os
from collections import defaultdict
from argparse import ArgumentParser

import torch
from torch import Tensor

import matplotlib.pyplot as plt
import seaborn as sns

from nn.run import Run


parser = ArgumentParser()
parser.add_argument("--run", type=str, default="debug_run")
parser.add_argument("--eval", type=str)


def main():
    # get run
    args = parser.parse_args()
    run = Run.init_from_name(args.run)
    eval_dir = os.path.join(run.eval_dir, args.eval)

    seen_labels = set()
    stats = defaultdict(lambda: {"distances": [], "potentials": []})

    for pt in os.listdir(eval_dir):
        if not pt.endswith(".pt"):
            continue

        # extract all data
        data: Tensor = torch.load(os.path.join(eval_dir, pt))["data"]
        labels = data[:, 0].int().cpu().view(-1)
        potential = data[:, 1].cpu().view(-1)
        dists = data[:, 2:].cpu()

        # append to seen labels
        seen_labels.update(labels.tolist())

        # add to each dict in stats
        for label in torch.unique(labels):
            mask = labels == label
            label = label.item()

            # mask-out data
            p = potential[mask]
            d = dists[mask, label]
            # d = dists[mask].min(dim=-1).values

            stats[label]["potentials"].append(p)
            stats[label]["distances"].append(d)

    for cat in stats:
        for key, val in stats[cat].items():
            stats[cat][key] = torch.cat(val, dim=0)  # type: ignore

    _, axes = plt.subplots(len(stats), 2, figsize=(10, 10))
    for i, cat in enumerate(stats):
        distances = stats[cat]["distances"]
        potentials = stats[cat]["potentials"]

        sns.histplot(distances, bins=100, kde=True, ax=axes[i][0], color="skyblue")
        axes[i][0].set_title(f"Category {cat} - Distances")

        sns.histplot(potentials, bins=100, kde=True, ax=axes[i][1], color="salmon")
        axes[i][1].set_title(f"Category {cat} - Potentials")

    plt.tight_layout()
    plt.savefig(os.path.join("visu", "plots", "stats.pdf"), format="pdf")


if __name__ == "__main__":
    main()
