import os
from argparse import ArgumentParser, BooleanOptionalAction

import torch
from torch import Tensor

from tqdm import tqdm

from faster_rcnn.run import Run


parser = ArgumentParser()
parser.add_argument("--run", type=str, default="frcnn_debug_run")
parser.add_argument("--eval", type=str)
parser.add_argument("--labels", type=int, nargs="+", default=[])
parser.add_argument(
    "--not_labels", type=bool, action=BooleanOptionalAction, default=False
)


def metric(eval_dir: str, on_cats: Tensor, device: torch.device | str) -> float:
    true_positives = 0
    count = 0

    for pt in (pbar := tqdm(os.listdir(eval_dir))):
        if not pt.endswith(".pt"):
            continue

        batch_data: dict[str, Tensor] = torch.load(os.path.join(eval_dir, pt))
        labels = batch_data["data"][:, 0].to(device).view(-1)
        logits = batch_data["data"][:, 1:].to(device)

        # extract only labels that you're interested in
        mask = torch.isin(labels, on_cats)
        labels = labels[mask]
        logits = logits[mask]

        sm = logits.softmax(dim=-1)
        preds = sm.argmax(dim=-1)
        preds[sm.max(dim=-1).values <= 0.5] = -1

        true_positives += (labels == preds).sum().item()
        count += labels.numel()

        pbar.set_description(f"Running Accuracy: {(true_positives / count):.4f}")

    acc = true_positives / count

    print(f"Total Accuracy: {acc:.4f}")
    return acc


def main():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # load run and data distributions
    args = parser.parse_args()
    run = Run.init_from_name(args.run)

    # get eval dir
    eval_dir = os.path.join(run.eval_dir, args.eval)

    # get labels which to analyze
    labels = set(range(100))  # large number to cover all possible classes
    if args.not_labels:
        labels = labels - set(args.labels)
    else:
        labels = labels & set(args.labels)
    labels = torch.tensor(list(labels), device=device)

    # run metric
    _ = metric(eval_dir, labels, device)


if __name__ == "__main__":
    main()
