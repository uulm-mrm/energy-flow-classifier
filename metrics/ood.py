import os
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

import torch
from torch import Tensor

from tqdm import tqdm

from nn.run import Run
from metrics.distros import DataDistributions


parser = ArgumentParser()
parser.add_argument("--run", type=str, default="debug_run")
parser.add_argument("--id_eval", type=str, default="val")
parser.add_argument("--ood_eval", type=str, default="coco")
parser.add_argument("--id_labels", type=int, nargs="+", default=[])
parser.add_argument("--ood_labels", type=int, nargs="+", default=[])
parser.add_argument(
    "--not_labels", type=bool, action=BooleanOptionalAction, default=False
)


def compute_eval_scores(
    dd: DataDistributions,
    eval_dir: str,
    on_cats: Tensor,
    device: torch.device | str,
    is_ood: bool,
) -> Tensor:
    nlls = []

    for pt in tqdm(
        os.listdir(eval_dir),
        desc=f"Running through {"ID" if not is_ood else "OOD"} .pt files",
    ):
        if not pt.endswith(".pt"):
            continue

        batch_data: dict[str, Tensor] = torch.load(os.path.join(eval_dir, pt))
        y = batch_data["data"][:, 0].to(device).view(-1)
        dist = batch_data["data"][:, 2:].to(device)

        # extract only labels that you're interested in
        mask = torch.isin(y, on_cats)
        y = y[mask]
        dist = dist[mask]

        preds = dist.argmin(dim=-1)
        nlls.append(dd.get_dist_nll(dist, preds))

    return torch.cat(nlls, dim=0)


def get_labels_and_scores(
    dd: DataDistributions,
    id_eval: str,
    ood_eval: str,
    id_labels: Tensor,
    ood_labels: Tensor,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    id_score = compute_eval_scores(
        dd, id_eval, on_cats=id_labels, device=device, is_ood=False
    )

    ood_score = compute_eval_scores(
        dd, ood_eval, on_cats=ood_labels, device=device, is_ood=True
    )

    y_true = torch.zeros((id_score.shape[0] + ood_score.shape[0]), dtype=torch.int32)
    y_true[: id_score.shape[0]] = 1
    y_true = y_true.cpu().numpy()

    # negate negative LL so that higher is better
    y_score = -1.0 * torch.cat([id_score, ood_score], dim=0).cpu().numpy()

    return y_true, y_score


def compute_fpr95(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx]


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return roc_auc_score(y_true, y_score)  # type: ignore


def compute_auprs(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return average_precision_score(y_true, y_score)  # type: ignore


def compute_aupre(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return average_precision_score(1 - y_true, -1.0 * y_score)  # type: ignore


def main():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # load run and data distributions
    args = parser.parse_args()
    run = Run.init_from_name(args.run)
    dd = DataDistributions(run=args.run, device=device)

    # get eval dirs
    id_dir = os.path.join(run.eval_dir, args.id_eval)
    ood_dir = os.path.join(run.eval_dir, args.ood_eval)

    # get labels which to analyze
    labels = set(range(100))  # large number to cover all possible classes
    if args.not_labels:
        id_labels = labels - set(args.id_labels)
        ood_labels = labels - set(args.ood_labels)
    else:
        id_labels = labels & set(args.id_labels)
        ood_labels = labels & set(args.ood_labels)

    id_labels = torch.tensor(list(id_labels), device=device)
    ood_labels = torch.tensor(list(ood_labels), device=device)

    y_true, y_score = get_labels_and_scores(
        dd, id_dir, ood_dir, id_labels, ood_labels, device
    )

    print(f"FPR@95: {compute_fpr95(y_true, y_score):.4}")
    print(f"AUROC: {compute_auroc(y_true, y_score):.4}")
    print(f"AUPR-S: {compute_auprs(y_true, y_score):.4}")
    print(f"AUPR-E: {compute_aupre(y_true, y_score):.4}")


if __name__ == "__main__":
    main()
