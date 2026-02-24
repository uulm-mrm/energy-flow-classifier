# Default Score -> highest softmax score
# MSP -> this is exactly the same as default score
# ODIN -> Softmax with a huge temp (i.e. 1000)
# MaxLogit -> Default score except without softmax
# Energy -> LogSumExp of all logits

import os
from argparse import ArgumentParser, BooleanOptionalAction

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

import torch
from torch import Tensor
import torch.nn.functional as F

from tqdm import tqdm

from faster_rcnn.run import Run


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
    eval_dir: str, on_cats: Tensor, device: torch.device | str, is_ood: bool
) -> dict[str, Tensor]:
    scores: dict[str, list[Tensor]] = {
        "msp": [],
        "odin": [],
        "max_logit": [],
        "energy": [],
    }

    for pt in tqdm(
        os.listdir(eval_dir),
        desc=f"Running through {"ID" if not is_ood else "OOD"} .pt files",
    ):
        if not pt.endswith(".pt"):
            continue

        # extract data
        batch_data: dict[str, Tensor] = torch.load(os.path.join(eval_dir, pt))
        labels = batch_data["data"][:, 0].to(device).view(-1)
        logits = batch_data["data"][:, 1:].to(device)

        # take only classes you want
        mask = torch.isin(labels, on_cats)
        labels = labels[mask]
        logits = logits[mask]

        # msp = max softmax prob
        msp_probs = F.softmax(logits, dim=1)
        msp_scores = torch.max(msp_probs, dim=1).values
        scores["msp"].append(msp_scores)

        # odin = temperature scaled softmax
        temperature = 1000.0
        odin_probs = F.softmax(logits / temperature, dim=1)
        odin_scores = torch.max(odin_probs, dim=1).values
        scores["odin"].append(odin_scores)

        # max logit = what the name says
        max_logit_scores = torch.max(logits, dim=1).values
        scores["max_logit"].append(max_logit_scores)

        # energy = log sum exp logits
        energy_scores = torch.logsumexp(logits, dim=1)
        scores["energy"].append(energy_scores)

    return {
        score_name: torch.cat(score_val, dim=0)
        for score_name, score_val in scores.items()
    }


def get_labels_and_scores(
    id_eval: str,
    ood_eval: str,
    id_labels: Tensor,
    ood_labels: Tensor,
    device: torch.device | str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    id_scores = compute_eval_scores(id_eval, id_labels, device, is_ood=False)
    ood_scores = compute_eval_scores(ood_eval, ood_labels, device, is_ood=True)

    y_true = torch.zeros(
        (id_scores["msp"].shape[0] + ood_scores["msp"].shape[0]), dtype=torch.int32
    )
    y_true[: id_scores["msp"].shape[0]] = 1
    y_true = y_true.cpu().numpy()

    scores: dict[str, np.ndarray] = {}
    for score in id_scores:
        scores[score] = (
            torch.cat([id_scores[score], ood_scores[score]], dim=0).cpu().numpy()
        )

    return y_true, scores


def compute_fpr95(
    y_true: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, float]:
    retval = {}
    for score in scores:
        fpr, tpr, _ = roc_curve(y_true, scores[score])
        idx = np.argmax(tpr >= 0.95)
        retval[score] = fpr[idx]

    return retval


def compute_auroc(
    y_true: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, float]:
    retval = {}
    for score in scores:
        retval[score] = roc_auc_score(y_true, scores[score])

    return retval


def compute_auprs(
    y_true: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, float]:
    retval = {}
    for score in scores:
        retval[score] = average_precision_score(y_true, scores[score])

    return retval


def compute_aupre(
    y_true: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, float]:
    y_true = 1 - y_true

    # msp and odin are [0, 1] so the reverse is 1 - score
    # logit and energy are unbound so the reverse is -1 * score
    return {
        "msp": average_precision_score(y_true, 1.0 - scores["msp"]),
        "odin": average_precision_score(y_true, 1.0 - scores["odin"]),
        "max_logit": average_precision_score(y_true, -1.0 * scores["max_logit"]),
        "energy": average_precision_score(y_true, -1.0 * scores["energy"]),
    }  # type: ignore


def main():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # load run and data distributions
    args = parser.parse_args()
    run = Run.init_from_name(args.run)

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

    labels, scores = get_labels_and_scores(
        id_dir, ood_dir, id_labels, ood_labels, device=device
    )

    print("FPR@95: ", compute_fpr95(labels, scores))
    print("AUROC: ", compute_auroc(labels, scores))
    print("AUPR-S: ", compute_auprs(labels, scores))
    print("AUPR-E: ", compute_aupre(labels, scores))


if __name__ == "__main__":
    main()
