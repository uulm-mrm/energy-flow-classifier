import os

from tqdm import tqdm

import torch
from torch import Tensor

from nn.run import Run


def metric(run: Run, eval_dir: str, device: torch.device | str) -> float:
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

    acc = true_positives / count

    print(f"Total Accuracy: {acc:.4f}")
    return acc
