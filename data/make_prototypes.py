import os
from tqdm import tqdm

import torch

from data.config import ExportConfig


def compute_prototypes(cfg: ExportConfig):
    # init dict for running sum to end up with mean
    prototype_dict = {
        cat: {"counts": 0, "running_sum": torch.zeros((256, 7, 7))}
        for cat in cfg.label_mapping
    }

    for fname in tqdm(os.listdir(cfg.output_dir)):
        if not fname.endswith(".pt"):
            continue

        frame = torch.load(os.path.join(cfg.output_dir, fname))
        features, labels = frame["features"], frame["labels"]

        for cat in prototype_dict:
            mask = labels == cat

            prototype_dict[cat]["counts"] += labels[mask].shape[0]
            prototype_dict[cat]["running_sum"] += features[mask].sum(dim=0)

    prototypes = [
        prototype_dict[cat]["running_sum"] / prototype_dict[cat]["counts"]
        for cat in prototype_dict
    ]
    prototypes = torch.stack(prototypes, dim=0)

    torch.save(prototypes, cfg.prototypes)
