# pylint: disable=W0201

import os
import json
from bisect import bisect_right
from collections import defaultdict
from math import ceil

import torch
from torch import Tensor

from data.config import ExportConfig


class RoIFeatureDataset:
    def __init__(self, export_cfg: ExportConfig) -> None:
        super().__init__()

        self.dataset_dir = export_cfg.output_dir
        self.prototypes_pt = export_cfg.prototypes

        # get .pt files-index LuT
        with open(
            os.path.join(self.dataset_dir, "index_lookup_table.json"),
            "r+",
            encoding="utf-8",
        ) as f:
            lut = json.loads(f.read())

        self.fnames = lut["fnames"]
        self.offsets = lut["offsets"]

        # because keys should be ints, not strings
        self.cat_dict = export_cfg.label_mapping

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, idx: int) -> tuple[str, int]:
        total = len(self)

        # handle negative indexing
        if idx < 0:
            idx += total

        # enforce bounds
        if idx < 0 or idx >= total:
            raise IndexError(idx)

        file_idx = bisect_right(self.offsets, idx)
        lower = 0 if file_idx == 0 else self.offsets[file_idx - 1]

        return self.fnames[file_idx], idx - lower


class RoIFeatureDataLoader:
    def __init__(
        self,
        dataset: RoIFeatureDataset,
        batch_size: int,
        shuffle: bool,
        skip_last: bool,
        device: torch.device | None = None,
    ) -> None:

        self.dataset = dataset

        self.batch_size = min(batch_size, len(dataset))

        self.shuffle = shuffle
        self.skip_last = skip_last

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.prototypes = torch.load(self.dataset.prototypes_pt)

    def get_from_file(self, fname: str, idxs: list[int]) -> tuple[Tensor, Tensor]:
        """Takes a list of indices in fname and returns features and labels associated to them"""

        frame: dict[str, Tensor] = torch.load(
            os.path.join(self.dataset.dataset_dir, fname)
        )

        return frame["features"][idxs], frame["labels"][idxs]

    def __iter__(self):
        # __iter__ is what's done at startup of iteration

        if self.shuffle:
            self.indices = torch.randperm(len(self.dataset)).tolist()
        else:
            self.indices = list(range(len(self.dataset)))

        if self.skip_last:
            self.num_batches = len(self.indices) // self.batch_size
        else:
            self.num_batches = ceil(len(self.indices) / self.batch_size)

        self.current_batch = 0

        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        # __next__ is what's done at each iteration point

        # check exit
        if self.current_batch >= self.num_batches:
            raise StopIteration

        # take out batch_size indices
        start = self.current_batch * self.batch_size
        end = start + self.batch_size

        batch_indices = self.indices[start:end]

        # update batch ctr
        self.current_batch += 1

        # group by fname
        fname_map: dict[str, list[int]] = defaultdict(list)

        for idx in batch_indices:
            fname, local_idx = self.dataset[idx]
            fname_map[fname].append(local_idx)

        # load data
        features: list[Tensor] = []
        labels: list[Tensor] = []

        for fname, idxs in fname_map.items():
            feats, labs = self.get_from_file(fname, idxs)

            features.append(feats)
            labels.append(self.prototypes[labs])

        # return and push to device
        return (
            torch.cat(features, dim=0).to(self.device),
            torch.cat(labels, dim=0).to(self.device),
        )


if __name__ == "__main__":
    with open(
        r"nuimg/dataset/nuimages-v1.0_frames/v1.0-mini/config.json",
        "r+",
        encoding="utf-8",
    ) as _f:
        cfg = json.loads(_f.read())
        cfg = ExportConfig.from_dict(cfg)

    ds = RoIFeatureDataset(cfg)
    dl = RoIFeatureDataLoader(ds, batch_size=50, shuffle=True, skip_last=False)

    for x, y in dl:
        print(x.shape, y.shape)
