import os
import json
from bisect import bisect_right


class RoIFeatureDataset:
    def __init__(self, dataset_dir: str) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir

        # get .pt files-index LuT
        with open(
            os.path.join(dataset_dir, "index_lookup_table.json"), "r+", encoding="utf-8"
        ) as f:
            lut = json.loads(f.read())

        self.fnames = lut["fnames"]
        self.offsets = lut["offsets"]

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


def main():
    from nuimg.data.consts import FEATURES_DATASET_DIR

    ds = RoIFeatureDataset(os.path.join(FEATURES_DATASET_DIR))

    for i in [1, 17, 0, 5, 2, 82]:
        print(ds[i])


if __name__ == "__main__":
    main()
