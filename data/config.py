import os
from typing import Literal


class ExportConfig:
    def __init__(self, dataset: Literal["nuimages-v1.0", "COCO"], version: str) -> None:
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")

        self.dataset = dataset
        self.version = version

        self.input_dir = os.path.join(dataset_dir, self.dataset)

        if dataset == "COCO":
            self.input_dir = os.path.join(self.input_dir, self.version)

        self.output_dir = os.path.join(
            dataset_dir, self.dataset + "_frames", self.version
        )

        self.label_mapping: dict[int, str] = {}

        self.prototypes = os.path.join(self.output_dir, "class_prototypes.pt")

    def set_labels(self, label_mapping: dict[int, str]):
        self.label_mapping = label_mapping

    @staticmethod
    def from_dict(d: dict) -> "ExportConfig":
        cfg = ExportConfig(d["dataset"], d["version"])
        cfg.set_labels({int(k): v for k, v in d["label_mapping"].items()})

        return cfg
