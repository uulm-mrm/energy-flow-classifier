import os

from tqdm import tqdm

from argparse import ArgumentParser

from typing import Optional
import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


from data.config import ExportConfig
from data.data_model import LabeledFrame

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["COCO", "nuimages-v1.0"])
parser.add_argument(
    "--version",
    type=str,
    default="v1.0-mini",
    choices=["v1.0-mini", "v1.0-train", "v1.0-val", "val2017"],
)


class FasterRCNNBoxHead(nn.Module):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__()

        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # get the part of the head that makes vectors
        self.box_head: nn.Module = faster_rcnn.roi_heads.box_head  # type: ignore

        # push the whole net to device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return self.box_head(x)


def main():
    device = torch.device("cuda")
    args = parser.parse_args()
    cfg = ExportConfig(dataset=args.dataset, version=args.version)

    box_head = FasterRCNNBoxHead(device=device)

    for pt in tqdm(os.listdir(cfg.output_dir), desc="Remaking Frames"):
        if not pt.endswith(".pt") or "prototypes" in pt or "box_head" in pt:
            continue

        frame: LabeledFrame = LabeledFrame(
            **torch.load(os.path.join(cfg.output_dir, pt))
        )

        features = box_head.forward(frame.features.to(device)).cpu()
        new_frame = LabeledFrame(features=features, labels=frame.labels)

        new_pt = os.path.join(cfg.output_dir, "box_head_" + pt)
        torch.save(new_frame.__dict__, new_pt)


if __name__ == "__main__":
    main()
