from typing import Optional
import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


class FasterRCNNPredictor(nn.Module):
    def __init__(self, num_cls: int, device: Optional[torch.device] = None) -> None:
        super().__init__()

        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        faster_rcnn = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # get the part of the head that makes vectors
        self.box_head: nn.Module = faster_rcnn.roi_heads.box_head  # type: ignore

        # actual part that learns stuff
        self.classifier = nn.Linear(1024, num_cls)

        # push the whole net to device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    @torch.no_grad()
    def run_box_head(self, x: Tensor) -> Tensor:
        return self.box_head(x)

    def forward(self, x: Tensor) -> Tensor:
        roi_features = self.run_box_head(x)
        return self.classifier(roi_features)


if __name__ == "__main__":
    frcnn = FasterRCNNPredictor(num_cls=3)
    roi_align = torch.rand((10, 256, 7, 7), device="cuda")
    print(frcnn(roi_align))
