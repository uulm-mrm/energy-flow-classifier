from typing import Optional
import torch
from torch import nn, Tensor


class FasterRCNNPredictor(nn.Module):
    def __init__(self, num_cls: int, device: Optional[torch.device] = None) -> None:
        super().__init__()
        # actual part that learns stuff
        self.classifier = nn.Linear(1024, num_cls)

        # push the whole net to device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input shaped (B, 1024)
        """
        return self.classifier(x)


if __name__ == "__main__":
    frcnn = FasterRCNNPredictor(num_cls=3)
    roi_align = torch.rand((10, 1024), device="cuda")
    print(frcnn(roi_align))
