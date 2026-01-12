from typing import Optional
import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.image_list import ImageList

from nuimg.data.model import RoIs


class RoIAlignExtractor(nn.Module):
    def __init__(
        self,
        labels: list[int],
        score_thresh=0.5,
        device: Optional[torch.device] = None,
        box_size=7,
    ):
        """Extracts RoI Align features from the FasterRCNN net with a ResNet50 FPN backbone
        trained on COCO

        Args:
            labels (list[int]): which labels to look for.
            score_thresh (float, optional): min threshold for a prediction. Defaults to 0.5.
            device (Optional[torch.device], optional): device which to use. Defaults to None.
            box_size (int, optional): size of boxes from RoI align. Defaults to 7.
        """
        super().__init__()

        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.eval()

        self.score_thresh = score_thresh
        self.box_size = box_size

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        self.preproc = weights.transforms()

        self.labels = torch.tensor(labels).to(self.device)

    @torch.no_grad()
    def forward(self, img: Tensor) -> Optional[RoIs]:
        """
        Args:
            img (Tensor): uint8 tensor of shape [3, H, W]

        Returns:
            Optional[RoIs]: RoIs object or None if no rois found
        """

        # preproc
        processed = self.preproc(img).to(self.device)

        image_list = ImageList(
            processed.unsqueeze(0), [(processed.shape[1], processed.shape[2])]
        )

        # bbone + fpn parts
        features = self.model.backbone(processed.unsqueeze(0))

        # rpn part
        proposals, _ = self.model.rpn(image_list, features)

        # proposals is a list of tensors [N_proposals, 4] for each img input
        # but we only have 1
        proposals = proposals[0]

        # detector head part
        detections, _ = self.model.roi_heads(
            features, [proposals], image_list.image_sizes
        )

        # same as with proposals, we get a dict of detections per image
        # and we only have an image
        det = detections[0]
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        # filter by score threshold
        keep_labels = torch.isin(labels, self.labels, assume_unique=False, invert=False)
        keep_scores = scores >= self.score_thresh
        keep = keep_labels & keep_scores

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # align boxes
        if len(boxes) == 0:
            return None

        roi_features = self.model.roi_heads.box_roi_pool(
            features, [boxes], image_list.image_sizes
        )  # type: ignore

        return RoIs(
            features=roi_features.cpu(),
            boxes=boxes.cpu(),
            scores=scores.cpu(),
            labels=labels.cpu(),
        )
