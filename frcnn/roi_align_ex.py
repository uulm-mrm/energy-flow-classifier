import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.image_list import ImageList


class RoIAlignExtractor(nn.Module):
    def __init__(self, score_thresh=0.5, device=None, box_size=7):
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

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            img (Tensor): uint8 tensor of shape [3, H, W]

        Returns:
            retval[0]: [rois, 256, 7, 7]
            retval[1]: [rois, 4]
            retval[2]: [rois]
            retval[3]: [rois]
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
        with torch.no_grad():
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
        keep = scores >= self.score_thresh
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # align boxes
        if len(boxes) == 0:
            return (
                torch.empty((0, 256, self.box_size, self.box_size)),
                boxes,
                scores,
                labels,
            )

        roi_features = self.model.roi_heads.box_roi_pool(
            features, [boxes], image_list.image_sizes
        )  # type: ignore

        return roi_features, boxes, scores, labels
