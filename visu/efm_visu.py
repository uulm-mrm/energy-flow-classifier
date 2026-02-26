import os
from argparse import ArgumentParser
from typing import Optional
import json

import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.image_list import ImageList
from torchvision.io.image import decode_image
from torchvision.utils import draw_bounding_boxes

from fm.flow_matching import PotentialProcess, RungeKuttaIntegrator, tableaus

from data.data_model import RoIs

from metrics.distros import DataDistributions
from nn.run import Run
from nn.models.cnn import TimeCNN


parser = ArgumentParser()
parser.add_argument("--run", type=str, default="debug_run")
parser.add_argument("--checkpoint", type=int, default=20)


class EFMDetector(nn.Module):
    def __init__(self, efm: TimeCNN, device: Optional[torch.device] = None):
        super().__init__()

        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )
        self.frcnn = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.frcnn.eval()

        self.efm = efm

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        self.preproc = weights.transforms()

        self.process = PotentialProcess(
            self.efm, RungeKuttaIntegrator(tableaus.RK4_TABLEAU, device=device)  # type: ignore
        )

    @torch.no_grad()
    def forward(
        self, img: Tensor, prototypes_flat: Tensor, dd: DataDistributions
    ) -> RoIs:
        # preproc
        processed = self.preproc(img).to(self.device)

        image_list = ImageList(
            processed.unsqueeze(0), [(processed.shape[1], processed.shape[2])]
        )

        # bbone + fpn parts
        features = self.frcnn.backbone(processed.unsqueeze(0))

        # rpn part
        proposals, _ = self.frcnn.rpn(image_list, features)

        # proposals is a list of tensors [N_proposals, 4] for each img input
        # but we only have 1
        proposals = proposals[0]

        # detector head part
        detections, _ = self.frcnn.roi_heads(
            features, [proposals], image_list.image_sizes
        )

        # same as with proposals, we get a dict of detections per image
        # and we only have an image
        det = detections[0]
        boxes = det["boxes"]
        scores = det["scores"]

        # filter boxes with low scores
        keep = scores >= 0.5
        boxes = boxes[keep]

        # get roi features
        roi_features: Tensor = self.frcnn.roi_heads.box_roi_pool(
            features, [boxes], image_list.image_sizes
        )  # type: ignore

        # get the distances from prototypes
        intervals = torch.tensor(
            [[0.0, 1.0]], dtype=roi_features.dtype, device=roi_features.device
        )
        intervals = intervals.expand(roi_features.shape[0], 2)

        _, x_traj = self.process.sample(roi_features, intervals, steps=10)
        sols = x_traj[-1]
        sols_flat = sols.view(sols.shape[0], -1)

        # get prediction
        dists = torch.cdist(sols_flat, prototypes_flat)
        preds = dists.argmin(dim=-1)
        nlls = dd.get_dist_nll(dists, preds)
        preds = dd.is_anomaly(preds, nlls)

        return RoIs(
            features=torch.empty((0,)).cpu(),
            boxes=boxes.cpu(),
            scores=nlls.cpu(),
            labels=preds.cpu(),
        )


def main():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)

    # get run
    args = parser.parse_args()
    run = Run.init_from_name(args.run)

    # get network and dd
    efm = TimeCNN(run.model_config["model"])
    efm.load_state_dict(
        torch.load(
            os.path.join(run.checkpoints_dir, f"checkpoint_{args.checkpoint}.pt")
        )
    )
    detector = EFMDetector(efm, device=device)
    dd = DataDistributions(args.run, device)

    # fetch prototypes and label map
    export_cfg = run.data_cfg.get_export_cfg()

    prototypes: Tensor = torch.load(export_cfg.prototypes)
    prototypes = prototypes.to(device)
    prototypes_flat = prototypes.view(prototypes.shape[0], -1)

    with open(
        os.path.join(export_cfg.output_dir, "config.json"), "r+", encoding="utf-8"
    ) as f:
        label_map = json.loads(f.read())["label_mapping"]
    label_map: dict[int, str] = {int(k): v for k, v in label_map.items()}

    # add OOD label
    label_map[-1] = "Out-of-Distribution"

    # detect objects
    img_root = r"dataset/nuimages-v1.0/samples/CAM_FRONT"
    img_path = r"n003-2018-01-02-11-48-43+0800__CAM_FRONT__1514865148310836.jpg"
    img = decode_image(os.path.join(img_root, img_path)).to(device)

    rois = detector.forward(img, prototypes_flat, dd)

    # plot image
    box_labels = [
        f"{label_map[l.item()]}: {s.item():.3f}"  # type: ignore
        for l, s in zip(rois.labels, rois.scores)
    ]
    colors = [
        "red" if "Out-of-Distribution" not in label else "blue" for label in box_labels
    ]

    img_with_boxes = draw_bounding_boxes(
        img, rois.boxes, labels=box_labels, colors=colors, width=3  # type: ignore
    )
    height, width = img_with_boxes.shape[1], img_with_boxes.shape[2]
    dpi = 100
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    ax = fig.add_axes([0, 0, 1, 1])  # type: ignore
    ax.axis("off")

    ax.imshow(img_with_boxes.permute(1, 2, 0))

    plt.savefig(
        os.path.join("visu", "plots", "nuimg_output.pdf"),
        format="pdf",
        bbox_inches="tight",  # Removes remaining tiny margins
        pad_inches=0,  # Sets padding specifically to zero
    )
    plt.show()


if __name__ == "__main__":
    main()
