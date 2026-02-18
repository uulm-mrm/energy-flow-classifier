import os
from argparse import ArgumentParser, Namespace
import json
import logging

from tqdm import tqdm

import torch

from torchvision.io.image import decode_image
from torchvision.ops import box_iou

from nuimages import NuImages

import data.data_model as m

from data.config import ExportConfig
from data.roi_align import RoIAlignExtractor
from data.utils import category_mappings, get_sample_data_gt
from data.make_prototypes import compute_prototypes

parser = ArgumentParser()
parser.add_argument(
    "--version",
    type=str,
    default="v1.0-mini",
    choices=["v1.0-mini", "v1.0-train", "v1.0-val"],
)
parser.add_argument(  # https://www.nuscenes.org/nuimages#data-annotation
    "--labels",
    type=str,
    nargs="+",
    default=["human.pedestrian.adult", "vehicle.car"],
)


def export(cfg: ExportConfig, args: Namespace):
    # iou threshold
    iou_thresh = 0.75

    # make dir to save data
    os.makedirs(cfg.output_dir, exist_ok=True)

    # start up logger
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, "data_export.log"),
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    nuim = NuImages(cfg.version, cfg.input_dir, verbose=True, lazy=True)
    extractor = RoIAlignExtractor(device=torch.device("cuda"))

    # make and save label mappings
    nuim_tokens_to_category, category_to_name = category_mappings(nuim, args.labels)
    cfg.set_labels(category_to_name)

    # save config
    with open(os.path.join(cfg.output_dir, "config.json"), "w+", encoding="utf-8") as f:
        f.write(json.dumps(cfg.__dict__, indent=4))

    # lookup table for quicker indexing later on
    lut = {"fnames": [], "offsets": []}
    offset = 0

    # loop over images
    for sample_data in tqdm(nuim.sample_data[:100_000], desc="Iterating Samples"):

        # only key frames have labels
        if not sample_data["is_key_frame"]:
            continue

        # extract ground truth from image
        gt = get_sample_data_gt(nuim, sample_data["token"], nuim_tokens_to_category)

        if not gt:
            logging.info("No boxes in %s", sample_data["filename"])
            continue

        # extract features using FRCNN from image
        img = decode_image(os.path.join(cfg.input_dir, sample_data["filename"]))
        rois = extractor.forward(img)

        if not rois:
            logging.info("No RoIs extracted in %s", sample_data["filename"])
            continue

        # match FRCNN boxes of features to GT features
        # [rois, gts]
        ious = box_iou(rois.boxes, gt.boxes)

        # get max ious for filtering
        max_ious, max_idx = ious.max(dim=1)
        iou_mask = max_ious >= iou_thresh

        if torch.all(~iou_mask):
            logging.info("No overlap between boxes in %s", sample_data["filename"])
            continue

        # filter features and labels
        labels = gt.labels[max_idx][iou_mask]  # [gt,] -> [roi,] -> [best fit]
        features = rois.features[iou_mask]  # [roi,] -> [best fit]

        frame = m.LabeledFrame(features=features, labels=labels.view(-1))

        # save labeled frame under filename
        # loaded as a dict {"features": Tensor, "labels": Tensor}
        fname = sample_data["filename"].split("/")[-1][:-3] + "pt"

        torch.save(frame.__dict__, os.path.join(cfg.output_dir, fname))

        # update LuT
        offset += frame.labels.shape[0]
        lut["fnames"].append(fname)
        lut["offsets"].append(offset)

    # write lookup table
    with open(
        os.path.join(cfg.output_dir, "index_lookup_table.json"), "w+", encoding="utf-8"
    ) as f:
        f.write(json.dumps(lut))


def main():
    args = parser.parse_args()
    cfg = ExportConfig(dataset="nuimages-v1.0", version=args.version)

    export(cfg, args)
    compute_prototypes(cfg)


if __name__ == "__main__":
    main()
