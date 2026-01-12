import os
import json

from tqdm import tqdm

import torch

from torchvision.io.image import decode_image
from torchvision.ops import box_iou

from nuimages import NuImages

import nuimg.data.consts as c
import nuimg.data.model as m

from nuimg.data.roi_align import RoIAlignExtractor
from nuimg.data.utils import category_mappings, get_sample_data_gt


def main():
    # make dir to save data
    os.makedirs(c.FEATURES_DATASET_DIR, exist_ok=True)

    nuim = NuImages(
        c.NUIM_DATASET_VERSION, c.NUIM_DATASET_ROOT, verbose=True, lazy=True
    )
    extractor = RoIAlignExtractor(
        labels=list(c.COCO_ID_CATEGORIES.values()), device=torch.device("cuda")
    )

    # make and save label mappings
    nuim_tokens_to_category, category_to_name = category_mappings(nuim)

    with open(
        os.path.join(c.FEATURES_DATASET_DIR, "token_to_cat.json"),
        "w+",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(nuim_tokens_to_category))

    with open(
        os.path.join(c.FEATURES_DATASET_DIR, "cat_to_name.json"), "w+", encoding="utf-8"
    ) as f:
        f.write(json.dumps(category_to_name))

    # loop over images
    for sample_data in tqdm(nuim.sample_data, desc="Iterating Samples"):

        # only key frames have labels
        if not sample_data["is_key_frame"]:
            continue

        # extract ground truth from image
        gt = get_sample_data_gt(nuim, sample_data["token"], nuim_tokens_to_category)

        if not gt:
            print(f"No boxes in {sample_data["filename"]}")
            continue

        # extract features using FRCNN from image
        img = decode_image(os.path.join(c.NUIM_DATASET_ROOT, sample_data["filename"]))
        rois = extractor.forward(img)

        if not rois:
            print(f"No RoIs extracted in {sample_data["filename"]}")
            continue

        # match FRCNN boxes of features to GT features
        # [rois, gts]
        ious = box_iou(rois.boxes, gt.boxes)

        # get max ious for filtering
        max_ious, max_idx = ious.max(dim=1)
        iou_thresh = max_ious >= 0.75

        # filter features and labels
        labels = gt.labels[max_idx][iou_thresh]  # [gt,] -> [roi,] -> [best fit]
        features = rois.features[iou_thresh]  # [roi,] -> [best fit]

        frame = m.LabeledFrame(features=features, labels=labels)

        # save labeled frame under filename
        # loaded as a dict {"features": Tensor, "labels": Tensor}
        fname = sample_data["filename"].split("/")[-1][:-3] + "pt"

        torch.save(frame.__dict__, os.path.join(c.FEATURES_DATASET_DIR, fname))


if __name__ == "__main__":
    main()
