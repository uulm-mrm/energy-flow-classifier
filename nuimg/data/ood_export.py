import os
import json
import logging

from tqdm import tqdm

import torch

from torchvision.io.image import decode_image

from nuimg.data.roi_align import RoIAlignExtractor
import nuimg.data.consts as c
import nuimg.data.model as m


def export():

    # make dirs
    os.makedirs(c.FEATURES_DATASET_DIR, exist_ok=True)

    # set up logging
    logging.basicConfig(
        filename=os.path.join(c.FEATURES_DATASET_DIR, "data_export.log"),
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # set up extractor
    extractor = RoIAlignExtractor(device=torch.device("cuda"))
    id_mask = torch.tensor(list(c.COCO_ID_CATEGORIES.keys()), device="cpu")

    # write categories to file
    with open(
        os.path.join(c.FEATURES_DATASET_DIR, "cat_to_name.json"), "w+", encoding="utf-8"
    ) as f:
        f.write(json.dumps(c.COCO_ID_CATEGORIES))

    # lookup table
    lut = {"fnames": [], "offsets": []}
    offset = 0

    # empty tensor for deltas so as to not constantly recreate it
    empty_deltas = torch.empty(size=(0,), device="cpu")

    # go over images
    for img_fname in (pbar := tqdm(os.listdir(c.COCO_DATASET_ROOT))):
        pbar.set_description(f"Processing {img_fname}")

        img_fpath = os.path.join(c.COCO_DATASET_ROOT, img_fname)

        img = decode_image(img_fpath)

        try:
            rois = extractor.forward(img)
        except Exception as e:  # pylint: disable=W0718
            logging.warning("Image %s had error: %s", img_fname, e)
            continue

        if not rois:
            logging.info("No RoIs extracted in %s", img_fname)
            continue

        label_mask = torch.isin(rois.labels, id_mask)
        label_mask = ~label_mask  # you want stuff that's not in the labels of coco

        if torch.all(~label_mask):
            logging.info("No labels matching wanted labels found in %s", img_fname)
            continue

        labels = rois.labels[label_mask]
        features = rois.features[label_mask]

        # save frame with empty deltas
        frame = m.LabeledFrame(features, labels, empty_deltas)

        fname = img_fname[:-3] + "pt"
        torch.save(frame.__dict__, os.path.join(c.FEATURES_DATASET_DIR, fname))

        # update lut
        offset += frame.labels.shape[0]
        lut["fnames"].append(fname)
        lut["offsets"].append(offset)

    # write lut
    with open(
        os.path.join(c.FEATURES_DATASET_DIR, "index_lookup_table.json"),
        "w+",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(lut))


if __name__ == "__main__":
    export()
