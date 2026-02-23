import os
from argparse import ArgumentParser, ArgumentTypeError
import json
import logging

from tqdm import tqdm

import torch

from torchvision.io.image import decode_image

import data.data_model as m

from data.config import ExportConfig
from data.roi_align import RoIAlignExtractor


def __kvp(argument: str):
    if "=" not in argument:
        raise ArgumentTypeError(f"Argument {argument} is not in key:value format")

    k, v = argument.split(":", 1)
    return k, v


parser = ArgumentParser()
parser.add_argument(
    "--version",
    type=str,
    default="val2017",
)

# https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
parser.add_argument(
    "--labels",
    type=__kvp,
    nargs="+",
    default=[],
)


def export(cfg: ExportConfig):
    # make dirs
    os.makedirs(cfg.output_dir, exist_ok=True)

    # set up logging
    logging.basicConfig(
        filename=os.path.join(cfg.output_dir, "data_export.log"),
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # set up extractor
    extractor = RoIAlignExtractor(device=torch.device("cuda"))
    id_mask = torch.tensor(list(cfg.label_mapping.keys()), device="cpu")

    # write categories to file
    with open(os.path.join(cfg.output_dir, "config.json"), "w+", encoding="utf-8") as f:
        f.write(json.dumps(cfg.__dict__, indent=4))

    # lookup table
    lut = {"fnames": [], "offsets": []}
    offset = 0

    # go over images
    for img_fname in (pbar := tqdm(os.listdir(cfg.input_dir))):
        pbar.set_description(f"Processing {img_fname}")

        img_fpath = os.path.join(cfg.input_dir, img_fname)

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
        frame = m.LabeledFrame(features, labels.view(-1))

        fname = img_fname[:-3] + "pt"
        torch.save(frame.__dict__, os.path.join(cfg.output_dir, fname))

        # update lut
        offset += frame.labels.shape[0]
        lut["fnames"].append(fname)
        lut["offsets"].append(offset)

    # write lut
    with open(
        os.path.join(cfg.output_dir, "index_lookup_table.json"),
        "w+",
        encoding="utf-8",
    ) as f:
        f.write(json.dumps(lut))


def main():
    args = parser.parse_args()
    cfg = ExportConfig(dataset="COCO", version=args.version)
    cfg.set_labels(dict(args.labels))

    export(cfg)


if __name__ == "__main__":
    main()
