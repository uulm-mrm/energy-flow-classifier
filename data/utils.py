from typing import Optional

import torch

from nuimages import NuImages

from data.data_model import GroundTruth


def category_mappings(
    nuim: NuImages, label_set: set[str]
) -> tuple[dict[str, int], dict[int, str]]:
    """Maps token categories to integers and maps integers to names

    Args:
        nuim (NuImages): nuimages dataset

    Returns:
        tuple[dict[str, int], dict[int, str]]: token -> int, int -> name
    """

    categories = nuim.category
    categories = list(filter(lambda d: d["name"] in label_set, categories))

    token_to_cat = {}
    cat_to_name = {}

    for i, cat in enumerate(categories):
        token_to_cat[cat["token"]] = i
        cat_to_name[i] = cat["name"]

    return token_to_cat, cat_to_name


def get_sample_data_gt(
    nuim: NuImages, sample_data_token: str, token_to_cat: dict[str, int]
) -> Optional[GroundTruth]:
    """Gets all gt boxes and labels found in an image with the sample_data_token

    Args:
        nuim (NuImages): nuimages dataset
        sample_data_token (str): sample data to look into
        token_to_cat (dict[str, int]): token to category dict for label mapping

    Returns:
        Optional[GroundTruth]: gt boxes [N, 4] and labels [N,] found in sample data
            if none are found then returns None
    """

    anns = nuim.object_ann  # list ann dicts
    boxes = []
    labels = []

    # seems slow but isn't really
    for ann in anns:
        if ann["sample_data_token"] != sample_data_token:
            continue

        # if unknown class then also skip
        if ann["category_token"] not in token_to_cat:
            continue

        # boxes are in xmin, ymin, xmax, ymax
        boxes.append(ann["bbox"])
        labels.append([token_to_cat[ann["category_token"]]])

    return (
        GroundTruth(
            boxes=torch.tensor(boxes, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.long),
        )
        if boxes
        else None
    )
