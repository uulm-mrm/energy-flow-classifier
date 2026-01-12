from dataclasses import dataclass

from torch import Tensor


@dataclass
class GroundTruth:
    boxes: Tensor  # [n, 4]; xmin, ymin, xmax, ymax
    labels: Tensor


@dataclass
class RoIs:
    features: Tensor  # [n, 256, 7, 7]
    boxes: Tensor  # [n, 4]; xmin, ymin, xmax, ymax
    scores: Tensor  # [n,]
    labels: Tensor  # [n,]


@dataclass
class LabeledFrame:
    features: Tensor  # [n, 256, 7, 7]
    labels: Tensor  # [n,]
