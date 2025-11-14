from dataclasses import dataclass
import math


@dataclass
class __Config:
    device = "cuda:0"
    shape = (1, 32, 32)
    classes = tuple(range(2))

    num_classes = len(classes)

    sigma = 1.0
    k = 3.0
    r = k * sigma * math.prod(shape) ** 0.5

    features = [16, 32, 64]

    batch_size = 512
    t_dims = 256
    lr = 1e-3
    epochs = 2048


CONFIG = __Config
