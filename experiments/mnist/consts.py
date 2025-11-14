from dataclasses import dataclass


@dataclass
class __Config:
    # torch consts
    device = "cuda:0"

    # data consts
    shape = (1, 32, 32)
    classes = tuple(range(5))
    num_classes = len(classes)
    k = 3.0

    # net consts
    features = [32, 64, 128]

    # training consts
    batch_size = 512
    t_dims = 256
    lr = 1e-3
    epochs = 1024


CONFIG = __Config
