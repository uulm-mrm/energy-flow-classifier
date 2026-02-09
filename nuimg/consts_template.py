import os

import torch

from nuimg.data.consts import NUIMG_ID_CATEGORIES

# general
DEVICE = torch.device("cuda")

# dataset consts
BATCH_SIZE = 24
SHUFFLE = True
SKIP_LAST = True

# noise setup
CLASSES = len(NUIMG_ID_CATEGORIES)
SHAPE = (256, 7, 7)

# model setup
IN_C = SHAPE[0]
T_DIMS = 128
LINEAR = 2

# training setup
LR = 1e-3
EPOCHS = 1_000

# save trained model setup
SAVE_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
SAVE_NAME = "test_model"  # without .pt

# eval setup
ODE_STEPS = 100
