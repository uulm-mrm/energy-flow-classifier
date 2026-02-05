import os
import math

from tqdm import tqdm

import torch

from flow_matching.flow_matching import ODEProcess, RungeKuttaIntegrator
from flow_matching.flow_matching.integrator_utils import RK4_TABLEAU

from nuimg.data import RoIFeatureDataset, RoIFeatureDataLoader, FEATURES_DATASET_DIR
import nuimg.utils as u
import nuimg.consts as c

from models.unet import UNet


def evaluate():
    torch.manual_seed(42)

    # dataset and dataloader
    dataset = RoIFeatureDataset(dataset_dir=FEATURES_DATASET_DIR)
    dataloader = RoIFeatureDataLoader(
        dataset,
        batch_size=c.BATCH_SIZE,
        shuffle=c.SHUFFLE,
        skip_last=c.SKIP_LAST,
        train=False,
        device=c.DEVICE,
    )

    # dirac deltas setup
    deltas = torch.zeros(
        size=(c.CLASSES, math.prod(c.SHAPE)), dtype=torch.float32, device=c.DEVICE
    )
    deltas[:, : c.CLASSES] = torch.eye(c.CLASSES, dtype=torch.float32, device=c.DEVICE)

    # load model
    unet = UNet(
        in_c=c.SHAPE[0], out_c=c.SHAPE[0], features=c.FEATURES, t_dims=c.T_DIMS
    ).to(c.DEVICE)
    unet.load_state_dict(torch.load(os.path.join(c.SAVE_DIR, c.SAVE_NAME + ".pt")))
    unet.eval()

    # process setup
    proc = ODEProcess(unet, RungeKuttaIntegrator(RK4_TABLEAU, device=c.DEVICE))  # type: ignore

    for x, y in tqdm(dataloader):
        intervals = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float32, device=c.DEVICE
        ).expand(x.shape[0], 2)

        _, x_traj = proc.sample(x, intervals, steps=c.ODE_STEPS)
        sols = x_traj[-1]

        # calculate evidence metrics
        measure = u.cosine_similarity(sols, deltas, c.CLASSES)
        quality = u.norm_decay(sols, c.CLASSES)

        measure = (measure + 1) * 0.5  # normalize to (0, 1]

        belief, vacuity = u.credal_measures(measure, quality, W=3.0)

        # TODO: this is just WIP add OOD metrics here
        print("COCO categories: ", y)
        print("Belief: ", belief)
        print("Vacuity: ", vacuity)
        break


if __name__ == "__main__":
    evaluate()
