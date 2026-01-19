import os

from tqdm import tqdm

import matplotlib.pyplot as plt

import torch

from flow_matching.flow_matching.distributions import MultiIndependentNormal
from flow_matching.flow_matching import ODEProcess, RungeKuttaIntegrator
from flow_matching.flow_matching.integrator_utils import RK4_TABLEAU

from nuimg.data import RoIFeatureDataset, RoIFeatureDataLoader, FEATURES_DATASET_DIR
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
        device=c.DEVICE,
    )

    # noise setup
    noise = MultiIndependentNormal(
        n=c.CLASSES,
        shape=c.SHAPE,
        r=c.R,
        var_coef=c.VAR,
        device=c.DEVICE,  # type: ignore
    )

    # load model
    unet = UNet(
        in_c=c.SHAPE[0], out_c=c.SHAPE[0], features=c.FEATURES, t_dims=c.T_DIMS
    ).to(c.DEVICE)
    unet.load_state_dict(torch.load(os.path.join(c.SAVE_DIR, c.SAVE_NAME + ".pt")))
    unet.eval()

    # process setup
    proc = ODEProcess(unet, RungeKuttaIntegrator(RK4_TABLEAU, device=c.DEVICE))  # type: ignore

    total_true_positives = 0
    total_points = 0
    accuracies = []

    for x, y in (pbar := tqdm(dataloader)):
        intervals = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float32, device=c.DEVICE
        ).expand(x.shape[0], 2)

        _, x_traj = proc.sample(x, intervals, steps=c.ODE_STEPS)
        sols = x_traj[-1]
        scores = noise.get_scores(sols)

        preds = torch.argmax(scores, dim=1)
        true_positives = sum(preds == y.reshape(-1))
        total_true_positives += true_positives
        total_points += y.shape[0]
        accuracies.append((total_true_positives / total_points).cpu().item())  # type: ignore

        pbar.set_description(f"Running Accuracy: {accuracies[-1]:.4f}")

    print(f"Total Accuracy: {(total_true_positives / total_points):.4f}")
    plt.plot(accuracies)
    plt.show()


if __name__ == "__main__":
    evaluate()
