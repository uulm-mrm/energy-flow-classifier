import os
import math

from tqdm import tqdm

import matplotlib.pyplot as plt

import torch

from flow_matching.flow_matching import PotentialProcess, RungeKuttaIntegrator
from flow_matching.flow_matching.integrator_utils import RK4_TABLEAU

from nuimg.data import RoIFeatureDataset, RoIFeatureDataLoader, FEATURES_DATASET_DIR
import nuimg.utils as u
import nuimg.consts as c

from models.cnn import TimeCNN


def evaluate():
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

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
    net = TimeCNN(
        in_c=c.IN_C, t_dims=c.T_DIMS, res_blocks=c.RES_BLOCKS, linear_layers=c.LINEAR
    ).to(c.DEVICE)
    net.load_state_dict(torch.load(os.path.join(c.SAVE_DIR, c.SAVE_NAME + ".pt")))
    net.eval()

    # process setup
    proc = PotentialProcess(net, RungeKuttaIntegrator(RK4_TABLEAU, device=c.DEVICE))  # type: ignore

    total_true_positives = 0
    total_points = 0
    accuracies = []

    for x, y in (pbar := tqdm(dataloader)):
        intervals = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float32, device=c.DEVICE
        ).expand(x.shape[0], 2)

        _, x_traj = proc.sample(x, intervals, steps=c.ODE_STEPS)
        sols = x_traj[-1]

        # calculate evidence metrics
        measure = torch.cdist(sols.flatten(1), deltas)
        print("Labels: ", y)
        print("Sols: ", sols.flatten(1)[:, :3])

        preds = torch.argmin(measure, dim=1)
        true_positives = sum(preds == y.squeeze(1))

        total_true_positives += true_positives
        total_points += y.shape[0]

        accuracies.append((total_true_positives / total_points).cpu().item())  # type: ignore

        pbar.set_description(f"Running Accuracy: {accuracies[-1]:.4f}")

    print(f"Total Accuracy: {(total_true_positives / total_points):.4f}")
    plt.plot(accuracies)
    plt.show()


if __name__ == "__main__":
    evaluate()
