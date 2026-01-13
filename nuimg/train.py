import os

from tqdm import tqdm

from matplotlib import pyplot as plt

import torch

from flow_matching.flow_matching import AffinePath
from flow_matching.flow_matching.distributions import MultiIndependentNormal
from flow_matching.flow_matching.scheduler import CosineScheduler
from flow_matching.modules.utils import EMA

from models.unet import UNet

import nuimg.consts as c
from nuimg.data import (
    RoIFeatureDataset,
    RoIFeatureDataLoader,
    FEATURES_DATASET_DIR,
)


def train():
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
    noise = MultiIndependentNormal(c=c.CLASSES, shape=c.SHAPE, k=c.K, device=c.DEVICE)  # type: ignore

    # flow matching setup
    unet = UNet(
        in_c=c.SHAPE[0], out_c=c.SHAPE[0], features=c.FEATURES, t_dims=c.T_DIMS
    ).to(c.DEVICE)
    ema = EMA(unet, rate=0.999)
    path = AffinePath(CosineScheduler())

    # torch setup
    optim = torch.optim.AdamW(unet.parameters(), lr=c.LR)
    losses = []

    # training loop
    for _ in (pbar := tqdm(range(c.EPOCHS))):
        epoch_loss = 0.0

        for x, y in dataloader:  # x: [B, *shape], y: [B, 1]
            optim.zero_grad()

            # sort data per category
            # TODO: no need to sort, just in sample arbitrary pass y to apply means accordingly
            # but this actually costs nothing
            x_data, y = dataset.sort_per_category(x, y.reshape(-1))

            # sample noise
            counts = dataset.counts_per_category(y)
            x_noise = noise.sample_arbitrary(*counts)

            # sample time
            t = torch.rand((x_data.shape[0],), dtype=x_data.dtype, device=x_data.device)

            # sample and predict path
            path_sample = path.sample(x_noise, x_data, t=t)
            dxt_hat = unet.forward(path_sample.xt, t.unsqueeze(1))  # time to vector

            # calculate loss
            loss = (dxt_hat - path_sample.dxt).square().mean()

            # update params
            loss.backward()
            optim.step()
            ema.update_ema_t()

            epoch_loss = epoch_loss + loss.item()

        # add avg epoch loss
        _loss = epoch_loss / dataloader.num_batches
        losses.append(_loss)
        pbar.set_description(f"Loss: {_loss:.3f}")

    ema.to_model()
    unet = unet.eval()

    torch.save(unet.state_dict(), os.path.join(c.SAVE_DIR, c.SAVE_NAME + ".pt"))

    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    train()
