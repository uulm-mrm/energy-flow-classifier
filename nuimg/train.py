import os

from tqdm import tqdm

import torch

from flow_matching.flow_matching import AffinePath
from flow_matching.flow_matching.scheduler import CosineScheduler
from flow_matching.modules.utils import EMA

from models.cnn import TimeResCNN

from nuimg.utils import get_data_loss, get_noise_loss
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

    # flow matching setup
    net = TimeResCNN(
        in_c=c.IN_C, t_dims=c.T_DIMS, res_blocks=c.RES_BLOCKS, linear_layers=c.LINEAR
    ).to(c.DEVICE)

    ema = EMA(net, rate=0.999)
    path = AffinePath(CosineScheduler())

    # torch setup
    optim = torch.optim.AdamW(net.parameters(), lr=c.LR)
    losses = []

    # training loop
    for _ in (pbar := tqdm(range(c.EPOCHS))):
        epoch_loss = 0.0

        for x, y in dataloader:  # x: [B, *shape], y: [B, *shape]
            optim.zero_grad()

            # get loss components
            loss_data = get_data_loss(x, y, net, path)
            loss_noise = get_noise_loss(x, y, net, path)

            loss = loss_data + loss_noise

            # update params
            loss.backward()
            optim.step()
            ema.update_ema_t()

            epoch_loss = epoch_loss + loss.item()

        # add avg epoch loss
        _loss = epoch_loss / dataloader.num_batches
        losses.append(_loss)
        pbar.set_description(f"Loss: {_loss:.4f}")

    ema.to_model()
    net = net.eval()

    torch.save(net.state_dict(), os.path.join(c.SAVE_DIR, c.SAVE_NAME + ".pt"))


if __name__ == "__main__":
    train()
