import os
from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch import Tensor

from fm.flow_matching import AffinePath
from fm.flow_matching.scheduler import CosineScheduler
from fm.modules import EMA

from data import RoIFeatureDataset, RoIFeatureDataLoader

from nn.run import Run
from nn.models import TimeCNN
from nn.losses import LOSS_DICT, apply_losses

parser = ArgumentParser()
parser.add_argument(
    "--model_config",
    type=str,
    default=os.path.join("nn", "models", "configs", "cnn.cfg.yaml"),
)
parser.add_argument(
    "--data_config",
    type=str,
    default=os.path.join("nn", "configs", "data.cfg.yaml"),
)
parser.add_argument(
    "--train_config",
    type=str,
    default=os.path.join("nn", "configs", "train.cfg.yaml"),
)
parser.add_argument(
    "--name",
    type=str,
)


def train():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # create run
    args = parser.parse_args()
    run = Run(args.model_config, args.data_config, args.train_config, args.name)

    # dataset
    ds = RoIFeatureDataset(run.data_cfg.get_export_cfg())

    # dataloader with prototypes as a variable
    dl = RoIFeatureDataLoader(
        ds,
        run.data_cfg.batch_size,
        run.data_cfg.shuffle,
        run.data_cfg.skip_last,
        device=device,
    )
    prototypes = dl.prototypes.to(device)

    # model
    net = TimeCNN(run.model_config["model"]).to(device)
    ema = EMA(net, rate=0.999)

    # path
    path = AffinePath(CosineScheduler())

    # optim and losses
    optim = torch.optim.AdamW(net.parameters(), lr=run.train_cfg.learn_rate)
    losses = {l: LOSS_DICT[l] for l in run.train_cfg.losses}

    # epoch loop
    for e in (pbar := tqdm(range(run.train_cfg.epochs))):
        # batch loop
        for x, y, _ in dl:
            # reset optimizer
            optim.zero_grad()

            # get x, y
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)

            # get all losses
            loss_vals = apply_losses(
                x,
                y,
                prototypes,
                path,
                net,
                losses,
                run.train_cfg.lambdas,
                **run.train_cfg.losses_kwargs,
            )

            # sum losses up
            loss: Tensor = sum(loss_vals.values())  # type: ignore

            # update grads and ema
            loss.backward()

            optim.step()
            ema.update_ema_t()

            # update batch loss
            run.update_batch_loss(loss_vals)

        # update epoch loss
        run.update_epoch_loss(num_batches=dl.num_batches)

        # log states
        run.log_state(epoch=e)

        pbar.set_description(f"Loss: {run.epoch_losses["total"][-1]:.4f}")

    # freeze model
    ema.to_model()
    net = net.eval()

    # plot losses
    run.plot_losses()

    # save net
    torch.save(net.state_dict(), run.model_fname)


if __name__ == "__main__":
    train()
