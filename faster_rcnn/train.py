import os
from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch import Tensor

from data import RoIFeatureDataset, RoIFeatureDataLoader

from faster_rcnn.run import Run
from faster_rcnn.net import FasterRCNNPredictor

parser = ArgumentParser()
parser.add_argument(
    "--data_config",
    type=str,
    default=os.path.join("faster_rcnn", "configs", "data.cfg.yaml"),
)
parser.add_argument(
    "--train_config",
    type=str,
    default=os.path.join("faster_rcnn", "configs", "train.cfg.yaml"),
)
parser.add_argument("--num_cls", type=int, default=5)
parser.add_argument("--name", type=str)


def train():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # create run
    args = parser.parse_args()
    run = Run(args.data_config, args.train_config, args.name, train=True)
    save_every = run.train_cfg.epochs // 20

    # dataset
    ds = RoIFeatureDataset(run.data_cfg.get_export_cfg())

    # dataloader with prototypes as a variable
    dl = RoIFeatureDataLoader(
        ds,
        run.data_cfg.batch_size,
        run.data_cfg.shuffle,
        run.data_cfg.skip_last,
        device=device,
        box_head_features=True,
    )

    # model
    net = FasterRCNNPredictor(num_cls=args.num_cls, device=device)

    # optim and losses
    optim = torch.optim.AdamW(net.parameters(), lr=run.train_cfg.learn_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # epoch loop
    for e in (pbar := tqdm(range(1, run.train_cfg.epochs + 1))):
        # batch loop
        for x, _, y in dl:
            # reset optimizer
            optim.zero_grad()

            # get x, y
            x: Tensor = x.to(device)
            y: Tensor = y.to(device)

            # predict
            y_hat = net.forward(x)

            # get loss
            loss = criterion.forward(y_hat, y)

            # update grads
            loss.backward()
            optim.step()

            # update batch loss
            run.update_batch_loss(loss)

        # update epoch loss
        run.update_epoch_loss(num_batches=dl.num_batches)

        # log states
        run.log_state(epoch=e)

        # save net every 1/10 epochs
        if (e % save_every) == 0:
            torch.save(net.state_dict(), run.checkpoint_fname.format(e // save_every))

        pbar.set_description(f"Loss: {run.epoch_losses["total"][-1]:.4f}")

    # freeze model
    net = net.eval()

    # plot losses
    run.plot_losses()


if __name__ == "__main__":
    train()
