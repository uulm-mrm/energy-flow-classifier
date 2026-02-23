import os
import shutil
from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch import Tensor

from data import RoIFeatureDataset, RoIFeatureDataLoader

from faster_rcnn.net import FasterRCNNPredictor
from faster_rcnn.configs import EvalConfig, load_yaml

parser = ArgumentParser()
parser.add_argument(
    "--eval_config",
    type=str,
    default=os.path.join("faster_rcnn", "configs", "eval.cfg.yaml"),
)
parser.add_argument("--num_cls", type=int, default=5)
parser.add_argument("--name", type=str, required=False)


def evaluate():
    # torch consts
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # get config
    args = parser.parse_args()
    cfg = EvalConfig(**load_yaml(args.eval_config)["eval_config"])

    # make cfg dir
    eval_dir = cfg.get_eval_dir(args.name)
    os.makedirs(eval_dir, exist_ok=True)

    # copy config
    shutil.copy(args.eval_config, os.path.join(eval_dir, "eval.cfg.yaml"))

    # dataset
    ds = RoIFeatureDataset(cfg.get_export_cfg())

    # dataloader
    dl = RoIFeatureDataLoader(
        ds, cfg.batch_size, shuffle=False, skip_last=False, device=device, train=False
    )

    # model
    net = FasterRCNNPredictor(args.num_cls, device=device)
    net.load_state_dict(torch.load(cfg.get_model_weights()))
    net.eval()

    # go over data
    for batch, (x, _, y) in tqdm(enumerate(dl), desc="Processing Batches"):
        state = {}

        x: Tensor = x.to(device)
        y: Tensor = y.to(device)
        y = y.view(-1, 1)

        # compute potential
        with torch.no_grad():
            preds = net.forward(x)

        # save state as pt
        state["meta"] = ["labels", "preds"]
        state["sizes"] = [y.shape[1], preds.shape[1]]
        state["data"] = torch.cat([y, preds], dim=1)

        fname = os.path.join(eval_dir, f"batch_{batch}.pt")
        torch.save(state, fname)


if __name__ == "__main__":
    evaluate()
