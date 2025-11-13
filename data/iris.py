import torch
from torch import Tensor

from sklearn.datasets import load_iris


def get_iris(device: str) -> tuple[Tensor, Tensor, Tensor]:
    x, y = load_iris(return_X_y=True)

    x = torch.from_numpy(x)
    x = x.float().to(device)

    xmin = x.min(dim=0, keepdim=True)[0]
    xmax = x.max(dim=0, keepdim=True)[0]

    x = (x - xmin) / (xmax - xmin) * 2 - 1

    y = torch.from_numpy(y).to(device)

    return x[y == 0], x[y == 1], x[y == 2]


def main():
    x1, x2, x3 = get_iris(device="cuda")
    print(x1, x2, x3)


if __name__ == "__main__":
    main()
