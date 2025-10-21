from os import path
from typing import Literal

from torch.utils.data import Dataset, Subset

from torchvision import transforms
from torchvision.datasets import MNIST

DATASETS_ROOT = path.join(path.dirname(__file__), "datasets")


def get_mnist(subset: Literal["train", "test"]) -> Dataset:
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_dataset = MNIST(
        root=DATASETS_ROOT,
        train=(subset == "train"),
        download=True,
        transform=transform,
    )

    return mnist_dataset


def sample_mnist(
    dataset: Dataset,
    classes: list[int],
) -> Dataset:
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]  # type: ignore
    sampled_dataset = Subset(dataset, indices)
    return sampled_dataset


if __name__ == "__main__":
    ds = get_mnist("train")
    ds = sample_mnist(ds, [1, 2, 3])

    print(ds[2])
