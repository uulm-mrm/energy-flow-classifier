# pylint: disable=W0201

from os import path
from typing import Literal

import torch
from torch import Tensor

from torchvision import transforms
from torchvision.datasets import MNIST

DATASETS_ROOT = path.join(path.dirname(__file__), "datasets")


def get_mnist(subset: Literal["train", "test"]) -> MNIST:
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_dataset = MNIST(
        root=DATASETS_ROOT,
        train=(subset == "train"),
        download=True,
        transform=transform,
    )

    return mnist_dataset


class MNISTSampler:
    def __init__(
        self, mnist: MNIST, classes: tuple[int, ...], batch_size: int, device: str
    ) -> None:
        self.device = device

        self.data, self.labels = self.__unpack_tuples(mnist)
        self.classes = classes

        self.indices = self.__get_class_indices()
        self.max_len = max(len(v) for v in self.indices.values())

        self.batch_size = min(batch_size, max(len(v) for v in self.indices.values()))

        self.batches = max(len(v) // batch_size for v in self.indices.values()) + 1

    def __unpack_tuples(self, mnist: MNIST) -> tuple[Tensor, Tensor]:
        data = []
        labels = []
        for x, y in mnist:
            data.append(x)
            labels.append(y)

        data = torch.stack(data, dim=0)
        labels = torch.tensor(labels)
        return data, labels

    def __get_class_indices(self) -> dict[int, Tensor]:
        indices = {}
        for c in self.classes:
            indices[c] = torch.where(self.labels == c)[0]

        return indices

    def __iter__(self):
        # batch indices is the max of
        self.batch_indices = torch.randperm(self.max_len)
        self.current_batch = 0

        # extend randomly the class indices to match the longest one
        self.extended_indices = {}
        for c in self.classes:
            diff = len(self.batch_indices) - len(self.indices[c])
            self.extended_indices[c] = torch.cat(
                [
                    self.indices[c],
                    self.indices[c][torch.randperm(len(self.indices[c]))[:diff]],
                ],
                dim=0,
            )

        return self

    def __next__(self) -> tuple[Tensor, ...]:
        # here you check if you still have batches to do
        if self.current_batch < self.batches:
            batch_indices = self.batch_indices[
                self.current_batch
                * self.batch_size : (self.current_batch + 1)
                * self.batch_size
            ]

            sampled_data = tuple(
                self.data[self.extended_indices[c][batch_indices]].to(self.device)
                for c in self.classes
            )

            self.current_batch += 1

            return sampled_data

        raise StopIteration


if __name__ == "__main__":
    ds = get_mnist("train")
    sampler = MNISTSampler(ds, (0, 1, 2), batch_size=10_000, device="cuda")

    for k, v in sampler.indices.items():
        print(k, len(v))

    print(sampler.batch_size)
    print(sampler.batches)

    for _ in range(2):
        for x1, x2, x3 in sampler:
            print(x1.shape, x2.shape, x3.shape)
