from torch import nn, Tensor
import torch.nn.functional as F

from flow_matching.modules import cnn
from flow_matching.modules.utils import (
    TimeDependentSequential,
    TimeDependentModule,
    SinusoidalTimeEmbedding,
)


class DoubleConv(TimeDependentModule):
    def __init__(self, in_c: int, out_c: int, t_dims: int) -> None:
        super().__init__()

        self.res_conv = cnn.TimeConditionedConv(
            in_c, out_c, ksize=1, t_dim=t_dims, stride=1, padding=0
        )

        self.conv = TimeDependentSequential(
            cnn.TimeConditionedConv(
                in_c, out_c, ksize=3, t_dim=t_dims, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
            cnn.TimeConditionedConv(
                out_c, out_c, ksize=3, t_dim=t_dims, stride=1, padding=1
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.conv(x, t) + self.res_conv(x, t)
        return F.relu(h)


class Downsample(TimeDependentModule):
    def __init__(self, in_c: int, out_c: int, t_dims: int) -> None:
        super().__init__()

        self.down = TimeDependentSequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c, t_dims),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.down(x, t)


class Upsample(TimeDependentModule):
    def __init__(self, in_c: int, out_c: int, t_dims: int) -> None:
        super().__init__()

        self.up = TimeDependentSequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            DoubleConv(out_c, out_c, t_dims),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.up(x, t)


class UNet(TimeDependentModule):
    def __init__(self, in_c: int, out_c: int, features: list[int], t_dims: int) -> None:
        super().__init__()

        self.time_emb = (
            SinusoidalTimeEmbedding(emb_dim=t_dims) if t_dims != 1 else nn.Identity()
        )

        self.project = nn.Conv2d(in_c, features[0], kernel_size=1, padding=0)

        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()

        # down
        channels = features[0]
        for feature in features:
            self.downsample.append(Downsample(channels, feature, t_dims))
            channels = feature

        # bottleneck
        self.bottleneck = TimeDependentSequential(
            cnn.TimeConditionedConv(
                features[-1],
                2 * features[-1],
                ksize=3,
                t_dim=t_dims,
                padding=1,
                stride=1,
            ),
            cnn.ConvMHSA(2 * features[-1], heads=(2 * features[-1]) // 4),
            nn.ReLU(inplace=True),
            cnn.TimeConditionedConv(
                2 * features[-1],
                features[-1],
                ksize=3,
                t_dim=t_dims,
                padding=1,
                stride=1,
            ),
            cnn.ConvMHSA(features[-1], heads=features[-1] // 4),
            nn.ReLU(inplace=True),
        )

        # up
        channels = features[-1]
        for feature in [features[0], *features][::-1]:
            self.upsample.append(Upsample(channels, feature, t_dims))
            channels = feature

        self.reproject = nn.Conv2d(features[0], out_c, kernel_size=1, padding=0)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): size (B, D...)
            t (Tensor): size (B,)

        Returns:
            Tensor: size (B, D...)
        """
        t = t.reshape(-1, 1)

        t_emb = self.time_emb(t)

        x = self.project.forward(x)

        skips = [x]
        for down in self.downsample:
            x = down.forward(x, t_emb)
            skips.append(x)

        x = self.bottleneck.forward(x, t_emb)

        for i, up in enumerate(self.upsample):
            x = up.forward(x, t_emb)
            skip = skips[-(i + 1)]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = x + skip

        x = self.reproject.forward(x)

        return x
