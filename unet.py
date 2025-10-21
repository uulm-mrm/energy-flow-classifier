import torch
from torch import nn, Tensor
import torch.nn.functional as F

from flow_matching.modules import cnn
from flow_matching.modules.utils import (
    TimeDependentModule,
    TimeDependentSequential,
    SinusoidalTimeEmbedding,
)


class NConv2D(TimeDependentModule):
    """Does time conditioned convolutions N times in a row"""

    def __init__(self, in_c: int, out_c: int, t_dims: int, times: int = 2) -> None:
        super().__init__()

        convs = []
        for i in range(times):
            convs.append(
                cnn.TimeConditionedConv(
                    in_c if i == 0 else out_c,
                    out_c,
                    ksize=3,
                    t_dim=t_dims,
                    stride=1,
                    padding=1,
                )
            )

        self.convs = TimeDependentSequential(*convs)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.convs.forward(x, t)


class Downsample(TimeDependentModule):
    """Halves the input size and performs NConv2D on the result"""

    def __init__(self, in_c: int, out_c: int, t_dims: int, times: int = 2) -> None:
        super().__init__()

        self.downsample = TimeDependentSequential(
            nn.MaxPool2d(kernel_size=2), NConv2D(in_c, out_c, t_dims, times)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.downsample.forward(x, t)


class Upsample(TimeDependentModule):
    """Doubles the input size then performs NConv2D on the result"""

    def __init__(self, in_c: int, out_c: int, t_dims: int, times: int = 2) -> None:
        super().__init__()

        self.up = cnn.TimeConditionedUpConv(in_c, out_c, t_dims, scale=2)
        self.conv = NConv2D(out_c + out_c, out_c, t_dims, times=times)

    def __pad_x_for_skip(self, x: Tensor, skip: Tensor) -> Tensor:
        if x.shape == skip.shape:
            return x

        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]

        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])

        return x

    def forward(self, x: Tensor, t: Tensor, skip: Tensor) -> Tensor:  # type: ignore
        z = self.up.forward(x, t)  # [B, C, ~H, ~W]
        z = self.__pad_x_for_skip(z, skip)  # [B, C, H, W]

        h = torch.cat([skip, z], dim=1)  # [B, 2C, H, W]

        return self.conv.forward(h, t)


class TimeCondUnet(TimeDependentModule):
    """Time conditioned UNet autoencoder with encoder-wide attention"""

    def __init__(
        self, in_c: int, filters: int = 32, heads: int = 4, t_dims: int = 128
    ) -> None:
        """
        Args:
            in_c (int): input channels
            filters (int, optional): base number of filters. Defaults to 32.
            heads (int, optional): number of heads for base filters. Defaults to 4.
            t_dims (int, optional): time embedding dimensions. Defaults to 128.
        """
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(emb_dim=t_dims)

        # projection conv to c=filters
        self.proj = NConv2D(in_c, filters, t_dims=t_dims, times=2)

        # downsampling
        self.ds1 = TimeDependentSequential(
            Downsample(1 * filters, 2 * filters, t_dims, times=2),
            cnn.ConvMHSA(2 * filters, 2 * heads),
        )
        self.ds2 = TimeDependentSequential(
            Downsample(2 * filters, 4 * filters, t_dims, times=2),
            cnn.ConvMHSA(4 * filters, 4 * heads),
        )
        self.ds3 = TimeDependentSequential(
            Downsample(4 * filters, 8 * filters, t_dims, times=2),
            cnn.ConvMHSA(8 * filters, 8 * heads),
        )

        # bottleneck
        self.bottleneck = NConv2D(8 * filters, 8 * filters, t_dims, times=2)

        # upsample
        self.up3 = Upsample(8 * filters, 4 * filters, t_dims, times=2)
        self.up2 = Upsample(4 * filters, 2 * filters, t_dims, times=2)
        self.up1 = Upsample(2 * filters, 1 * filters, t_dims, times=2)

        # reprojection con to c=in_c
        self.reproj = cnn.TimeConditionedConv(
            1 * filters, in_c, ksize=1, t_dim=t_dims, stride=1, padding=0
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input image size (B, IN_C, H, W)
            t (Tensor): time size (B, 1)

        Returns:
            Tensor: output image pixel-wise velocities for time t size (B, IN_C, H, W)
        """

        # project
        t_emb = self.time_emb.forward(t)
        z0 = self.proj.forward(x, t_emb)

        # downsample
        z1 = self.ds1.forward(z0, t_emb)
        z2 = self.ds2.forward(z1, t_emb)
        z3 = self.ds3.forward(z2, t_emb)

        # bottleneck
        bottleneck = self.bottleneck.forward(z3, t_emb)

        # upsample
        h3 = self.up3.forward(bottleneck, t_emb, z2)
        h2 = self.up2.forward(h3, t_emb, z1)
        h1 = self.up1.forward(h2, t_emb, z0)

        # reproject
        out = self.reproj.forward(h1, t_emb)

        return out


def main():
    x = torch.rand((5, 1, 28, 28))
    t = torch.rand((5, 1))
    t_emb = torch.rand((5, 64))

    nconv = NConv2D(1, 32, 64)
    z0 = nconv(x, t_emb)
    print(z0.shape)

    ds1 = Downsample(32, 64, 64)
    z1 = ds1(z0, t_emb)
    print(z1.shape)

    up1 = Upsample(64, 32, 64)
    h1 = up1.forward(z1, t_emb, z0)
    print(h1.shape)

    unet = TimeCondUnet(in_c=1, t_dims=64)
    y = unet.forward(x, t)
    print(y.shape)


if __name__ == "__main__":
    main()
