import torch
from torch import nn, Tensor
import torch.nn.functional as F

from flow_matching.modules import cnn
from flow_matching.modules.utils import (
    TimeDependentSequential,
    TimeDependentModule,
    SinusoidalTimeEmbedding,
)


class ResConvBlock(TimeDependentModule):
    def __init__(self, in_c: int, out_c: int, t_dims: int) -> None:
        super().__init__()

        # 1x1 conv for residual
        self.res_conv = cnn.TimeConditionedConv(
            in_c, out_c, ksize=1, t_dim=t_dims, stride=1, padding=0
        )

        self.conv = cnn.TimeConditionedConv(
            in_c, out_c, ksize=3, t_dim=t_dims, stride=1, padding=1
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = F.relu(self.conv(x, t)) + self.res_conv(x, t)
        z = F.relu(h)

        return F.max_pool2d(z, kernel_size=2)


class TimeResCNN(TimeDependentModule):
    def __init__(
        self, in_c: int, t_dims: int, res_blocks: int, linear_layers: int
    ) -> None:
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(emb_dim=t_dims)

        _res_blocks = []  # no attention
        _linear = []  # no time in linear

        in_chans = in_c
        for _ in range(res_blocks):
            out_chans = 2 * in_chans

            _res_blocks.append(
                ResConvBlock(in_c=in_chans, out_c=out_chans, t_dims=t_dims)
            )

            in_chans = out_chans

        in_neu = in_chans
        for _ in range(linear_layers):
            out_neu = in_neu // 2

            # no bias bcs BN
            _linear.append(nn.Linear(in_neu, out_neu, bias=False))
            _linear.append(nn.BatchNorm1d(out_neu))
            _linear.append(nn.SiLU(inplace=True))

            in_neu = out_neu

        self.res_blocks = TimeDependentSequential(*_res_blocks)
        self.linear = nn.Sequential(*_linear)

        self.project = nn.Linear(in_neu, 1)  # project to scalar

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = t.reshape(-1, 1)

        t_emb = self.time_emb(t)

        x = self.res_blocks(x, t_emb)  # (B, C', H, W)
        x = torch.flatten(x, start_dim=1)  # (B, C'*H*W)

        x = self.linear(x)  # (B, D')

        return self.project(x)  # (B, 1)
