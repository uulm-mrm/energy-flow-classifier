from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from flow_matching.modules import ConvMHSA
from flow_matching.modules import (
    TimeDependentSequential,
    TimeDependentModule,
    SinusoidalTimeEmbedding,
)


class TimeDependentLinear(TimeDependentModule):
    def __init__(self, in_features: int, out_features: int, time_dims: int) -> None:
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.film = nn.Linear(time_dims, out_features * 2)

        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.norm(x)

        gamma, beta = self.film(t).chunk(2, dim=-1)
        x = x * (1 + gamma) + beta

        return F.silu(x)


class TimeResConv(TimeDependentModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dims: int,
        attn_heads: Optional[int],
        use_pooling: bool,
    ) -> None:
        super().__init__()

        # residual
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # conv block
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.film = nn.Linear(time_dims, out_channels * 2)  # gamma and beta for FiLM

        # attn
        self.attn = (
            ConvMHSA(out_channels, heads=attn_heads) if attn_heads else nn.Identity()
        )

        # pooling
        self.pool = nn.MaxPool2d(2) if use_pooling else nn.Identity()

        # zero init film params to not mess up training
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    # move this to utils or something with automatic reshape
    def apply_film(self, x: Tensor, t_emb: Tensor, proj: nn.Module) -> Tensor:
        gamma, beta = proj(t_emb).chunk(2, dim=-1)

        gamma = gamma.view(*gamma.shape, 1, 1)
        beta = beta.view(*beta.shape, 1, 1)

        return x * (1 + gamma) + beta

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.conv(x)
        h = self.norm(h)
        h = self.apply_film(h, t, self.film)

        out = F.silu(h + self.res_conv(x))
        out = self.attn(out)

        return self.pool(out)


class TimeMLP(nn.Module):
    def __init__(self, base_channels: int, time_dims: int) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dims),
            nn.SiLU(),
            nn.Linear(time_dims, time_dims),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.time_mlp(t)


class TimeCNN(TimeDependentModule):
    def __init__(self, model_cfg: dict) -> None:
        super().__init__()

        # time embedding, mlp for global time
        self.time_mlp = TimeMLP(**model_cfg["time_mlp"])

        self.conv_backbone = TimeDependentSequential(
            *[TimeResConv(**layer_cfg) for layer_cfg in model_cfg["conv_backbone"]]
        )

        # linear block
        self.linear_backbone = TimeDependentSequential(
            *[
                TimeDependentLinear(**layer_cfg)
                for layer_cfg in model_cfg["linear_backbone"]
            ]
        )

        # energy projection
        last_linear_features = model_cfg["linear_backbone"][-1]["out_features"]
        self.proj = nn.Linear(last_linear_features, 1)

        # init proj with low weights, to not explode E
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # embed time
        t = t.reshape(-1, 1)  # vectorize just in case
        time_emb = self.time_mlp(t)

        # conv bbone
        x = self.conv_backbone(x, time_emb)

        # flatten
        x = torch.flatten(x, 1)

        # linear part
        x = self.linear_backbone(x, time_emb)

        return self.proj(x)


def main():
    import yaml

    with open(r"nuimg/model.cfg.yaml", "r+", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["model"]

    x = torch.rand((10, 256, 7, 7))
    t = torch.rand((10,))

    net = TimeCNN(cfg)

    print(net(x, t).shape)


if __name__ == "__main__":
    main()
