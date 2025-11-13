import torch
from torch import Tensor, nn


class VectorField(nn.Module):
    def __init__(self, in_d: int, h_d: int, t_d: int) -> None:
        super().__init__()

        self.in_d = in_d
        self.t_d = t_d

        self.mlp = nn.Sequential(
            nn.Linear(in_d + t_d, h_d),
            nn.SiLU(),
            nn.Linear(h_d, h_d),
            nn.SiLU(),
            nn.Linear(h_d, h_d),
            nn.SiLU(),
            nn.Linear(h_d, in_d),
        )

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        z = torch.cat([xt, t], dim=-1)

        return self.mlp(z)


# make two optimizers and two schedulers, one for each network
class MLP(nn.Module):
    def __init__(self, in_features: int, emb_features: list[int]) -> None:
        super().__init__()

        emb_layers = []
        in_f = in_features
        for features in emb_features:
            emb_layers += [
                nn.Linear(in_f, features),
                nn.BatchNorm1d(num_features=features),
                nn.SiLU(),
            ]

            in_f = features

        self.emb = nn.Sequential(*emb_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.emb(x)
