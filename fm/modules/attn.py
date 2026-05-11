import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ConvMHSA(nn.Module):
    def __init__(self, in_c: int, heads: int = 4) -> None:
        super().__init__()

        self.heads = heads
        self.in_c = in_c
        self.head_dim = in_c // heads
        self.scale = self.head_dim**-0.5

        # projections
        self.qkv = nn.Conv2d(in_c, in_c * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(in_c, in_c, kernel_size=1)

        # init gain as 1
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        n = h * w

        # QKV
        qkv = self.qkv(x)  # (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)

        # prep for MHSA
        # (B, C, H, W) -> (B, heads, N, head_dim)
        def reshape_for_heads(t: Tensor) -> Tensor:
            return t.view(b, self.heads, self.head_dim, n).transpose(-1, -2)

        q, k, v = map(reshape_for_heads, (q, k, v))  # (B, heads, N, head_dim)

        # scaled matmul
        # (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # weighted sum
        # (B, heads, N, N) @ (B, heads, N, head_dim) -> (B, heads, N, head_dim)
        out = attn @ v

        # reshape back
        # (B, heads, N, head_dim) -> (B, C, H, W)
        out = out.transpose(-1, -2).contiguous().view(b, c, h, w)

        # final projection and residual with gamma
        return x + self.gamma * self.out_proj(out)
