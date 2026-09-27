import math

import torch
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth: zero a residual branch for a random subset of samples.
    Kept samples are scaled by 1 / (1 - drop_prob) to preserve the mean at eval time.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    # One Bernoulli draw per sample, broadcast over every other axis, divide by keep
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the embedding dimension by the angles in rope."""
    sin, cos = rope.chunk(2, dim=-1)
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class RotaryEmbedding(torch.nn.Module):
    """Return sine and cosine for each patch to rotate queries and keys with."""

    def __init__(
        self, dim: int, temperature: float = 100.0, rescale_coords: float = 2.0
    ) -> None:
        super().__init__()
        self.rescale_coords = rescale_coords
        exponents = 2.0 * torch.arange(dim // 4) / (dim // 2)
        # Distance on [-1, 1] to complete a period, shared by x and y axes
        self.register_buffer("periods", temperature**exponents, persistent=False)

    def forward(self, grid: tuple[int, int]) -> torch.Tensor:
        height, width = grid
        device, dtype = self.periods.device, self.periods.dtype
        # Patch midpoints, normalized on x and y, then shifted to [-1, 1] like in DINOv3
        coords_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        coords_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        coords_y, coords_x = 2 * coords_y - 1, 2 * coords_x - 1
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
        coords = torch.stack((grid_y, grid_x), dim=-1).flatten(0, 1)  # (patches, 2)
        if self.training and self.rescale_coords != 0.0:
            # RoPE jittering: one random scale for the whole grid, between 1/R and R
            bound = math.log(self.rescale_coords)
            coords = coords * coords.new_empty(1).uniform_(-bound, bound).exp()
        # Repeat each angle across both halves of the head
        angles = (2 * torch.pi * coords.unsqueeze(-1) / self.periods).flatten(1).tile(2)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)  # (patches, 2 * dim)


class PatchEmbed(torch.nn.Module):
    """Split the image into patches and embed each one."""

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (batch, tokens, embed_dim)


class Mlp(torch.nn.Module):
    """Position-wise feed-forward: widen dimension, apply nonlinearity, project back."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(torch.nn.Module):
    """Multi-head self attention over the token axis."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = torch.nn.Linear(dim, 3 * dim)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        # Split the fused projection into q, k, v, each (batch, heads, tokens, head_dim)
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # CLS token carries no positional information, so only the patches are rotated
        q = torch.cat((q[:, :, :1], apply_rope(q[:, :, 1:], rope)), dim=2)
        k = torch.cat((k[:, :, :1], apply_rope(k[:, :, 1:], rope)), dim=2)
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attn.transpose(1, 2).reshape(batch, tokens, dim))


class Block(torch.nn.Module):
    """Pre-norm transformer block: attention then MLP, each added as a residual."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        self.drop_prob = drop_path
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        prob, training = self.drop_prob, self.training
        x = x + drop_path(self.attn(self.norm1(x), rope), prob, training)
        return x + drop_path(self.mlp(self.norm2(x)), prob, training)


class ViT(torch.nn.Module):
    """Vision transformer with a CLS token. Default params match a ViT tiny with RoPE.
    forward gives the CLS embedding, forward_features all tokens (image tokens + CLS).
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        drop_path_rate: float = 0.1,
        rope_rescale_coords: float = 2.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.rope = RotaryEmbedding(
            embed_dim // num_heads, rescale_coords=rope_rescale_coords
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.cls_token, std=1e-6)
        # Drop probability ramps linearly with depth, so early blocks stay reliable
        drop_probs = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.blocks = torch.nn.ModuleList(
            Block(embed_dim, num_heads, drop_path=prob) for prob in drop_probs
        )
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected (batch, channels, height, width), got {x.shape}"
        stride_h, stride_w = self.patch_embed.proj.stride
        rope = self.rope((x.shape[-2] // stride_h, x.shape[-1] // stride_w))
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x)  # (batch, 1 + patches, embed_dim), CLS at index 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[:, 0]  # (batch, embed_dim), CLS only
