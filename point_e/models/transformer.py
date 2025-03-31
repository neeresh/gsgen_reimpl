import math
from typing import List, Tuple

import torch
from torch import nn

from point_e.models.checkpoint import checkpoint
from point_e.models.util import timestep_embedding


def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, width: int, init_scale: float):
        super().__init__()
        self.width = width

        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)

        self.gelu = nn.GELU()

        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)

        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)  # More stable with f16 than dividing afterwards

        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)

        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class MultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, n_ctx: int, width: int,
                 heads: int, init_scale: float, ):
        super().__init__()

        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads

        self.c_qkv = nn.Linear(width, width * 3, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)

        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx)

        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)

        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, n_ctx: int, width: int,
                 heads: int, init_scale: float = 1.0, ):
        super().__init__()

        self.attn = MultiheadAttention(device=device, dtype=dtype, n_ctx=n_ctx, width=width,
                                       heads=heads, init_scale=init_scale, )

        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class Transformer(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, n_ctx: int, width: int, layers: int,
                 heads: int, init_scale: float = 0.25):
        super().__init__()

        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers

        init_scale = init_scale * math.sqrt(1.0 / width)

        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(device=device, dtype=dtype, n_ctx=n_ctx, width=width, heads=heads,
                                    init_scale=init_scale, ) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)

        return x


class PointDiffusionTransformer(nn.Module):

    """
    The PointDiffusionTransformer is a transformer-based model used in the Point-E project.
    The goal is to generate 3D point clouds via diffusion models.
    It takes a tensor representing a sequence of point features and
    predicts another tensor of the same shape — essentially learning how points in a cloud evolve over time (timesteps)
    """

    def __init__(self, *, device: torch.device, dtype: torch.dtype, input_channels: int = 3,
                 output_channels: int = 3, n_ctx: int = 1024, width: int = 512, layers: int = 12, heads: int = 8,
                 init_scale: float = 0.25, time_token_cond: bool = False):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx

        self.time_token_cond = time_token_cond
        self.time_embed = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale * math.sqrt(1.0 / width))

        self.ln_pre = nn.LayerNorm(width, device=device, dtype=dtype)
        self.backbone = Transformer(device=device, dtype=dtype, width=width, layers=layers, heads=heads,
                                    init_scale=init_scale, n_ctx=n_ctx + int(time_token_cond))
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)

        self.input_proj = nn.Linear(input_channels, width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, output_channels, device=device, dtype=dtype)

        with torch.no_grad():
            self.output_proj.weight.zero_()
            self.output_proj.bias.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        :param x: an [N x C x T] tensor.
            N: batch size
            C: number of input channels (e.g., 3 for XYZ coordinates)
            T: number of tokens (or points in the point cloud)
            Example: For a batch of 4 point clouds, each with 3D coordinates and 1024 points: x.shape = (4, 3, 1024)

        :param t: an [N] tensor.
        A scalar timestep for each sample in the batch, representing the current point in the diffusion process.

        :return: an [N x C' x T] tensor.

        x.shape = [4, 3, 1024]
        t.shape = [4]

        -> t_embed: [4, 512]
        -> input_proj(x): [4, 1024, 512]
        -> +t_embed: [4, 1024, 512]
        -> backbone: [4, 1024, 512]
        -> output_proj: [4, 1024, 3]
        -> permute: [4, 3, 1024]
        """
        assert x.shape[-1] == self.n_ctx

        t_embed = self.time_embed(timestep_embedding(t, self.backbone.width))

        # self.time_token_cond is a boolean:
        # If True, the time embedding is added as an extra token;
        # If False, it is added directly to each input token via broadcasting.

        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(self, x: torch.Tensor, cond_as_token: List[Tuple[torch.Tensor, bool]]) -> torch.Tensor:

        # Transposes x from [N, C, T] to [N, T, C], then projects each token (point) from C to width dimensions.
        h = self.input_proj(x.permute(0, 2, 1))  # NCL -> NLC

        # h.shape is now [N, T, width].

        for emb, as_token in cond_as_token:
            # Depending on as_token:
            # If False: Adds t_embed to each token → [N, T, width]
            # If True: Prepends it as a token → [N, T+1, width]
            if not as_token:
                h = h + emb[:, None]

        extra_tokens = [
            (emb[:, None] if len(emb.shape) == 2 else emb)
            for emb, as_token in cond_as_token
            if as_token
        ]

        if len(extra_tokens):
            h = torch.cat(extra_tokens + [h], dim=1)

        # [N, T, width]
        h = self.ln_pre(h)

        # [N, T, width]
        h = self.backbone(h)

        # [N, T, width]
        h = self.ln_post(h)

        # If a time token was added as a prepended token, it's now removed before projecting the output.
        if len(extra_tokens):
            h = h[:, sum(h.shape[1] for h in extra_tokens):]

        # [N, T, output_channels]
        h = self.output_proj(h)

        # [N, output_channels, T]
        return h.permute(0, 2, 1)


class CLIPImagePointDiffusionTransformer(PointDiffusionTransformer):
    pass
