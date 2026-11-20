"""Native Torch still-image ViT, patch merging and spatial RoPE.

Input patches/grids follow the existing processor protocol. Each call encodes
one still image; frame selection and the original text positions stay with the
shared host pipeline. This encoder does not enable a full Torch language model.
"""

import torch
import torch.nn.functional as F

from .ops import cast, matmul_f32


def coordinates(height, width, merge, device):
    if height % merge or width % merge:
        raise ValueError("Image grid must divide into merge groups")
    row, col = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    return torch.stack(
        [
            a.reshape(height // merge, merge, width // merge, merge).permute(0, 2, 1, 3).reshape(-1)
            for a in (row, col)
        ],
        dim=-1,
    )


class Pair(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def affine(self, x):
        return cast(matmul_f32(x, self.weight) + self.bias.float(), x.dtype)

    def norm(self, x):
        xf = x.float()
        mean = xf.mean(dim=-1, keepdim=True)
        normalized = (xf - mean) * torch.rsqrt(((xf - mean) ** 2).mean(dim=-1, keepdim=True) + 1e-6)
        return cast(normalized * self.weight + self.bias, x.dtype)


class Block(torch.nn.Module):
    def __init__(self, weights, c):
        super().__init__()
        self.c = c
        self.parts = torch.nn.ModuleDict({name: Pair(**pair) for name, pair in weights.items()})

    def rope(self, x, pos):
        dim = self.c.hidden // self.c.heads
        inv = self.c.theta ** (
            -torch.arange(0, dim // 2, 2, device=x.device, dtype=torch.float32) / (dim // 2)
        )
        angle = (pos[..., None] * inv).reshape(x.shape[0], dim // 2)
        a, b = x.float().chunk(2, dim=-1)
        cs, sn = angle.cos()[:, None], angle.sin()[:, None]
        return cast(torch.cat((a * cs - b * sn, b * cs + a * sn), dim=-1), x.dtype)

    def forward(self, x, pos):
        c, w = self.c, self.parts
        qkv = w["qkv"].affine(w["ln1"].norm(x)).reshape(len(x), 3, c.heads, c.hidden // c.heads)
        q, k, v = qkv.unbind(dim=1)
        q, k = self.rope(q, pos), self.rope(k, pos)
        score = cast(matmul_f32(q.transpose(0, 1), k.transpose(0, 1)), x.dtype)
        score = cast(score.float() / (c.hidden // c.heads) ** 0.5, x.dtype).float()
        probs = cast(torch.softmax(score, dim=-1), x.dtype)
        attention = matmul_f32(probs, v.permute(1, 2, 0)).transpose(0, 1).reshape(x.shape)
        x = cast(x + w["out"].affine(cast(attention, x.dtype)), x.dtype)
        expanded = w["up"].affine(w["ln2"].norm(x))
        activated = cast(F.gelu(expanded.float(), approximate="tanh"), x.dtype)
        return cast(x + w["down"].affine(activated), x.dtype)


class VisionEncoder(torch.nn.Module):
    def __init__(self, weights, c):
        super().__init__()
        self.c = c
        self.patch = Pair(**weights["patch"])
        self.register_buffer("position", weights["position"])
        self.post_norm = Pair(**weights["post_norm"])
        self.merge_up = Pair(**weights["merge_up"])
        self.merge_down = Pair(**weights["merge_down"])
        self.blocks = torch.nn.ModuleList([Block(w, c) for w in weights["blocks"]])

    def forward(self, patches, grid):
        temporal, height, width = grid
        if temporal != 1:
            raise ValueError("Still images only; temporal segmentation is not implemented")
        c = self.c
        xy = coordinates(height, width, c.merge, patches.device)
        h = xy[:, 0] * (c.grid_side - 1) / max(height - 1, 1)
        w = xy[:, 1] * (c.grid_side - 1) / max(width - 1, 1)
        h0, w0 = h.long(), w.long()
        h1, w1 = (h0 + 1).clamp(max=c.grid_side - 1), (w0 + 1).clamp(max=c.grid_side - 1)
        dh = (h - h0).float()[:, None]
        dw = (w - w0).float()[:, None]

        def table(row, col):
            indices = row * c.grid_side + col
            return self.position[indices]

        pos = (1 - dh) * ((1 - dw) * table(h0, w0) + dw * table(h0, w1)) + dh * (
            (1 - dw) * table(h1, w0) + dw * table(h1, w1)
        )
        x = self.patch.affine(patches) + pos.to(patches.dtype)
        for block in self.blocks:
            x = block(x, xy)
        x = self.post_norm.norm(x).reshape(-1, c.hidden * c.merge**2)
        x = self.merge_up.affine(x)
        return self.merge_down.affine(cast(F.gelu(x.float(), approximate="none"), x.dtype))
