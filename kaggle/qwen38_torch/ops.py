"""Packed storage and precision boundaries shared by native Torch experiments.

No JAX imports or host transfers occur in these operators. CPU execution is a
correctness reference, not evidence of TPU lowering or performance.
"""

import torch


def round_bf16_float32(x):
    # Integer ties-to-even rounding also protects a subsequent FP32 consumer.
    bits = x.float().contiguous().view(torch.int32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & -65536
    nan = (bits & 0x7FFFFFFF) > 0x7F800000
    return torch.where(nan, bits | 0x00400000, rounded).view(torch.float32)


def cast(x, dtype):
    if dtype == torch.bfloat16:
        x = round_bf16_float32(x)
    return x.to(dtype)


def fp16_field(raw):
    """Decode two little-endian bytes with FP32 arithmetic, including subnormals.

    Avoid FP16 TPU arithmetic and dtype views on strided packed byte slices.
    """
    bits = raw[..., 0].int() | (raw[..., 1].int() << 8)
    exponent, mantissa = (bits >> 10) & 31, bits & 1023
    value = torch.where(
        exponent == 0,
        mantissa.float() * (2.0**-24),
        (1 + mantissa.float() / 1024) * torch.exp2(exponent.float() - 15),
    )
    value = torch.where(exponent == 31, torch.full_like(value, float("inf")), value)
    value = torch.where((exponent == 31) & (mantissa != 0), float("nan"), value)
    return torch.where((bits & 32768) != 0, -value, value)


def decode_rows(raw, qtype, width, *, dtype=torch.float32):
    """Decode original Q6_K/Q8_0 blocks, without changing scales or codes."""
    if qtype not in {"Q6_K", "Q8_0"}:
        raise ValueError("The first Torch port supports Q6_K and Q8_0 expert storage")
    block, size = (256, 210) if qtype == "Q6_K" else (32, 34)
    if width < 1 or raw.dtype != torch.uint8 or raw.ndim < 2:
        raise ValueError("Expected uint8 [..., blocks, bytes] and positive width")
    blocks = (width + block - 1) // block
    if raw.shape[-2:] != (blocks, size):
        raise ValueError("Packed block count or size differs from the logical row width")
    leading = raw.shape[:-2]
    if qtype == "Q8_0":
        codes = raw[..., 2:].int()
        codes = torch.where(codes >= 128, codes - 256, codes).float()
        values = fp16_field(raw[..., :2])[..., None] * codes
    else:
        low = raw[..., :128].int().reshape(*leading, blocks, 2, 64)
        low = torch.stack((low & 15, low >> 4), dim=-2).reshape(*leading, blocks, 8, 32)
        high = raw[..., 128:192].int().reshape(*leading, blocks, 2, 32)
        shifts = (torch.arange(4, device=raw.device, dtype=torch.int32) * 2)[:, None]
        high = ((high[..., None, :] >> shifts) & 3).reshape(*leading, blocks, 8, 32)
        codes = ((low | (high << 4)) - 32).reshape(*leading, blocks, 16, 16).float()
        scales = raw[..., 192:208].int()
        scales = torch.where(scales >= 128, scales - 256, scales).float()
        values = (fp16_field(raw[..., 208:210])[..., None] * scales)[..., None] * codes
    return cast(values.reshape(*leading, blocks * block)[..., :width], dtype)


def matmul_f32(x, w):
    """BF16 inputs / FP32 accumulator and output on TPU; no silent fallback.

    CPU's dtype-output mm is unavailable in some PyTorch versions. The CPU
    oracle expands already-rounded operands to FP32; this path is never timed.
    """
    right = w.transpose(-1, -2)
    if x.device.type == "cpu" or x.dtype == torch.float32:
        return torch.matmul(x.float(), right.float())
    if x.device.type != "tpu" or x.dtype != torch.bfloat16 or w.dtype != x.dtype:
        raise ValueError("Supported compute is FP32 reference or native TPU BF16")
    if x.ndim == 2:
        return torch.ops.aten.mm.dtype(x, right, torch.float32)
    if x.ndim == 3:
        return torch.ops.aten.bmm.dtype(x, right, torch.float32)
    raise ValueError("Projection expects a matrix or grouped batch of matrices")


def matmul_bf16_x6(x, w):
    """Six BF16 products with FP32 accumulation, for finite FP32 operands.

    Split each operand into three rounded residuals. Retain hh, hm, mh,
    hl, mm, lh; discard ml, lm and ll. Small terms are added first. This is
    an experimental explicit counterpart to TorchTPU's built-in highest
    matmul precision, not a reason to round recurrent state to BF16.
    BF16 overflow/underflow and cancellation preclude a universal relative
    error bound; the comparison worker measures them against FP64 references.
    """
    if x.dtype != torch.float32 or w.dtype != torch.float32:
        raise ValueError("Six-product emulation requires FP32 operands")
    if x.ndim not in (2, 3) or w.ndim != x.ndim or x.shape[-1] != w.shape[-1]:
        raise ValueError("Expected matching matrices or grouped matrices")

    def split(value):
        hi = cast(value, torch.bfloat16)
        residual = value - hi.float()
        mid = cast(residual, torch.bfloat16)
        lo = cast(residual - mid.float(), torch.bfloat16)
        return hi, mid, lo

    xh, xm, xl = split(x)
    wh, wm, wl = split(w)
    small = matmul_f32(xh, wl) + matmul_f32(xm, wm) + matmul_f32(xl, wh)
    middle = matmul_f32(xh, wm) + matmul_f32(xm, wh)
    return matmul_f32(xh, wh) + (middle + small)


def silu(x):
    return cast(x.float() * torch.sigmoid(x.float()), x.dtype)


def quantize_cache(x, group=32):
    if group < 1 or x.shape[-1] % group:
        raise ValueError("Cache groups must divide the head width")
    grouped = x.float().reshape(*x.shape[:-1], -1, group)
    rounded = round_bf16_float32(grouped.abs().amax(dim=-1) / 127)
    divisor = torch.where(rounded == 0, 1, rounded)
    codes = torch.round(grouped / divisor[..., None]).clamp(-127, 127).to(torch.int8)
    return codes.reshape(x.shape), rounded.to(torch.bfloat16)


def gather_scales(plane, indices, groups):
    """Row-wise gather from the unchanged [batch * groups, time] scale plane."""
    if indices.ndim != 2 or plane.ndim != 2 or plane.shape[0] != indices.shape[0] * groups:
        raise ValueError("Scale plane and selected-token shapes differ")
    batch, selected = indices.shape
    addresses = indices[:, None, :].expand(batch, groups, selected)
    values = torch.gather(plane, 1, addresses.reshape(batch * groups, selected).long())
    return values.reshape(batch, groups, selected).transpose(1, 2)


def read_cache(codes, scales, indices, heads, width, group=32, *, dtype=torch.bfloat16):
    """Read valid token addresses only; caller substitutes zero for masked slots."""
    if width % group or codes.ndim != 3 or codes.shape[-1] != heads * width:
        raise ValueError("Invalid packed KV shape")
    batch, selected = indices.shape
    address = indices[..., None].expand(batch, selected, heads * width).long()
    chosen = torch.gather(codes, 1, address)
    scale = gather_scales(scales, indices, heads * (width // group)).float()
    values = chosen.reshape(batch, selected, -1, group).float() * scale[..., None]
    return cast(values.reshape(batch, selected, heads, width), dtype)
