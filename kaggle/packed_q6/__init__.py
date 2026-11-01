"""Experimental, model-independent GGUF Q6_K projection for Pallas TPU.

Optional notebook preflight diagnostic; not used by Qwen model inference.
"""

from .layout import PackedQ6, pack_q6, unpack_q6

__all__ = ["PackedQ6", "pack_q6", "unpack_q6"]
