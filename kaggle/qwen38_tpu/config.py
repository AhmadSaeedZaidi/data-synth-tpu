"""Pinned architecture and explicit memory/context constraints."""

from dataclasses import dataclass

MODEL_REPO = "Qwen/Qwen3.8-Flash-Next"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
GGUF_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
GGUF_REVISION = "766911a6b7369840a91dbcd95f9f997acaab6cd6"


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 2560
    layers: int = 48
    streams: int = 4
    lowrank: int = 320
    experts: int = 512
    top_k: int = 10
    intermediate: int = 640
    vocab: int = 248320
    key_heads: int = 16
    value_heads: int = 48
    state_dim: int = 128
    conv_kernel: int = 4
    query_heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 10_000_000.0
    mrope_sections: tuple[int, int, int] = (11, 11, 10)
    qsa_budget: int = 2048
    indexer_heads: int = 4
    indexer_dim: int = 128
    indexer_compress: int = 4
    max_context: int = 2048
    kv_cache_dtype: str = "bf16"
    kv_scale_gather: str = "indexed"
    eps: float = 1e-6
    ple_layer: int = 1
    ple_dim: int = 2560
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_base: int = 20_000_000
    ngram_seed: int = 1234
    eos: int = 248044
    generation_eos: tuple[int, ...] = (248046, 248044)
    image_token: int = 248056
    vision_start: int = 248053
    vision_end: int = 248054

    def __post_init__(self):
        if type(self.max_context) is not int or not 0 < self.max_context <= 32768:
            raise ValueError("Validated context capacity must be in [1, 32768]")
        if self.kv_cache_dtype not in {"bf16", "q8"}:
            raise ValueError("KV cache dtype must be bf16 or q8")
        if self.kv_scale_gather not in {"indexed", "rowwise"}:
            raise ValueError("KV scale gather must be indexed or rowwise")
        if (
            self.indexer_heads < 1
            or self.indexer_dim < self.rotary_dim
            or self.indexer_compress < 2
            or self.qsa_budget < self.indexer_compress
            or self.qsa_budget % self.indexer_compress
        ):
            raise ValueError("Invalid sparse attention indexer dimensions or budget")
        if self.head_dim % self.cache_group:
            raise ValueError("KV head dimension must divide into cache quantization groups")
        if self.value_heads % self.key_heads or self.query_heads % self.kv_heads:
            raise ValueError("Head counts must have integer repetition factors")
        if self.rotary_dim % 2 or self.rotary_dim > self.head_dim:
            raise ValueError("Invalid partial rotary dimension")
        if self.ple_dim % ((self.ngram_size - 1) * self.heads_per_ngram):
            raise ValueError("PLE dimension must divide into n-gram heads")
        if not 0 < self.top_k <= self.experts:
            raise ValueError("Invalid expert top-k")

    def is_qsa(self, layer: int) -> bool:
        return layer % 4 == 3

    @property
    def conv_channels(self):
        return (2 * self.key_heads + self.value_heads) * self.state_dim

    def state_bytes(self, batch: int) -> int:
        return sum(self.state_components(batch).values())

    @property
    def sparse_qsa(self):
        return self.max_context > self.qsa_budget

    @property
    def cache_group(self):
        return min(32, self.head_dim)

    def state_components(self, batch: int) -> dict[str, int]:
        qsa = sum(self.is_qsa(i) for i in range(self.layers))
        gdn = self.layers - qsa
        recurrent = batch * gdn * self.value_heads * self.state_dim**2 * 4
        convolution = batch * gdn * (self.conv_kernel - 1) * self.conv_channels * 2
        values = batch * qsa * self.max_context * self.kv_heads * self.head_dim * 2
        kv = values * (1 if self.kv_cache_dtype == "q8" else 2)
        scales = (values // self.cache_group) * 2 if self.kv_cache_dtype == "q8" else 0
        indexer = 0
        if self.sparse_qsa:
            blocks = (self.max_context + self.indexer_compress - 1) // self.indexer_compress
            indexer = (
                batch * qsa * ((blocks + self.indexer_compress) * self.indexer_dim * 2 + 3 * 4)
            )
        ple = batch * (self.conv_kernel - 1) * self.ngram_size * self.streams * self.hidden * 2
        hidden = batch * self.streams * self.hidden * 2
        return dict(
            recurrent=recurrent,
            convolution=convolution,
            kv=kv,
            kv_scales=scales,
            indexer=indexer,
            ple=ple,
            hidden=hidden,
        )


@dataclass(frozen=True)
class VisionConfig:
    hidden: int = 1152
    heads: int = 16
    intermediate: int = 4304
    layers: int = 27
    patch: int = 16
    temporal: int = 2
    merge: int = 2
    grid_side: int = 48
    output: int = 2560
    theta: float = 10000.0
