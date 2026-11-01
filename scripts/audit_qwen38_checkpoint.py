"""Inspect pinned Qwen GGUF headers without downloading the full checkpoint."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kaggle"))
sys.path.insert(0, str(ROOT / "kaggle/vendor"))

from glm53.gguf_reader import GGML_NAMES, GGML_SIZES, GGUFShard  # noqa: E402
from glm53_offload.weights import RangeFile, hf_url  # noqa: E402

REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
REVISION = "766911a6b7369840a91dbcd95f9f997acaab6cd6"
OUT = ROOT / "output/qwen38-strata-research"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant", choices=("UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"), default="UD-Q4_K_XL"
    )
    args = parser.parse_args()
    GGML_SIZES.update({2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24)})
    GGML_NAMES.update({2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1"})
    meta = json.loads((OUT / "unsloth-model-info.json").read_text())
    if meta["sha"] != REVISION:
        raise ValueError("Checkpoint metadata revision differs")
    selected = [
        f
        for f in meta["siblings"]
        if f["rfilename"].startswith(args.quant + "/") or f["rfilename"] == "mmproj-BF16.gguf"
    ]
    tensors, metadata, files = {}, {}, []
    for f in selected:
        name = f["rfilename"]
        size = f.get("size") or f["lfs"]["size"]
        file = RangeFile(hf_url(REPO, REVISION, name), size, chunk_bytes=256 << 10, attempts=3)
        # Vocabulary-only split: no weight tensors. The already audited Q4
        # vocabulary has the same model configuration; do not re-download its
        # 10 MiB tokenizer arrays for each bit width.
        if args.quant != "UD-Q4_K_XL" and "00001-of-" in name:
            prefix = file.pread(0, 32)
            if bytes(prefix[:4]) != b"GGUF" or struct.unpack_from("<Q", prefix, 8)[0] != 0:
                raise ValueError("Expected a vocabulary-only split")
            base = json.loads((OUT / "q4-header-audit.json").read_text())
            metadata.update(base["metadata"])
            files.append(
                dict(
                    file=name,
                    file_bytes=size,
                    tensor_count=0,
                    fetched_bytes=32,
                    metadata_source="q4-header-audit.json",
                )
            )
            print(json.dumps(files[-1]), flush=True)
            continue
        cached = OUT / (Path(name).name + ".header")
        for window in (1, 2, 4, 8, 16, 32):
            prefix = (
                cached.read_bytes() if cached.exists() else file.pread(0, min(size, window << 20))
            )
            view = SimpleNamespace(
                size=len(prefix), pread=lambda offset, length: prefix[offset : offset + length]
            )
            try:
                shard = GGUFShard(view)
                break
            except (struct.error, IndexError, UnicodeDecodeError):
                if window == 32:
                    raise
        metadata.update(shard.kv)
        (OUT / (Path(name).name + ".header")).write_bytes(prefix[: shard.data0])
        for tensor, info in shard.tensors.items():
            if tensor in tensors:
                raise ValueError(f"Duplicate tensor {tensor}")
            tensors[tensor] = dict(info, file=name, data0=shard.data0)
        files.append(
            dict(
                file=name,
                file_bytes=size,
                header_bytes=shard.data0,
                tensor_count=len(shard.tensors),
                fetched_bytes=file.bytes_read,
            )
        )
        print(json.dumps(files[-1]), flush=True)
    categories = Counter()
    types = Counter()
    for name, info in tensors.items():
        category = (
            "ngram"
            if name == "per_layer_token_embd.weight"
            else "experts"
            if "_exps." in name
            else "vision"
            if name.startswith(("v.", "mm."))
            else "other"
        )
        categories[category] += info["nbytes"]
        types[info["type"]] += info["nbytes"]
    report = dict(
        repo=REPO,
        revision=REVISION,
        files=files,
        tensors=tensors,
        metadata=metadata,
        tensor_bytes_by_category=dict(categories),
        tensor_bytes_by_type=dict(types),
    )
    stem = {"UD-Q4_K_XL": "q4", "UD-Q6_K_XL": "q6", "Q8_0": "q8"}[args.quant]
    (OUT / f"{stem}-header-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("GiB by category", {k: v / 2**30 for k, v in categories.items()})
    print("GiB by type", {k: v / 2**30 for k, v in types.items()})


if __name__ == "__main__":
    main()
