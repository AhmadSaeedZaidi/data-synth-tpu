"""Frozen native vision gate; CPU reference or compiled TorchTPU execution."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def run(device, output):
    import torch
    from qwen38_tpu.config import VisionConfig

    from .comparison import save
    from .vision import VisionEncoder

    report = dict(status="running", device=device, production_enabled=False, cases=[])
    save(output, report)
    if device == "tpu":
        from .runtime import configure_tpu

        report["environment"] = configure_tpu()
    try:
        root = Path(__file__).resolve().parents[1] / "qwen38_tpu/fixtures"
        for name, dtype in [("tiny-reference", torch.float32), ("long-reference", torch.bfloat16)]:
            manifest = json.loads((root / (name + ".json")).read_text())
            path = root / (name + ".npz")
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["fixture_sha256"]:
                raise ValueError("Vision reference checksum differs")
            c = VisionConfig(**manifest["vision_config"])
            with np.load(path, allow_pickle=False) as a:

                def array(key):
                    v = torch.from_numpy(a[key].copy())
                    rounded = v.ndim >= 2 or (key.endswith("bias") and "norm" not in key)
                    return v.to(dtype) if rounded else v

                def pair(prefix):
                    return {k: array(prefix + "." + k) for k in ("weight", "bias")}

                patch = pair("model.visual.patch_embed.proj")
                patch["weight"] = patch["weight"].reshape(c.hidden, -1)
                weights = dict(
                    patch=patch,
                    position=array("model.visual.pos_embed.weight"),
                    post_norm=pair("model.visual.merger.norm"),
                    merge_up=pair("model.visual.merger.linear_fc1"),
                    merge_down=pair("model.visual.merger.linear_fc2"),
                    blocks=[
                        {
                            target: pair(f"model.visual.blocks.{i}.{source}")
                            for target, source in [
                                ("ln1", "norm1"),
                                ("ln2", "norm2"),
                                ("qkv", "attn.qkv"),
                                ("out", "attn.proj"),
                                ("up", "mlp.linear_fc1"),
                                ("down", "mlp.linear_fc2"),
                            ]
                        }
                        for i in range(c.layers)
                    ],
                )
                model = VisionEncoder(weights, c).to(device).eval()
                function = torch.compile(
                    model,
                    backend="tpu" if device == "tpu" else "eager",
                    fullgraph=True,
                    dynamic=False,
                )
                pixels = torch.from_numpy(a["pixels"].copy()).to(dtype).to(device)
                actual = []
                offset = 0
                started = time.monotonic()
                with torch.inference_mode():
                    for grid in a["grids"]:
                        count = int(np.prod(grid))
                        actual.append(
                            function(pixels[offset : offset + count], tuple(int(v) for v in grid))
                            .float()
                            .cpu()
                            .numpy()
                        )
                        offset += count
                result = np.concatenate(actual)
                reference = a["bf16_vision" if dtype == torch.bfloat16 else "vision"]
                np.testing.assert_allclose(
                    result,
                    reference,
                    atol=0.008 if dtype == torch.bfloat16 else 2e-6,
                    rtol=0.01 if dtype == torch.bfloat16 else 2e-5,
                )
                report["cases"].append(
                    dict(
                        fixture=name,
                        dtype=str(dtype),
                        max_error=float(np.max(np.abs(result - reference))),
                        status="passed",
                        compile_and_execution_seconds=time.monotonic() - started,
                    )
                )
                save(output, report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error)[:1500])
    save(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "tpu"], default="cpu")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(0 if run(args.device, args.output)["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
