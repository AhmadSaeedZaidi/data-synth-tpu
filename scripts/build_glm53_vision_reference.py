#!/usr/bin/env python3
"""Regenerate the small independent Transformers vision fixture (CPU only).

Requires torch 2.12.0 and transformers 5.18.0 in a separate environment.
The runtime/tests consume the frozen NPZ and do not import PyTorch.
"""

from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel


def main():
    if transformers.__version__ != "5.18.0":
        raise RuntimeError("Regenerate with the pinned Transformers 5.18.0 reference")
    torch.set_num_threads(2)
    torch.manual_seed(813)
    config = Glm5NextVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=8,
        intermediate_size=64,
        out_hidden_size=32,
        projection_intermediate_size=64,
    )
    config._attn_implementation = "eager"
    model = Glm5NextVisionModel(config).eval()
    patches = torch.randn(32, 3 * 2 * 14 * 14)
    grids = torch.tensor([[1, 4, 6], [1, 2, 4]])
    data = {"weight/" + k: v.detach().numpy() for k, v in model.state_dict().items()}
    data.update(patches=patches.numpy(), grids=grids.numpy())
    with torch.no_grad():
        data["expected"] = model(patches, grid_thw=grids).pooler_output.numpy()
        data["expected_first"] = model(patches[:24], grid_thw=grids[:1]).pooler_output.numpy()
        data["expected_bf16"] = (
            model.to(torch.bfloat16)(patches.to(torch.bfloat16), grid_thw=grids)
            .pooler_output.float()
            .numpy()
        )
    path = Path(__file__).resolve().parents[1] / "kaggle/glm53_offload/data/vision_reference.npz"
    np.savez_compressed(path, **data)
    print(f"Wrote {path}: {path.stat().st_size} bytes; torch={torch.__version__}")


if __name__ == "__main__":
    main()
