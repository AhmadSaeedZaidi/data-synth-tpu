"""Expose the standalone notebook runtime without shadowing Kaggle's client."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
