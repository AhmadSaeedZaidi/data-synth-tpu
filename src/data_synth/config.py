"""Explicit configuration loading; importing this module performs no I/O."""

import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_url: str = Field(default="sqlite:///data/research.db", repr=False, exclude=True)
    cache_dir: Path = Path(".cache/model-responses")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "z-ai/glm-5.3"
    annotator_model: str = "moonshotai/kimi-k3"
    reviewer_model: str = "z-ai/glm-5.3"
    model_max_tokens: int = Field(default=2048, ge=256, le=32768)
    nvidia_api_key: SecretStr | None = Field(default=None, exclude=True)
    hf_token: SecretStr | None = Field(default=None, exclude=True)
    hf_dataset_repo: str | None = None
    hf_bucket: str | None = None
    blob_dir: Path = Path("data/blobs")
    beamng_home: str | None = None
    beamng_edition: Literal["drive", "tech"] = "drive"
    beamng_host: str = "127.0.0.1"
    beamng_port: int = Field(default=25252, ge=1, le=65535)
    beamng_timeout: float = Field(default=5.0, gt=0, le=30)

    @classmethod
    def load(cls, env_file: Path | None = Path(".env")) -> "Settings":
        file_values = dotenv_values(env_file, interpolate=False) if env_file else {}
        values = {**file_values, **os.environ}

        # Aliases must respect source precedence, even when casing differs.
        def secret_alias(*names: str) -> str | None:
            return next(
                (
                    source[name]
                    for source in (os.environ, file_values)
                    for name in names
                    if source.get(name)
                ),
                None,
            )

        return cls(
            database_url=values.get("DATABASE_URL") or "sqlite:///data/research.db",
            cache_dir=values.get("CACHE_DIR") or ".cache/model-responses",
            nvidia_base_url=values.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1",
            nvidia_model=values.get("NVIDIA_MODEL") or "z-ai/glm-5.3",
            annotator_model=values.get("ANNOTATOR_MODEL") or "moonshotai/kimi-k3",
            reviewer_model=values.get("REVIEWER_MODEL") or "z-ai/glm-5.3",
            model_max_tokens=values.get("MODEL_MAX_TOKENS") or 2048,
            nvidia_api_key=secret_alias("nvidia_api_key", "NVIDIA_API_KEY"),
            hf_token=secret_alias("HF_TOKEN", "hf_token"),
            hf_dataset_repo=values.get("HF_DATASET_REPO") or None,
            hf_bucket=values.get("HF_BUCKET") or None,
            blob_dir=values.get("BLOB_DIR") or "data/blobs",
            beamng_home=values.get("BEAMNG_HOME") or None,
            beamng_edition=values.get("BEAMNG_EDITION") or "drive",
            beamng_host=values.get("BEAMNG_HOST") or "127.0.0.1",
            beamng_port=values.get("BEAMNG_PORT") or 25252,
            beamng_timeout=values.get("BEAMNG_TIMEOUT") or 5.0,
        )
