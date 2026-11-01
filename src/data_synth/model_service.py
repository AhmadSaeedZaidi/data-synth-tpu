"""Persist successful invocations without giving providers access to facts."""

import base64
from collections import Counter
from copy import deepcopy
from hashlib import file_digest, sha256
from pathlib import Path

from sqlalchemy.orm import Session

from .models import Asset, ModelRun, Scenario
from .providers import ModelProvider, ModelRequest


def run_model(
    session: Session,
    provider: ModelProvider,
    request: ModelRequest,
    scenario_id: str | None = None,
) -> ModelRun:
    """Validate asset provenance before calling the provider; caller commits.

    Cached responses create separate run records, recording reuse. Failed API
    calls raise a sanitized provider error and do not manufacture a successful run.
    """
    request = ModelRequest.model_validate(deepcopy(request.model_dump()))
    if scenario_id is not None and session.get(Scenario, scenario_id) is None:
        raise ValueError("scenario does not exist")
    if request.input_assets and scenario_id is None:
        raise ValueError("media requests require a scenario")
    embedded_digests = []
    for message in request.messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                try:
                    header, payload = part["image_url"]["url"].split(",", 1)
                    if not header.endswith(";base64"):
                        raise ValueError
                    embedded_digests.append(
                        sha256(base64.b64decode(payload, validate=True)).hexdigest()
                    )
                except Exception:
                    raise ValueError("input image must be a valid base64 data URL") from None
    if Counter(embedded_digests) != Counter(ref["sha256"] for ref in request.input_assets):
        raise ValueError("input asset references must match the exact embedded image bytes")
    for reference in request.input_assets:
        asset = session.get(Asset, reference["id"])
        if asset is None or asset.scenario_id != scenario_id:
            raise ValueError("input asset must belong to the requested scenario")
        if asset.kind not in {"frame", "storyboard"}:
            raise ValueError("only frame and storyboard model inputs are supported")
        if asset.sha256 != reference["sha256"]:
            raise ValueError("input asset checksum does not match the stored asset")
        if asset.local_path:
            with Path(asset.local_path).open("rb") as handle:
                digest = file_digest(handle, "sha256").hexdigest()
            if digest != asset.sha256:
                raise ValueError("input asset content changed; register a new asset")
    result = provider.complete(request)
    run = ModelRun(
        scenario_id=scenario_id,
        provider=request.provider,
        model=request.model,
        endpoint=request.endpoint,
        prompt_version=request.prompt_version,
        parameters=request.parameters,
        messages=request.messages,
        input_assets=request.input_assets,
        request_hash=result.request_hash,
        response=result.response,
        cache_hit=result.cache_hit,
    )
    session.add(run)
    session.flush()
    return run
