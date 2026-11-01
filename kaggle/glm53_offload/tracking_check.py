"""Verify tracking by reading acknowledged records from the remote services."""

import os
import time

import requests

from .diagnostics import bounded_call


class TrackingCheckError(RuntimeError):
    pass


def load_kaggle_key(timeout=50):
    """Retrieve credentials without ever including their value in diagnostics."""
    if os.environ.get("WANDB_API_KEY", "").strip():
        return {"source": "environment", "status": "PASS"}
    try:
        from kaggle_secrets import UserSecretsClient

        value = bounded_call(lambda: UserSecretsClient().get_secret("WANDB_API_KEY"), timeout)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty secret")
        os.environ["WANDB_API_KEY"] = value.strip()
    except Exception as error:
        # Kaggle BackendError messages can contain response bodies. Preserve only
        # the class and a classification against known service error messages.
        message = str(error).lower()
        reason = "lookup_failed"
        if "no user secrets exist" in message or "not attached" in message:
            reason = "secret_not_attached"
        elif "label" in message and ("not found" in message or "does not exist" in message):
            reason = "label_not_found"
        elif "jwt" in message or type(error).__name__ == "CredentialError":
            reason = "session_credentials_unavailable"
        elif isinstance(error, TimeoutError) or "timeout" in message:
            reason = "lookup_timeout"
        raise TrackingCheckError(
            f"Kaggle WANDB_API_KEY lookup failed: {type(error).__name__} ({reason})"
        ) from None
    return {"source": "kaggle_secret", "status": "PASS"}


def _post(url, body):
    key = os.environ.get("WANDB_API_KEY", "")
    if not key:
        raise TrackingCheckError("WANDB_API_KEY is unavailable")
    response = requests.post(url, json=body, auth=("api", key), timeout=(5, 10))
    if response.status_code != 200:
        raise TrackingCheckError(f"Tracking readback HTTP {response.status_code}")
    return response.json()


def read_weave_calls(project_id, trace_id):
    return _post(
        "https://trace.wandb.ai/calls/query",
        {
            "project_id": project_id,
            "filter": {"trace_ids": [trace_id]},
            "limit": 100,
            "columns": [
                "id",
                "op_name",
                "parent_id",
                "started_at",
                "ended_at",
                "exception",
                "output",
            ],
        },
    ).get("calls", [])


def read_wandb_summary(entity, project, run_id):
    import json

    data = _post(
        "https://api.wandb.ai/graphql",
        {
            "query": (
                "query($entity:String!,$project:String!,$run:String!) { "
                "project(name:$project,entityName:$entity) { "
                "run(name:$run) { name state summaryMetrics } } }"
            ),
            "variables": {"entity": entity, "project": project, "run": run_id},
        },
    )
    if data.get("errors"):
        raise TrackingCheckError("W&B summary readback returned GraphQL errors")
    project_data = (data.get("data") or {}).get("project") or {}
    run = project_data.get("run") or {}
    summary = run.get("summaryMetrics") or "{}"
    return {"state": run.get("state"), "summary": json.loads(summary)}


def operation_name(call):
    return call.get("op_name", "").split("/op/")[-1].split(":")[0]


def wait_for_remote(read, predicate, timeout=45):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            value = read()
            if predicate(value):
                return value
        except Exception as error:
            last_error = type(error).__name__
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    detail = f" ({last_error})" if last_error else ""
    raise TrackingCheckError("Remote tracking records were not acknowledged" + detail)


def require_live_tracking(monitor, config, directory):
    """Check an unfinished Weave operation and a W&B heartbeat before model work."""
    from pathlib import Path

    from . import tracing
    from .diagnostics import atomic_json

    if monitor.state.get("mode") != "online" or monitor.run is None:
        raise TrackingCheckError("Online W&B is required before model work")
    if not config.get("weave_enabled"):
        raise TrackingCheckError("Online Weave is required before model work")
    tracing.configure({**config, "allow_tracking_fallback": False}, directory)
    try:
        client, root = tracing._client, tracing._root
        project_id = f"{client.entity}/{config['wandb_project']}"
        marker = str(time.time_ns())
        monitor.run.summary["tracking/readback_marker"] = marker
        monitor.run.log({"tracking/readback_marker": marker})
        with tracing.span("tracking_ready", marker=marker):
            bounded_call(client.flush, timeout=15)
            calls = wait_for_remote(
                lambda: read_weave_calls(project_id, root.trace_id),
                lambda rows: any(
                    operation_name(c) == f"{config.get('trace_prefix', 'glm53')}.tracking_ready"
                    and not c.get("ended_at")
                    for c in rows
                ),
            )
            wait_for_remote(
                lambda: read_wandb_summary(client.entity, config["wandb_project"], monitor.run.id),
                lambda data: data["summary"].get("tracking/readback_marker") == marker,
            )
        proof = {
            "status": "PASS",
            "verified_at": time.time(),
            "wandb_run_id": monitor.run.id,
            "weave_trace_id": root.trace_id,
            "open_weave_span_observed": True,
            "wandb_heartbeat_observed": True,
            "weave_call_count": len(calls),
            "wandb_url": monitor.run.url,
            "weave_url": f"https://wandb.ai/{project_id}/weave/calls",
        }
        atomic_json(Path(directory) / "tracking-readback.json", proof)
        return proof
    finally:
        tracing.finish("TRACKING_CHECK_COMPLETE")
