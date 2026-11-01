"""Explicit, small Weave spans; model arrays and user content are never serialized."""

import functools
import json
import time
from contextlib import contextmanager
from pathlib import Path

_client = None
_root = None
_path = None
_warned = False
_prefix = "glm53"


def _warning(error):
    global _warned
    if not _warned:
        print("Weave tracing unavailable:", type(error).__name__, flush=True)
        _warned = True


def configure(config, directory):
    global _client, _root, _path, _warned, _prefix
    _client, _root, _path, _warned = None, None, None, False
    _prefix = config.get("trace_prefix", "glm53")
    if config.get("local_traces", False) or config.get("weave_enabled", False):
        _path = Path(directory) / "traces.jsonl"
    if not config.get("weave_enabled", False):
        return
    if config.get("wandb_mode") != "online":
        raise ValueError("Weave requires WANDB_MODE='online'; disable it for offline runs")
    entity = config.get("wandb_entity")
    project = config.get("wandb_project", "glm53-kaggle-tpu")
    try:
        import weave

        _client = weave.init(
            f"{entity}/{project}" if entity else project,
            settings={
                "capture_code": False,
                "implicitly_patch_integrations": False,
                "print_call_link": False,
                "client_parallelism": 2,
                "max_calls_queue_size": 1000,
                "retry_max_attempts": 2,
                "retry_max_interval": 2,
                "http_timeout": 10,
                # Publish start events immediately so a blocked call is visible.
                "use_calls_complete": False,
            },
        )
        monitor_path = Path(directory) / "monitor.json"
        monitor = json.loads(monitor_path.read_text()) if monitor_path.exists() else {}
        _root = _client.create_call(
            op=f"{_prefix}.experiment",
            inputs={key: config[key] for key in ("quant", "batch_sizes") if key in config},
            attributes={
                "run_directory": Path(directory).name,
                "wandb_run_id": monitor.get("wandb_run_id"),
                "runtime_bundle_sha256": config.get("runtime_bundle_sha256"),
            },
        )
    except Exception as error:
        _client, _root = None, None
        if not config.get("allow_tracking_fallback", False):
            raise
        _warning(error)
        (Path(directory) / "tracing-status.json").write_text(
            json.dumps(
                {
                    "mode": "local",
                    "startup_error": type(error).__name__,
                }
            )
            + "\n"
        )
        return
    print(f"Weave traces: https://wandb.ai/{_client.entity}/{project}/weave/calls", flush=True)


@contextmanager
def span(name, **inputs):
    output = {}
    call = None
    started = time.perf_counter()
    failure = None
    if _client is not None:
        try:
            call = _client.create_call(op=f"{_prefix}.{name}", inputs=inputs)
        except Exception as error:
            _warning(error)
    try:
        yield output
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        output["seconds"] = time.perf_counter() - started
        output["status"] = "ERROR" if failure else "PASS"
        record = {"operation": name, "inputs": inputs, "output": output, "error_type": failure}
        if _path is not None:
            with _path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        if call is not None:
            try:
                _client.finish_call(
                    call,
                    output=output,
                    exception=RuntimeError(failure) if failure else None,
                )
            except Exception as error:
                _warning(error)


def operation(name):
    """Capture scalar result metrics only; never introspect arbitrary arguments."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with span(name) as output:
                result = function(*args, **kwargs)
                if isinstance(result, dict):
                    output.update({k: v for k, v in result.items() if isinstance(v, (int, float))})
                return result

        return wrapped

    return decorate


def finish(status):
    global _client, _root, _path
    try:
        if _client is not None:
            _client.finish_call(_root, output={"status": status})
            _client.flush()
    except Exception as error:
        _warning(error)
    finally:
        _client, _root, _path = None, None, None
