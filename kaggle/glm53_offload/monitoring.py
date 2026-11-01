"""Parent-process W&B metrics and live log relay, independent of the JAX worker."""

import json
import os
import time
from pathlib import Path

from .diagnostics import bounded_call


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def read_execution_samples(directory):
    """Read a bounded live timing tail even before the next report checkpoint."""
    path = Path(directory) / "execution-timing.jsonl"
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - (64 << 10)))
            lines = stream.read(64 << 10).splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            item = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


class RunMonitor:
    def __init__(
        self, directory, config, *, mode="disabled", project="glm53-kaggle-tpu", entity=None
    ):
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("W&B mode must be online, offline or disabled")
        if mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Set the Kaggle secret WANDB_API_KEY before an online run")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run = None
        self.started = time.monotonic()
        self.log_offset = 0
        self.last_tick = None
        self.last_download = 0
        self.last_cpu = None
        self.last_label = None
        self.last_print = 0
        self.metrics = {}
        self.progress_age_seconds = 0.0
        self.state = {"mode": mode, "requested_mode": mode, "status": "STARTING"}
        if mode != "disabled":
            import wandb

            # Only explicit settings go to the service, never prompts, paths or credentials.
            safe_config = {
                key: config[key]
                for key in (
                    "quant",
                    "batch_sizes",
                    "max_context",
                    "kv_scale_gather",
                    "quality_first",
                    "max_new_tokens",
                    "prefill_chunk",
                    "prefill_moe_window",
                    "tile_routed_experts",
                    "expert_tile",
                    "expert_cache_mib_per_device",
                    "cache_safety_mib_per_device",
                    "compact_decode_batches",
                    "full_decode_warmup",
                    "direct_expert_cache",
                    "integer_dot_probe",
                    "integer_projection_probe",
                    "matched_replay_steps",
                    "profile_matched_decode",
                    "vision_resolution_control_tokens",
                    "capture_device_trace",
                    "beamng_fixture_sha256",
                    "beamng_image_max_tokens",
                    "vision_max_tokens",
                    "thinking_budget",
                    "reasoning_effort",
                    "comparison_id",
                    "deadline_seconds",
                    "run_model",
                    "max_no_progress_seconds",
                    "weave_enabled",
                    "experiment_mode",
                    "profile_experts",
                    "profile_repeats",
                    "runtime_bundle_sha256",
                    "allow_tracking_fallback",
                    "require_live_tracking",
                    "local_traces",
                )
                if key in config
            }
            init_arguments = dict(
                project=project,
                entity=entity or None,
                mode=mode,
                name=self.directory.name,
                dir=str(self.directory),
                config=safe_config,
                settings=wandb.Settings(
                    init_timeout=30,
                    finish_timeout=20,
                    console="off",
                    disable_code=True,
                    disable_git=True,
                ),
            )
            try:
                self.run = bounded_call(lambda: wandb.init(**init_arguments), timeout=35)
            except Exception as error:
                if not config.get("allow_tracking_fallback", False):
                    raise
                self.state["tracking_error"] = type(error).__name__
                self.state["mode"] = "disabled"
                print(
                    "W&B startup failed; local metrics continue:", type(error).__name__, flush=True
                )
            if self.run is not None and mode == "online":
                self.state["url"] = self.run.url
                self.state["wandb_run_id"] = self.run.id
                print("W&B run:", self.run.url, flush=True)
            elif self.run is not None:
                print("W&B offline: metrics are saved locally; no live dashboard.", flush=True)
        self._save_state()

    def _save_state(self):
        (self.directory / "monitor.json").write_text(json.dumps(self.state, indent=2) + "\n")

    def relay(self):
        """Read only new log bytes. Never wait for a newline or the worker to exit."""
        path = self.directory / "worker.log"
        if not path.exists():
            return
        with path.open("rb") as stream:
            stream.seek(self.log_offset)
            data = stream.read(256 << 10)
            self.log_offset = stream.tell()
        if data:
            print(data.decode("utf-8", errors="replace"), end="", flush=True)

    def tick(self, pid=None, *, force=False):
        self.relay()
        now = time.monotonic()
        if not force and self.last_tick is not None and now - self.last_tick < 15:
            return
        report = read_json(self.directory / "report.json")
        progress = read_json(self.directory / "progress.json")
        stage = progress.get("stage", report.get("stage", "WORKER_STARTING"))
        detail = progress.get("detail", "")
        metrics = {"runtime/elapsed_seconds": now - self.started}
        if progress.get("updated_at"):
            metrics["progress/age_seconds"] = max(0, time.time() - progress["updated_at"])
        self.progress_age_seconds = metrics.get("progress/age_seconds", now - self.started)
        if progress.get("stage_started_at"):
            metrics["progress/stage_seconds"] = max(0, time.time() - progress["stage_started_at"])
        for key, value in progress.items():
            if isinstance(value, (int, float)) and not key.endswith("_at"):
                metrics[f"progress/{key}"] = value
        downloaded = progress.get("download_bytes", 0)
        if self.last_tick is not None and now > self.last_tick:
            metrics["download/bytes_per_second"] = max(0, downloaded - self.last_download) / (
                now - self.last_tick
            )
        if pid is not None:
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    key, _, value = line.partition(":")
                    if key in {"VmRSS", "VmHWM"}:
                        metrics[f"worker/{key}_bytes"] = int(value.split()[0]) * 1024
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                cpu = (int(stat[11]) + int(stat[12])) / os.sysconf("SC_CLK_TCK")
                metrics["worker/cpu_seconds"] = cpu
                if (
                    self.last_cpu is not None
                    and self.last_tick is not None
                    and now > self.last_tick
                ):
                    metrics["worker/cpu_percent"] = (
                        100 * (cpu - self.last_cpu) / (now - self.last_tick)
                    )
                self.last_cpu = cpu
            except (OSError, ValueError, IndexError):
                pass
        for group in ("expert_stats", "expert_cache", "routing_stats"):
            for key, value in report.get(group, {}).items():
                if isinstance(value, (int, float)):
                    metrics[f"{group}/{key}"] = value
        cache = report.get("expert_cache", {})
        total = cache.get("hits", 0) + cache.get("misses", 0)
        if total:
            metrics["expert_cache/hit_fraction_by_count"] = cache.get("hits", 0) / total
        for key in (
            "cache_bytes_per_device",
            "requested_cache_bytes_per_device",
            "cache_safety_bytes_per_device",
        ):
            value = report.get("hbm_plan", {}).get(key)
            if isinstance(value, (int, float)):
                metrics[f"hbm_plan/{key}"] = value
        for item in read_execution_samples(self.directory):
            if any(key not in item for key in ("phase", "tokens", "sample_number")):
                continue
            prefix = (
                f"execution/{item['phase']}/tokens_{item['tokens']}/sample_{item['sample_number']}"
            )
            if item.get("purpose"):
                prefix = (
                    f"execution/{item['purpose']}/layer_{item.get('layer')}/{item['phase']}/"
                    f"tokens_{item['tokens']}/sample_{item['sample_number']}"
                )
            metrics[f"{prefix}/completed"] = int(item.get("status") == "DONE")
            for key in (
                "attention_router_ready_seconds",
                "host_gather_dispatch_seconds",
                "staging_cache_wait_seconds",
                "routing_plan_and_expert_post_seconds",
                "elapsed_seconds",
                "packed_bytes_submitted",
                "program_count_before",
                "program_count_after",
            ):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    metrics[f"{prefix}/{key}"] = value
        for name, result in report.get("validation", {}).items():
            if result in ("PASS", "FAIL"):
                metrics[f"validation/{name}_pass"] = int(result == "PASS")
        quality = report.get("quality", {})
        metrics["quality/asserted_cases"] = quality.get("asserted_cases", 0)
        metrics["quality/failed_cases"] = len(quality.get("failed_cases", []))
        evaluation = report.get("evaluation", {})
        for key in ("expected_cases", "completed_cases"):
            if isinstance(evaluation.get(key), int):
                metrics[f"evaluation/{key}"] = evaluation[key]
        for name, group in evaluation.get("groups", {}).items():
            for key in (
                "completed",
                "asserted",
                "passed",
                "syntax_and_schema_passed",
                "content_passed",
            ):
                if isinstance(group.get(key), int):
                    metrics[f"evaluation/{name}/{key}"] = group[key]
        for item in report.get("benchmarks", []):
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    metrics[f"batch_{item['batch_size']}/{key}"] = value
        for item in report.get("expert_profile", {}).get("cases", []):
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    metrics[f"expert_profile/{item['name']}/{key}"] = value
        for name in ("early_decode_probe", "long_image_benchmarks", "matched_replay_benchmarks"):
            for item in report.get(name, []):
                for key, value in item.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{name}/{item['batch_size']}/{key}"] = value
        for item in report.get("vision_resolution_controls", []):
            prefix = f"vision_resolution_controls/{item['case']}"
            metrics[prefix + "/completed"] = int(item.get("status") in ("PASS", "FAIL"))
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    metrics[f"{prefix}/{key}"] = value
        diagnostic = report.get("matched_decode_diagnostic", {})
        metrics["matched_decode_diagnostic/completed"] = int(
            diagnostic.get("status") in ("MEASURED", "COMPILE_CONTAMINATED")
        )
        for key, value in diagnostic.get("replay", {}).items():
            if isinstance(value, (int, float)):
                metrics[f"matched_decode_diagnostic/{key}"] = value
        for name in ("integer_dot_probe", "integer_projection_probe"):
            for item in report.get(name, {}).get("cases", []):
                dtype = item.get("dtype", item.get("mode"))
                rows = item.get("tokens", item.get("shape_per_chip", [0])[0])
                prefix = f"{name}/{dtype}/rows_{rows}"
                metrics[f"{prefix}/measured"] = int(item.get("status") in ("PASS", "MEASURED"))
                for key, value in item.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{prefix}/{key}"] = value
        # These are explicitly stage snapshots, not live/peak memory measurements.
        snapshot = report.get("memory_by_stage", {}).get(report.get("stage"), {})
        for device in snapshot.get("devices", []):
            for key, value in device.items():
                if key != "id" and isinstance(value, (int, float)):
                    metrics[f"tpu_snapshot/{device['id']}/{key}"] = value
        self.state.update(stage=stage, detail=detail, status=report.get("status", "RUNNING"))
        record = {"time": time.time(), "stage": stage, "detail": detail, **metrics}
        with (self.directory / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.metrics = metrics
        if self.run is not None:
            try:
                self.run.log(metrics)
                self.run.summary.update(self.state)
            except Exception as error:
                self.state["tracking_error"] = type(error).__name__
                print(
                    "W&B logging failed; local metrics continue:", type(error).__name__, flush=True
                )
        label = (stage, detail)
        if force or label != self.last_label or now - self.last_print >= 60:
            age = metrics.get("progress/age_seconds")
            age_text = f"{age:.0f}s" if age is not None else "unknown"
            print(
                f"Monitor: {stage} {detail}; downloaded {downloaded / 2**30:.2f} GiB; "
                f"last worker progress {age_text} ago",
                flush=True,
            )
            self.last_print, self.last_label = now, label
        self.last_tick, self.last_download = now, downloaded
        self._save_state()

    def finish(self, exit_code):
        self.tick(force=True)
        self.state["worker_exit_code"] = exit_code
        if exit_code:
            self.state["status"] = "ERROR"
        self._save_state()
        if self.run is not None:
            try:
                self.run.summary.update(self.state)
                self.run.finish(exit_code=exit_code)
            except Exception as error:
                self.state["tracking_error"] = type(error).__name__
                self._save_state()
                print("W&B finish failed; local metrics remain:", type(error).__name__, flush=True)
