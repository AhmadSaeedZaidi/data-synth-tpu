"""Exercise the generated launcher's quota guard without starting a TPU worker."""

import ast
import importlib.util
import json
import subprocess
import time
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def builder():
    path = Path(__file__).resolve().parents[2] / "scripts/build_glm53_offload_notebook.py"
    spec = importlib.util.spec_from_file_location("notebook_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_model_notebook_disables_external_tracking():
    env = {}
    exec(builder()._configuration_cell(False), env)
    assert env["CONFIG"]["wandb_mode"] == "disabled"
    assert env["CONFIG"]["weave_enabled"] is False
    assert env["CONFIG"]["require_live_tracking"] is False


def test_tracking_failure_prevents_model_worker_spawn(tmp_path, monkeypatch):
    from glm53_offload import monitoring, tracking_check

    module = builder()
    environment = {}
    exec(module._configuration_cell(False), environment)
    environment["CONFIG"].update(
        wandb_mode="online", weave_enabled=True, require_live_tracking=True
    )
    finished = []
    monkeypatch.setattr(
        monitoring,
        "RunMonitor",
        lambda *args, **kwargs: type(
            "Monitor",
            (),
            {"state": {"mode": "online"}, "finish": lambda self, code: finished.append(code)},
        )(),
    )

    def fail(*args, **kwargs):
        raise tracking_check.TrackingCheckError("remote records missing")

    monkeypatch.setattr(tracking_check, "require_live_tracking", fail)
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: spawned.append(True))
    environment.update(
        Path=Path,
        tempfile=__import__("tempfile"),
        _runtime_root=tmp_path,
        _runtime_dir=tmp_path,
        _EXPECTED_ARCHIVE_SHA256="fixture",
        _experiment_started_at=time.monotonic(),
    )
    with pytest.raises(RuntimeError, match="before model work"):
        exec(module._run_cell(), environment)
    assert spawned == [] and finished == [1]
    failure = json.loads((environment["RUN_DIR"] / "tracking-readback.json").read_text())
    assert failure == {"status": "FAIL", "error_type": "TrackingCheckError"}


def test_task_key_is_embedded_only_in_private_upload_directory(tmp_path):
    import stat

    module = builder()
    module.REPO_ROOT = tmp_path
    key_file = tmp_path / "task-key"
    key_file.write_text("task-test-key")
    notebook = {"cells": [], "metadata": {"glm53_offload": {}}}
    path = tmp_path / "output/private.ipynb"
    path.parent.mkdir()
    path.touch()
    module.embed_tracking_key(notebook, path, key_file)
    assert "task-test-key" in path.read_text()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert notebook["metadata"]["glm53_offload"]["private_task_authentication"]
    with pytest.raises(ValueError, match="ignored output"):
        module.embed_tracking_key(notebook, tmp_path / "tracked.ipynb", key_file)
    assert not (tmp_path / "tracked.ipynb").exists()


@pytest.mark.parametrize("quant", ["Q6_K_XL", "Q4_K_XL"])
def test_quantization_trials_use_native_reasoning_and_identical_quality_settings(tmp_path, quant):
    module = builder()
    notebook = module.build_notebook(out_path=tmp_path / "trial.ipynb", run_model=True, quant=quant)
    cell = next(cell for cell in notebook["cells"] if "# Human-editable" in "".join(cell["source"]))
    env = {}
    exec("".join(cell["source"]), env)
    config = env["CONFIG"]
    assert config["quant"] == notebook["metadata"]["glm53_offload"]["quant"] == quant
    assert config["thinking_budget"] is None
    assert config["reasoning_effort"] == "max"
    assert config["max_new_tokens"] == 256
    assert config["deadline_seconds"] == 21600
    assert config["batch_sizes"] == [1, 4, 8]
    assert config["prefill_chunk"] == 32
    assert config["prefill_moe_window"] == 256
    assert config["tile_routed_experts"] is True
    assert config["expert_cache_mib_per_device"] == 7168
    assert config["cache_safety_mib_per_device"] == 1024
    assert config["compact_decode_batches"] is True
    assert config["full_decode_warmup"] is True
    assert config["direct_expert_cache"] is True
    assert config["matched_replay_steps"] == 16
    assert config["profile_matched_decode"] is True
    assert config["vision_resolution_control_tokens"] == 1024
    assert config["require_live_tracking"] is True
    assert config["allow_tracking_fallback"] is False


def test_combined_notebook_configures_two_trials_and_one_shared_deadline(tmp_path):
    module = builder()
    notebook = module.build_notebook(
        out_path=tmp_path / "comparison.ipynb", run_model=True, experiment_mode="comparison"
    )
    config_cell = next(
        cell for cell in notebook["cells"] if "# Human-editable" in "".join(cell["source"])
    )
    env = {}
    exec("".join(config_cell["source"]), env)
    assert env["QUANTIZATIONS"] == ["Q4_K_XL", "Q6_K_XL"]
    assert notebook["metadata"]["glm53_offload"]["quantizations"] == env["QUANTIZATIONS"]
    assert env["CONFIG"]["deadline_seconds"] == 21600
    assert env["CONFIG"]["experiment_mode"] == "comparison"
    assert not env["CONFIG"]["long_image_benchmarks"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "combined-notebook", "exec")
    # The generated coordinator passes the original setup start, so dependency
    # time counts against the total budget rather than restarting per model.
    tree = ast.parse(module._comparison_run_cell())
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "run_comparison"
    )
    arguments = {arg.arg: arg.value for arg in call.keywords}
    assert ast.unparse(arguments["session_started_at"]) == "_experiment_started_at"
    assert ast.literal_eval(arguments["launcher_source"]) == module._run_cell()


@pytest.mark.parametrize(
    "run_model,mode",
    [(True, "full"), (True, "expert_profile"), (False, "full"), (True, "comparison")],
)
def test_notebook_fits_kaggle_source_limit_and_bootstrap_preserves_bundle(
    tmp_path, monkeypatch, run_model, mode
):
    import hashlib
    import tempfile

    module = builder()
    notebook = module.build_notebook(
        out_path=tmp_path / "trial.ipynb", run_model=run_model, experiment_mode=mode
    )
    # Normalize the actual upload shape, rather than measuring the on-disk file.
    uploaded = json.loads(json.dumps(notebook))
    for cell in uploaded["cells"]:
        cell["source"] = "".join(cell["source"])
        if cell["cell_type"] == "code":
            cell["outputs"] = []
    assert len(json.dumps(uploaded).encode("utf-8")) < 1_000_000
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    env = {}
    bootstrap = next(cell for cell in uploaded["cells"] if "_PAYLOAD_B85 =" in cell["source"])
    exec(bootstrap["source"], env)
    archive, _ = module._archive_sources()
    assert env["_archive_bytes"] == archive
    assert (
        hashlib.sha256(archive).hexdigest()
        == notebook["metadata"]["glm53_offload"]["bundle_sha256"]
    )


def test_builder_rejects_oversized_source_before_writing(tmp_path, monkeypatch):
    module = builder()
    monkeypatch.setattr(module, "_bootstrap_cell", lambda *args: "#" + "x" * 1_000_000)
    path = tmp_path / "oversized.ipynb"
    with pytest.raises(ValueError, match="must be below 1,000,000 bytes"):
        module.build_notebook(out_path=path, run_model=True)
    assert not path.exists()


@pytest.mark.parametrize("fallback", [True, False])
def test_secret_failure_is_handled_before_dependency_installation(tmp_path, monkeypatch, fallback):
    import sys

    module = builder()
    env = {"_runtime_dir": tmp_path, "sys": sys}
    exec(module._configuration_cell(False), env)
    env["CONFIG"].update(wandb_mode="online", weave_enabled=True, allow_tracking_fallback=fallback)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    secrets = ModuleType("kaggle_secrets")

    class Secrets:
        def get_secret(self, name):
            raise RuntimeError("secret access unavailable")

    secrets.UserSecretsClient = Secrets
    monkeypatch.setitem(sys.modules, "kaggle_secrets", secrets)
    if not fallback:
        with pytest.raises(RuntimeError, match="Online tracking required"):
            exec(module._dependency_cell(), env)
    else:
        exec(module._dependency_cell(), env)
        assert env["CONFIG"]["wandb_mode"] == "offline"
        assert env["CONFIG"]["weave_enabled"] is False
        assert "WANDB_API_KEY" not in env["os"].environ


@pytest.mark.parametrize("timeout", [False, True])
def test_generated_launcher_preserves_evidence_on_success_and_timeout(
    tmp_path, monkeypatch, timeout
):
    module = builder()
    env = {}
    exec(module._configuration_cell(False), env)
    config = env["CONFIG"]
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    actions = []

    class Process:
        pid = 987654

        def __init__(self, command, **kwargs):
            self.returncode = None if timeout else 0
            report_path = Path(command[-1])
            report_path.write_text(
                json.dumps(
                    {
                        "status": "RUNNING" if timeout else "PREFLIGHT_ONLY",
                        "model_check": "NOT_RUN",
                        "validation": {"decoder": "PASS", "vision": "PASS"},
                        "config": config,
                    }
                )
            )
            self.directory = report_path.parent

        def poll(self):
            return self.returncode

        def terminate(self):
            actions.append("terminate")
            self.returncode = -15

        def kill(self):
            actions.append("kill")
            self.returncode = -9

        def wait(self, timeout=None):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("worker", timeout)
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", Process)
    env.update(
        {
            "Path": Path,
            "tempfile": __import__("tempfile"),
            "_runtime_root": tmp_path,
            "_runtime_dir": runtime,
            "_EXPECTED_ARCHIVE_SHA256": "fixture-hash",
            "_experiment_started_at": time.monotonic()
            - (config["deadline_seconds"] + 1 if timeout else 0),
        }
    )
    if timeout:
        with pytest.raises(RuntimeError, match="timed out"):
            exec(module._run_cell(), env)
        assert actions == ["terminate"]
    else:
        exec(module._run_cell(), env)
        assert not actions
    with zipfile.ZipFile(env["EVIDENCE_PATH"]) as archive:
        parent = json.loads(archive.read("supervisor.json"))
        worker = json.loads(archive.read("report.json"))
        assert parent["status"] == ("TIMEOUT" if timeout else "EXITED")
        assert worker["status"] == ("RUNNING" if timeout else "PREFLIGHT_ONLY")
        assert "EVIDENCE_MANIFEST.json" in archive.namelist()


def test_profile_notebook_has_short_deadline_and_distinct_success_status(tmp_path):
    module = builder()
    env = {}
    exec(module._configuration_cell(True, "expert_profile"), env)
    config = env["CONFIG"]
    assert config["deadline_seconds"] == 900
    assert config["max_no_progress_seconds"] == 300
    assert config["profile_experts"] == 96
    assert config["experiment_mode"] == "expert_profile"
    tree = ast.parse(module._run_cell())
    guard = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "experiment_mode" in ast.unparse(node.test)
    )
    for completed in (True, False):
        cases = [{"status": "PASS"}] * (5 if completed else 4)
        check_env = {
            "CONFIG": config,
            "report": {"expert_profile": {"cases": cases}},
            "status": "profile_complete",
            "model_check": "NOT_RUN",
            "has_error": False,
        }
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "profile-guard", "exec"), check_env)
        assert check_env["has_error"] is not completed
    notebook = module.build_notebook(
        out_path=tmp_path / "probe.ipynb", run_model=True, experiment_mode="expert_profile"
    )
    assert notebook["metadata"]["glm53_offload"]["experiment_mode"] == "expert_profile"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook-cell", "exec")
