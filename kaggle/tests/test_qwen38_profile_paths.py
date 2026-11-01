"""Exercise actual generation bookkeeping with fresh nested profile paths."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import jax.profiler
import numpy as np
import pytest
from qwen38_tpu import run


@pytest.mark.parametrize("label", ["evaluation-first", "batch-8"])
@pytest.mark.parametrize("early_eos", [False, True])
def test_generation_creates_profile_directory_before_metadata_or_skip(
    tmp_path, monkeypatch, label, early_eos
):
    directory = tmp_path / "profiles" / label / "device-trace"
    captured, forwards = [], []

    @contextmanager
    def trace(path, **kwargs):
        # The production optional_device_trace wrapper remains in use; only
        # the backend profiler is replaced so this test needs no TPU.
        context = json.loads((directory.parent / "profile-context.json").read_text())
        assert context["label"] == label and context["decode_step"] == 1
        assert context["active_rows"] == 1 and context["prompt_tokens"] == [2]
        captured.append(path)
        yield

    monkeypatch.setattr(jax.profiler, "trace", trace)
    monkeypatch.setattr(
        run, "checked_argmax", lambda logits: np.array([[99 if early_eos else 7, 0]])
    )
    ready = SimpleNamespace(block_until_ready=lambda: None)
    engine = SimpleNamespace(
        config=SimpleNamespace(generation_eos=(99,)),
        forward=lambda *a: forwards.append(True),
        logits=lambda: ready,
    )
    prompts = [
        dict(
            ids=np.array([1, 2]),
            next_position=2,
            case=dict(name="profile-case", match=dict(kind="exact", expected="OK")),
        )
    ]
    table = SimpleNamespace(gather=lambda ids, **kw: ids, hits=0, misses=0)
    rows, _ = run.generate(
        engine,
        prompts,
        ready,
        None,
        SimpleNamespace(step=lambda ids, history, active: (ids, history)),
        table,
        table,
        SimpleNamespace(decode=lambda *a, **kw: "reasoning</think>OK"),
        3,
        lambda **kw: None,
        record_cases=False,
        profile_directory=directory,
    )
    assert len(rows) == 1
    status = json.loads((directory.parent / "device-trace-status.json").read_text())
    assert status["status"] == ("SKIPPED" if early_eos else "CAPTURED")
    assert len(captured) == (0 if early_eos else 1)
    assert len(forwards) == (0 if early_eos else 2)
    assert not list(directory.parent.glob("*.tmp"))
