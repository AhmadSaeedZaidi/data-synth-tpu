"""Regression coverage for active experts and phase-specific execution policy."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from qwen38_tpu.compute import pack_q8
from qwen38_tpu.expert_calibration import CANDIDATES, select_candidate, select_phase_candidates
from qwen38_tpu.experts import expert_parallel
from qwen38_tpu.trial import evaluation_sanity
from qwen38_tpu.validation import validate_long_context


def measurements(*, small_groups=False):
    return [
        dict(
            candidate=c["name"],
            quant=q,
            rows=b,
            routing=r,
            accuracy_pass=True,
            temporary_bytes=1024,
            seconds=1.0,
        )
        for c in CANDIDATES
        if small_groups or c["name"] in {"serial_ep8", "grouped_ep8", "grouped_ep4_tp2"}
        for q in ("mixed_q6_q8", "q8")
        for b in (1, 8, 128, 1024)
        for r in ("spread", "hot")
    ]


def test_smaller_groups_can_share_tp2_placement_across_phases():
    rows = measurements(small_groups=True)
    for r in rows:
        if r["candidate"] == "sliced_ep4_tp2":
            r["seconds"] = 0.5 if r["rows"] <= 8 else 1.2
        if r["candidate"] == "paired_ep4_tp2":
            r["seconds"] = 0.8
    selected = select_phase_candidates(rows, [1, 8])["selected"]
    assert selected["tensor_parallel"] == 2
    assert selected["group_size"] == 1 and selected["prefill_group_size"] == 2
    # Corrupting the new fast path must remove it from selection.
    for r in rows:
        if r["candidate"] == "sliced_ep4_tp2":
            r["accuracy_pass"] = False
    assert select_phase_candidates(rows, [1, 8])["selected"]["group_size"] == 2


def test_fast_decode_survives_slow_prefill_without_redistributing_weights():
    rows = measurements()
    for r in rows:
        if r["candidate"] == "grouped_ep8":
            r["seconds"] = 0.7 if r["rows"] <= 8 else 1.5
        if r["candidate"] == "grouped_ep4_tp2":
            r["seconds"] = 0.5 if r["rows"] <= 8 else 1.3
    # A single policy rejects both optimizations; phase selection can use EP8
    # grouped decode and the proven serial prefill on the SAME packed arrays.
    assert select_candidate(rows)["selected"]["name"] == "serial_ep8"
    chosen = select_phase_candidates(rows, [1, 8])["selected"]
    assert chosen["tensor_parallel"] == 1
    assert chosen["group_size"] == 4 and chosen["prefill_group_size"] == 1


def test_tp2_requires_both_phases_to_pass():
    rows = measurements()
    for r in rows:
        if r["candidate"] == "grouped_ep4_tp2":
            r["seconds"] = 0.6
    chosen = select_phase_candidates(rows, [1, 8])["selected"]
    assert chosen["tensor_parallel"] == 2
    assert chosen["group_size"] == chosen["prefill_group_size"] == 4
    rows[-1]["accuracy_pass"] = False
    assert select_phase_candidates(rows, [1, 8])["selected"]["tensor_parallel"] == 1
    assert select_phase_candidates(rows[:-1], [1, 8])["selected"]["tensor_parallel"] == 1


@pytest.mark.parametrize("prefill_seconds", [1.0, 1.1])
def test_tp2_decode_gain_can_pay_for_safe_neutral_or_slightly_slower_prefill(prefill_seconds):
    rows = measurements()
    for r in rows:
        if r["candidate"] == "grouped_ep4_tp2":
            r["seconds"] = 0.6 if r["rows"] <= 8 else prefill_seconds
    assert select_phase_candidates(rows, [1, 8])["selected"]["tensor_parallel"] == 2
    rows[-1]["seconds"] = 1.16
    assert select_phase_candidates(rows, [1, 8])["selected"]["tensor_parallel"] == 1


def test_small_combined_gain_keeps_the_safe_baseline():
    rows = measurements()
    for r in rows:
        if r["candidate"] != "serial_ep8":
            r["seconds"] = 0.96
    selected = select_phase_candidates(rows, [1, 8])["selected"]
    assert selected["group_size"] == selected["prefill_group_size"] == 1
    assert selected["tensor_parallel"] == 1


def test_unsafe_baseline_is_never_used_as_fallback():
    rows = measurements()
    for r in rows:
        if r["candidate"] == "serial_ep8":
            r["temporary_bytes"] = 505 << 20  # v9's reported decode allocation
    with pytest.raises(ValueError, match="memory limit"):
        select_candidate(rows)
    with pytest.raises(ValueError, match="safe expert placement"):
        select_phase_candidates(rows, [1, 8])
    for r in rows:
        if r["candidate"] == "grouped_ep4_tp2":
            r["seconds"] = 0.6
    assert select_phase_candidates(rows, [1, 8])["selected"]["tensor_parallel"] == 2


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_timings_cannot_win(seconds):
    rows = measurements()
    for r in rows:
        if r["candidate"] == "grouped_ep8":
            r["seconds"] = seconds
    assert select_candidate(rows)["selected"]["name"] == "serial_ep8"
    rows[0]["seconds"] = seconds
    with pytest.raises(ValueError, match="valid timings"):
        select_candidate(rows)


@pytest.mark.parametrize("policy", ["bf16", "int8_q8"])
def test_serial_does_not_evaluate_unrouted_experts_even_on_idle_owners(policy):
    mesh = Mesh(np.asarray(jax.devices()), ("expert",))
    count, width = 16, 32
    # A poisoned unused expert exposes work done before applying a zero route
    # mask. This failed under v9's unconditional padded 16-expert head.
    raw = pack_q8(np.full((count * width, width), 0.02, np.float32))
    raw = raw.reshape(count, width, 1, 34)
    raw[..., :2] = np.array([np.nan], "<f2").view(np.uint8)
    raw[3, ..., :2] = np.array([0.02 / 127], "<f2").view(np.uint8)
    packed = jax.device_put(raw.reshape(count, -1), NamedSharding(mesh, P("expert")))
    replica = NamedSharding(mesh, P())
    inputs = tuple(
        jax.device_put(x, replica)
        for x in (
            np.full((8, width), 0.2, np.dtype(jnp.bfloat16)),
            np.full((8, 1), 3, np.int32),
            np.ones((8, 1), np.dtype(jnp.bfloat16)),
        )
    )
    fn = expert_parallel(
        mesh,
        gate_type="Q8_0",
        up_type="Q8_0",
        down_type="Q8_0",
        hidden=width,
        intermediate=width,
        compute=policy,
    )
    active = np.asarray(fn(packed, packed, packed, *inputs), np.float32)
    assert np.isfinite(active).all() and np.any(active != 0)
    inactive = fn(packed, packed, packed, *inputs[:2], jnp.zeros_like(inputs[2]))
    np.testing.assert_array_equal(np.asarray(inactive, np.float32), np.zeros((8, width)))


@pytest.mark.parametrize(
    "tp,fused,dtype,decode_group,prefill_group",
    [
        (1, True, jnp.float32, 4, 1),
        (1, False, jnp.float32, 4, 1),
        (2, True, jnp.float32, 4, 1),
        (1, True, jnp.bfloat16, 4, 1),
        (2, True, jnp.float32, 1, 2),
        (2, True, jnp.bfloat16, 1, 2),
        (2, True, jnp.float32, 2, 1),
        (2, True, jnp.bfloat16, 2, 1),
    ],
)
def test_phase_specific_experts_match_independent_full_model_reference(
    tp, fused, dtype, decode_group, prefill_group
):
    result = validate_long_context(
        Mesh(np.asarray(jax.devices()), ("expert",)),
        expert_group_size=decode_group,
        prefill_expert_group_size=prefill_group,
        expert_tensor_parallel=tp,
        fuse_layers=fused,
        dtype=dtype,
    )
    assert result["status"] == "PASS"


def test_optional_chunk_prefill_keeps_full_model_reference_gate():
    result = validate_long_context(
        Mesh(np.asarray(jax.devices()), ("expert",)), gdn_prefill="chunked"
    )
    assert result["status"] == "PASS" and result["gdn_prefill"] == "chunked"


def test_first_cohort_guard_requires_a_complete_thinking_collapse():
    rows = [dict(asserted=True, truncated=True, answer_phase="THINKING") for _ in range(8)]
    assert evaluation_sanity(rows)["status"] == "FAIL"
    assert evaluation_sanity(rows[:3])["status"] == "PASS"
    finished = deepcopy(rows)
    finished[0].update(truncated=False, answer_phase="FINAL", passed=False)
    assert evaluation_sanity(finished)["status"] == "PASS"
    unasserted = [dict(asserted=False, truncated=False, answer_phase="FINAL")]
    assert evaluation_sanity(rows + unasserted)["status"] == "FAIL"
