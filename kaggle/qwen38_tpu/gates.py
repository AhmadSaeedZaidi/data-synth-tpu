"""Collect independent tiny-gate failures before admitting full model weights."""

import time

from glm53_offload import tracing
from glm53_offload.diagnostics import atomic_json


class NumericalGateError(RuntimeError):
    pass


def run_gates(specifications, report, directory, emit):
    failures = []
    for name, function in specifications:
        emit(stage=name)
        started = time.monotonic()
        metrics = {}
        try:
            with tracing.span(name) as metrics:
                metrics.update(function(metrics))
                if metrics.get("status") != "PASS":
                    raise NumericalGateError(f"{name} did not establish PASS")
        except Exception as error:
            metrics.update(
                status="ERROR", error_type=type(error).__name__, error_message=str(error)[:6000]
            )
            failures.append(name)
        metrics["seconds"] = time.monotonic() - started
        report[name] = dict(metrics)
        report["gate_failures"] = list(failures)
        atomic_json(directory / "report.json", report)
    if failures:
        raise NumericalGateError("Numerical gates rejected full weights: " + ", ".join(failures))
