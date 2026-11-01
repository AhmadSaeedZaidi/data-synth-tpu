"""A small, strict temporal review contract for two-car video experiments."""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PairReview(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contact: Literal["yes", "no", "uncertain"]
    first_contact_seconds: float | None = Field(default=None, ge=0)
    red_front_damage_visible: Literal["yes", "no", "uncertain"]
    blue_rear_damage_visible: Literal["yes", "no", "uncertain"]
    red_stopped_short: Literal["yes", "no", "uncertain"]
    reasons: list[str] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def contact_time_consistency(self) -> "PairReview":
        if (self.contact == "yes") != (self.first_contact_seconds is not None):
            raise ValueError("only a positive contact claim may include its time")
        if self.contact == "yes" and self.red_stopped_short == "yes":
            raise ValueError("contact and stopping short cannot both be yes")
        return self


_FIELDS = (
    ("Vehicle contact", "contact", r"yes|no|uncertain"),
    (
        "First contact time seconds",
        "first_contact_seconds",
        r"unknown|(?:0|[1-9][0-9]*)(?:\.[0-9]+)?",
    ),
    ("Red front damage visible", "red_front_damage_visible", r"yes|no|uncertain"),
    ("Blue rear damage visible", "blue_rear_damage_visible", r"yes|no|uncertain"),
    ("Red stopped short", "red_stopped_short", r"yes|no|uncertain"),
)


def parse_pair_review(text: str, duration_seconds: float) -> PairReview:
    if not isinstance(text, str) or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("invalid pair review")
    lines = text.replace("\r\n", "\n").rstrip("\n").split("\n")
    if len(lines) < 10 or lines[:2] != ["# Pair review", ""] or lines[7:9] != ["", "## Reasons"]:
        raise ValueError("invalid pair review")
    values = {}
    for line, (label, key, pattern) in zip(lines[2:7], _FIELDS, strict=True):
        match = re.fullmatch(rf"{re.escape(label)}: ({pattern})", line)
        if match is None:
            raise ValueError("invalid pair review")
        values[key] = match.group(1)
    seconds = values["first_contact_seconds"]
    values["first_contact_seconds"] = None if seconds == "unknown" else float(seconds)
    if (
        values["first_contact_seconds"] is not None
        and values["first_contact_seconds"] > duration_seconds
    ):
        raise ValueError("pair review contact time exceeds video duration")
    reasons = []
    for line in lines[9:]:
        if not line.startswith("- ") or not line[2:].strip() or "<" in line or ">" in line:
            raise ValueError("invalid pair review")
        reasons.append(line[2:])
    try:
        return PairReview(**values, reasons=reasons)
    except ValueError:
        raise ValueError("invalid pair review") from None


def make_pair_prompt(duration_seconds: float, grounding: dict | None = None) -> str:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("video duration must be positive")
    if grounding is None:
        source = "No simulator telemetry is supplied. Judge only what the video supports."
    else:
        source = (
            "Independent simulator Damage scalars at the first and last frames only: "
            f"red {grounding['red_first']}, {grounding['red_last']}; "
            f"blue {grounding['blue_first']}, {grounding['blue_last']}. "
            "Damage measures structural deformation, not the exact contact time or cause. "
            "Keep visible damage claims separate from these scalar readings."
        )
    return (
        f"Review this {duration_seconds:g}-second, 20 simulation-fps, fixed-camera video. "
        "A red car begins behind a blue car. Decide whether the red car contacts the blue car "
        "or stops short. A small gap in one frame is not proof about the full clip. "
        "Report the earliest approximate contact time only if the video supports contact. "
        "Distinguish visible front damage to the red car and visible rear damage to the blue car "
        "from merely knowing their positions. Use uncertain for weak visual evidence. "
        f"{source}\n\n"
        "Return exactly these Markdown lines with one choice per field, "
        "no code fence or extra fields:\n\n"
        "# Pair review\n\n"
        "Vehicle contact: yes|no|uncertain\n"
        "First contact time seconds: number|unknown\n"
        "Red front damage visible: yes|no|uncertain\n"
        "Blue rear damage visible: yes|no|uncertain\n"
        "Red stopped short: yes|no|uncertain\n\n"
        "## Reasons\n- one short evidence reason per line"
    )


def make_pair_frames_prompt(times_seconds: list[float], grounding: dict | None = None) -> str:
    """Adapt the same contract to sparse, timestamped stills without implying full video."""
    if (
        len(times_seconds) < 2
        or len(times_seconds) > 8
        or any(not math.isfinite(value) or value < 0 for value in times_seconds)
        or times_seconds != sorted(set(times_seconds))
    ):
        raise ValueError("select 2..8 distinct increasing nonnegative frame times")
    base = make_pair_prompt(times_seconds[-1] + 0.05, grounding)
    video_intro = (
        f"Review this {times_seconds[-1] + 0.05:g}-second, 20 simulation-fps, fixed-camera video. "
    )
    return (
        base.replace(
            video_intro,
            "Review these selected fixed-camera frames from a 20 simulation-fps sequence. "
            "Each image has its simulation time immediately before it. "
            "Intervening frames are omitted. ",
        )
        .replace(
            "A small gap in one frame is not proof about the full clip. ",
            "A small gap in one frame is not proof about the whole sequence. ",
        )
        .replace(
            "Report the earliest approximate contact time only if the video supports contact. ",
            "Report the earliest approximate contact time only if the shown frames "
            "support contact; "
            "otherwise use unknown. ",
        )
    )
