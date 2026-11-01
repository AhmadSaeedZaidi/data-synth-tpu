"""Strict output scoring. No repair, teacher-as-truth, or JSON type coercion."""

import json
import math

from glm53_offload import quality

from .annotation_reference import StructuredAnnotation
from .pair_review_reference import parse_pair_review

KINDS = {"exact", "json_exact", "jsonl_exact", "annotation_json", "pair_markdown"}
LEGACY = {"contains_all", "first_int", "ints_in_order"}


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite_number")

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite_number")
        return result

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            typed_equal(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            typed_equal(a, e) for a, e in zip(actual, expected, strict=True)
        )
    return actual == expected


def same_shape(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            same_shape(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_shape(a, e) for a, e in zip(actual, expected, strict=True)
        )
    return True


def validate_spec(spec):
    if spec.get("kind") not in KINDS | LEGACY:
        raise ValueError("unsupported_evaluation_matcher")
    if spec["kind"] in {"exact", "json_exact", "jsonl_exact", "annotation_json"}:
        if "expected" not in spec:
            raise ValueError("missing_expected_output")
    if spec["kind"] == "pair_markdown" and spec.get("duration_seconds", 0) <= 0:
        raise ValueError("invalid_duration")


def match(spec, answer):
    validate_spec(spec)
    kind = spec["kind"]
    if kind in LEGACY:
        passed, detail = quality.match(spec, answer)
        return passed, {**detail, "assertion_scope": "answer_correctness"}
    detail = dict(
        schema_valid=False,
        semantic_valid=None,
        assertion_scope="schema_only" if kind == "pair_markdown" else "format_and_content",
    )
    try:
        if kind == "pair_markdown":
            value = parse_pair_review(answer, spec["duration_seconds"]).model_dump(mode="json")
            detail.update(
                schema_valid=True,
                assertion_scope="schema_only",
                parsed=value,
                reference_status="independent_event_labels_missing",
                training_ready=False,
            )
            return True, detail
        if kind == "exact":
            passed = answer == spec["expected"]
            return passed, dict(
                schema_valid=passed, semantic_valid=passed, assertion_scope="exact_text"
            )
        if kind == "jsonl_exact":
            lines = answer.splitlines()
            if not lines or any(not line.strip() for line in lines):
                raise ValueError("empty_jsonl_record")
            value = [strict_json(line) for line in lines]
            if any(not isinstance(row, dict) for row in value):
                raise ValueError("jsonl_records_must_be_objects")
        else:
            value = strict_json(answer)
            if kind == "annotation_json":
                # Validate with the exact production class; compare original JSON
                # separately so trimming/coercion cannot hide formatting mistakes.
                StructuredAnnotation.model_validate(value)
        detail.update(
            schema_valid=same_shape(value, spec["expected"]),
            semantic_valid=typed_equal(value, spec["expected"]),
        )
        return detail["semantic_valid"], detail
    except (ValueError, TypeError, KeyError) as error:
        detail["error_type"] = type(error).__name__
        return False, detail
