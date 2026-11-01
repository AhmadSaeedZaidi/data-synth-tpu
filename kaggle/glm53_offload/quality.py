"""Same image fixtures and answer matchers as the authenticated image gate."""

import re

from . import probe_images

PROBE_VERSION = 3

IMAGE_SIZE = probe_images.SIZE


def _ints(text):
    return [int(m) for m in re.findall(r"-?\d+", text or "")]


def _words_in(text):
    # Alphanumeric tokens, so "STOP 7" yields {"stop", "7"} and "red-green" yields two
    # tokens: a hedged answer that names a second colour is a failure, not a pass.
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def match(spec, answer):
    """-> (passed, detail). `spec` is one of the match kinds below."""
    kind = spec["kind"]
    if kind == "contains_all":
        present, absent = _words_in(answer), set(spec.get("absent", []))
        missing = sorted(set(spec["words"]) - present)
        forbidden = sorted(absent & present)
        return (not missing and not forbidden), {"missing": missing, "forbidden_present": forbidden}
    if kind == "first_int":
        found = _ints(answer)
        return (bool(found) and found[0] == spec["value"]), {
            "integers": found[:8],
            "expected": spec["value"],
        }
    if kind == "ints_in_order":
        found = [n for n in _ints(answer) if 0 <= n <= 99]
        return found == list(spec["values"]), {
            "integers": found[:8],
            "expected": list(spec["values"]),
        }
    raise ValueError(f"unknown match kind {kind!r}")


COLOUR_Q = "What is the dominant colour of this image? Answer with one word."

COUNT_Q = "How many circles are in this image? Answer with a single number."

TEXT_Q = "What text is written in this image? Reply with the text only."

SIDE_Q = "Which side of this image is the red area on: the left or the right? Answer with one word."

FRAME_Q = (
    "This message contains two images. Report the number printed in the first image, "
    "then the number printed in the second image, in that order. "
    "Answer as two numbers separated by a comma."
)


def cases(images=None):
    """The battery. `images` defaults to the generated fixtures."""
    available = set(probe_images.fixtures() if images is None else images)
    out = []

    def add(name, group, image_names, prompt, spec, asserted=True):
        if not set(image_names) <= available:
            raise KeyError(f"{name}: missing fixtures {sorted(set(image_names) - available)}")
        out.append(
            {
                "name": name,
                "group": group,
                "images": image_names,
                "prompt": prompt,
                "match": spec,
                "asserted": asserted,
            }
        )

    for name, word, others in (
        ("solid_red", "red", ["green", "blue"]),
        ("solid_green", "green", ["red", "blue"]),
        ("solid_blue", "blue", ["red", "green"]),
    ):
        add(
            f"colour_{word}",
            "single_image_colour",
            [name],
            COLOUR_Q,
            {"kind": "contains_all", "words": [word], "absent": others},
        )

    # Counterfactual pair: identical question, different images, different answers.
    add("count_circles_7", "counting", ["circles_7"], COUNT_Q, {"kind": "first_int", "value": 7})
    add("count_circles_3", "counting", ["circles_3"], COUNT_Q, {"kind": "first_int", "value": 3})

    add(
        "read_text_stop_7",
        "text_reading",
        ["text_stop_7"],
        TEXT_Q,
        {"kind": "contains_all", "words": ["stop", "7"]},
    )
    add(
        "read_text_go_3",
        "text_reading",
        ["text_go_3"],
        TEXT_Q,
        {"kind": "contains_all", "words": ["go", "3"]},
    )

    add(
        "red_on_left",
        "spatial_side",
        ["half_red_left"],
        SIDE_Q,
        {"kind": "contains_all", "words": ["left"], "absent": ["right"]},
    )
    add(
        "red_on_right",
        "spatial_side",
        ["half_red_right"],
        SIDE_Q,
        {"kind": "contains_all", "words": ["right"], "absent": ["left"]},
    )

    # The order test. The reversed pair is the one that cannot pass by accident.
    add(
        "frame_order_1_then_2",
        "image_order",
        ["frame_1", "frame_2"],
        FRAME_Q,
        {"kind": "ints_in_order", "values": [1, 2]},
    )
    add(
        "frame_order_2_then_1",
        "image_order",
        ["frame_2", "frame_1"],
        FRAME_Q,
        {"kind": "ints_in_order", "values": [2, 1]},
    )

    # The first case, asked again at the end with the same image and the same question.
    # Its latency against colour_red's separates the two explanations for a slow image
    # path: a per-request stall stays slow, a per-distinct-image cost does not. That
    # difference decides whether a pipeline can amortise the cost by asking several
    # questions of one frame, so it is worth a single request.
    add(
        "repeat_colour_red",
        "repeat_cost",
        ["solid_red"],
        COLOUR_Q,
        {"kind": "contains_all", "words": ["red"], "absent": ["green", "blue"]},
    )

    # Not asserted. Recorded because a model that answers 7 with no image at all
    # weakens the counting result above, and that has to stay visible.
    add(
        "no_image_control",
        "control",
        [],
        COUNT_Q,
        {"kind": "first_int", "value": 7},
        asserted=False,
    )
    return out
