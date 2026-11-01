"""Spatial answers known by construction and strict SFT transformation tasks."""

from glm53_offload import probe_images as p


def text_controls():
    rows = (
        (
            "collision_time",
            "A lead car is 25 metres ahead. Our closing speed is 5 metres per second "
            "and remains constant. How many seconds until collision?",
            5,
        ),
        (
            "braking_distance",
            "A car travels at 20 metres per second and brakes with constant deceleration "
            "of 5 metres per second squared. Ignore reaction time. "
            "What is the stopping distance in metres?",
            40,
        ),
        (
            "telemetry_order",
            "Three telemetry samples have timestamps in milliseconds and speeds "
            "in metres per second: (300, 12), (100, 8), (200, 10). "
            "What is the speed in the chronologically latest sample?",
            12,
        ),
        (
            "frame_sampling",
            "Exactly 20 seconds of video contain 30 frames per second. Keep every tenth "
            "frame starting with frame zero, excluding a frame at the 20-second boundary. "
            "How many frames are kept?",
            60,
        ),
    )
    return [
        dict(
            name="text_" + name,
            group="text_control",
            images=[],
            prompt=prompt + " Answer with a single number.",
            match={"kind": "first_int", "value": expected},
            asserted=True,
        )
        for name, prompt, expected in rows
    ]


def extra_fixtures():
    images, cases = {}, []

    def add(name, group, prompt, expected, image=None, kind="exact"):
        cases.append(
            dict(
                name=name,
                group=group,
                images=[] if image is None else [image],
                prompt=prompt,
                match=dict(kind=kind, expected=expected),
                asserted=True,
                max_new_tokens=4096 if group == "sft_formatting" else 2048,
            )
        )

    def spatial(name, draw, question, answer):
        canvas = p.Canvas()
        draw(canvas)
        images[name] = canvas.png()
        add(
            name,
            "general_spatial",
            question + " Answer with exactly one lowercase word.",
            answer,
            name,
        )

    for name, red_y, blue_y, answer in (
        ("spatial_above", 128, 384, "above"),
        ("spatial_below", 384, 128, "below"),
    ):
        spatial(
            name,
            lambda c, ry=red_y, by=blue_y: (
                c.circle(256, ry, 50, p.RED),
                c.rect(206, by - 50, 306, by + 50, p.BLUE),
            ),
            "Is the red circle above or below the blue square?",
            answer,
        )
    spatial(
        "spatial_inside",
        lambda c: (c.rect(80, 80, 432, 432, p.BLUE), c.circle(256, 256, 40, p.RED)),
        "Is the red circle inside or outside the blue square?",
        "inside",
    )
    for name, farthest, answer in (
        ("spatial_nearest", False, "blue"),
        ("spatial_farthest", True, "green"),
    ):
        spatial(
            name,
            lambda c: (
                c.circle(80, 256, 35, p.RED),
                c.circle(190, 256, 35, p.BLUE),
                c.circle(430, 256, 35, p.GREEN),
            ),
            "Which circle is "
            + ("farthest from" if farthest else "nearest to")
            + " the red circle: blue or green?",
            answer,
        )
    spatial(
        "spatial_overlap",
        lambda c: (c.rect(96, 96, 320, 320, p.BLUE), c.rect(240, 240, 432, 432, p.RED)),
        "Do the red and blue squares overlap? Answer yes or no.",
        "yes",
    )
    for name, x, y, answer in (
        ("spatial_northwest", 100, 100, "northwest"),
        ("spatial_southeast", 412, 412, "southeast"),
    ):
        spatial(
            name,
            lambda c, x=x, y=y: (c.circle(256, 256, 35, p.BLUE), c.circle(x, y, 35, p.RED)),
            "With image top as north, which compass direction is "
            "the red circle from the blue circle?",
            answer,
        )
    for name, prompt, expected in (
        (
            "spatial_rotate",
            "A person faces north, turns right 90 degrees, then left 180 degrees. "
            "Which direction do they face? Reply with one lowercase compass word.",
            "west",
        ),
        (
            "spatial_coordinates",
            "In image coordinates x increases right and y increases down. "
            "Point A=(10,20), B=(40,50). Which compass direction is B from A? "
            "Reply with one lowercase compass word.",
            "southeast",
        ),
        (
            "spatial_route",
            "Starting at (0,0), move 3 east, 2 north, 5 west. "
            "Return the final coordinates as a JSON array [x,y] only.",
            [-2, 2],
        ),
    ):
        add(
            name,
            "general_spatial",
            prompt,
            expected,
            kind="json_exact" if isinstance(expected, list) else "exact",
        )

    def fmt(name, prompt, expected, *, kind="json_exact", image=None):
        suffix = " Return only the requested JSON; no Markdown fences, comments, or prose."
        if kind == "jsonl_exact":
            suffix = " Return only JSONL: one object per line, no blank lines, fences, or prose."
        add("format_" + name, "sft_formatting", prompt + suffix, expected, image, kind)

    fmt(
        "typed_values",
        'Return {"count":7,"visible":true,"unknown":null}; preserve those JSON types.',
        {"count": 7, "visible": True, "unknown": None},
    )
    fmt(
        "unicode",
        'Return an object with only key "text" whose exact value is "سڑک — café 🚗".',
        {"text": "سڑک — café 🚗"},
    )
    fmt(
        "escaping",
        'Return an object with key "text" containing two lines: first line He said "STOP".; '
        "second line C:\\captures\\frame.png. Use a JSON escaped newline inside the string.",
        {"text": 'He said "STOP".\nC:\\captures\\frame.png'},
    )
    fmt(
        "identifiers",
        'Return {"id":"00017","timestamp_ms":1750,"confidence":0.75}. '
        "Keep id as a string and timestamp_ms as an integer.",
        {"id": "00017", "timestamp_ms": 1750, "confidence": 0.75},
    )
    fmt(
        "unknowns",
        "No image or temporal evidence is supplied. Return an object with exactly "
        "traffic_light, crossed_stop_line, event; set every value to null.",
        {"traffic_light": None, "crossed_stop_line": None, "event": None},
    )
    annotation = dict(
        subject_id="vehicle-red",
        timestamp_ms=1750,
        claims=dict(traffic_light="red", crossed_stop_line=False, event="no_red_light_running"),
        confidence=0.75,
    )
    fmt(
        "annotation",
        "Format this supplied record as a production StructuredAnnotation: "
        "subject_id vehicle-red, timestamp_ms 1750, claims traffic_light red, "
        "crossed_stop_line false, "
        "event no_red_light_running, confidence 0.75. Include exactly those four top-level keys.",
        annotation,
        kind="annotation_json",
    )
    fmt(
        "vision_annotation",
        'For this single BeamNG frame return exactly subject_id="beamng-roadside", '
        "timestamp_ms=2000, claims with only visible_cars (an integer counted from the image), "
        "crossed_stop_line=null, event=null, and confidence=null. A single image does not "
        "establish a crossing event.",
        dict(
            subject_id="beamng-roadside",
            timestamp_ms=2000,
            claims=dict(visible_cars=1, crossed_stop_line=None, event=None),
            confidence=None,
        ),
        kind="annotation_json",
        image="beamng_roadside",
    )
    fmt(
        "jsonl_order",
        'Convert records (id="b",timestamp_ms=200) and (id="a",timestamp_ms=100) '
        "to two JSONL objects with exactly id and timestamp_ms, sorted by timestamp_ms.",
        [dict(id="a", timestamp_ms=100), dict(id="b", timestamp_ms=200)],
        kind="jsonl_exact",
    )
    messages = {
        "messages": [
            dict(role="user", content="How many cars?"),
            dict(role="assistant", content="2"),
        ]
    }
    fmt(
        "sft_chat",
        "Create one SFT object with only messages. It contains two messages with only "
        'role and content: user "How many cars?", assistant "2". Do not include thinking.',
        messages,
    )
    fmt(
        "sft_jsonl",
        "Create two SFT JSONL records in this order. Each has only messages, each message "
        'only role/content. Record 1: user "Signal?", assistant "red". '
        'Record 2: user "Crossing?", assistant "unknown". No thinking text.',
        [
            {
                "messages": [
                    dict(role="user", content="Signal?"),
                    dict(role="assistant", content="red"),
                ]
            },
            {
                "messages": [
                    dict(role="user", content="Crossing?"),
                    dict(role="assistant", content="unknown"),
                ]
            },
        ],
        kind="jsonl_exact",
    )
    fmt(
        "repair",
        "Repair only the syntax of this record: {'id':'001','ok':True,'label':None,}. "
        "Preserve values and use only id, ok, label.",
        dict(id="001", ok=True, label=None),
    )
    fmt(
        "vision_ocr",
        'Read the text in this image and return an object with only key "text", '
        "preserving uppercase letters and the space.",
        {"text": "STOP 7"},
        image="text_stop_7",
    )
    return images, cases
