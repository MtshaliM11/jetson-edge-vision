"""Original deterministic labelled detections; no downloaded data or media."""


def observation(object_id, x=10, *, y=10, class_id=1, class_name="person"):
    return ({"class_id": class_id, "class_name": class_name, "confidence": 0.9,
             "box_xyxy": [x, y, x + 30, y + 30]}, object_id)


def frame(*observations, visible=None, reset=False):
    if visible is None:
        visible = {identity: {"class_id": record["class_id"], "class_name": record["class_name"]}
                   for record, identity in observations if identity is not None}
    return {"detections": [record for record, _ in observations],
            "truth_ids": [identity for _, identity in observations],
            "visible_objects": visible, "reset_before": reset}


def synthetic_scenarios() -> list[dict]:
    """Expected values are diagnostic counts at the unchanged default config."""
    person = {"A": {"class_id": 1, "class_name": "person"}}
    scenarios = []

    def add(name, description, frames, **expected):
        scenarios.append({"name": name, "description": description,
                          "frames": [dict(row, frame_index=i) for i, row in enumerate(frames)],
                          "expected_default": expected})

    add("one_moving", "One object moves four pixels per frame; overlap remains strong.",
        [frame(observation("A", 10 + 4 * i)) for i in range(8)],
        id_switches=0, extra_identity_tracks=0, identity_merges=0, tracks_created=1)
    add("two_independent", "Two same-class objects move with alternating detector order.",
        [frame(*([observation("A", 10 + i * 2), observation("B", 100 - i * 2)][::(-1 if i % 2 else 1)]))
         for i in range(7)], id_switches=0, extra_identity_tracks=0, tracks_created=2)
    add("crossing", "Identical boxes at crossing, followed by changed order, swap physical identities.",
        [frame(observation("A", a), observation("B", b)) for a, b in ((10, 70), (20, 60), (30, 50), (40, 40))]
        + [frame(observation("B", 30), observation("A", 50)), frame(observation("B", 20), observation("A", 60))],
        id_switches=2, extra_identity_tracks=2, identity_merges=2)
    add("temporary_misses", "Visible A is not detected for two frames, then returns at the same box.",
        [frame(observation("A")), frame(observation("A")), frame(visible=person), frame(visible=person),
         frame(observation("A")), frame(observation("A"))],
        id_switches=0, missed_object_frames=2, tracks_created=1)
    add("leaves_scene", "A leaves permanently; its retained track must expire.",
        [frame(observation("A")), frame(observation("A"))] + [frame() for _ in range(12)],
        id_switches=0, missed_object_frames=0, active_tracks=0, tracks_expired=1)
    add("reappears_after_gap", "Same physical A is absent seven frames, then reappears.",
        [frame(observation("A")), frame(observation("A"))] + [frame() for _ in range(7)]
        + [frame(observation("A")), frame(observation("A"))],
        id_switches=1, extra_identity_tracks=1, missed_object_frames=0)
    add("category_flip", "A is physically a person but one detection is classified as bird.",
        [frame(observation("A")),
         frame(observation("A", class_id=16, class_name="bird"), visible=person),
         frame(observation("A"))],
        category_errors=1, id_switches=2, extra_identity_tracks=1)
    add("sudden_jump", "Same A jumps without box overlap; all tested thresholds must lose its ID.",
        [frame(observation("A", x)) for x in (10, 100, 200)],
        id_switches=2, extra_identity_tracks=2)
    add("scene_change_without_reset", "A new physical B replaces A at the same box after a cut.",
        [frame(observation("A")), frame(observation("A")),
         frame(observation("B")), frame(observation("B"))],
        id_switches=0, identity_merges=1, tracks_created=1)
    add("explicit_scene_reset", "An explicit reset scopes IDs to a new session/scene.",
        [frame(observation("A")), frame(observation("A")),
         frame(observation("B"), reset=True), frame(observation("B"))],
        id_switches=0, identity_merges=0, tracks_created=2, resets=1)
    add("identical_overlap", "Fully overlapping A/B reverse detection order; tie-breaking cannot know identity.",
        [frame(observation("A"), observation("B")), frame(observation("B"), observation("A"))],
        id_switches=2, extra_identity_tracks=2, identity_merges=2)
    add("false_positive_burst", "Two observations of a false detection create and confirm a false-born track.",
        [frame(observation(None)), frame(observation(None)), frame(), frame()],
        false_positive_detections=2, false_tracks_created=1, false_tracks_confirmed=1)
    add("persistent_false_positive", "Three false observations can confirm even min_hits=3.",
        [frame(observation(None)) for _ in range(3)],
        false_positive_detections=3, false_tracks_created=1, false_tracks_confirmed=1)
    add("box_jitter", "Alternating 12-pixel localization jitter has adjacent IoU 0.4286.",
        [frame(observation("A", x)) for x in (10, 22, 10, 22)],
        id_switches=0, tracks_created=1)
    add("fast_motion", "Twenty-pixel steps have IoU 0.2; a lower threshold can preserve IDs.",
        [frame(observation("A", x)) for x in (10, 30, 50, 70)],
        id_switches=3, extra_identity_tracks=3)
    add("stale_identity_after_gap", "A leaves; a different B arrives in its old box after seven empty frames.",
        [frame(observation("A")), frame(observation("A"))] + [frame() for _ in range(7)]
        + [frame(observation("B")), frame(observation("B"))],
        identity_merges=0, tracks_created=2)
    add("crowded_20", "Twenty independent same-class objects with reversed order on alternate frames.",
        [frame(*([observation(f"O{j}", 60 * j + 2 * i) for j in range(20)][::(-1 if i % 2 else 1)]))
         for i in range(8)], id_switches=0, identity_merges=0, tracks_created=20)
    return scenarios
