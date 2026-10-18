# Phase 3B: reproducible tracking evaluation

Completed 2026-10-18. The purpose is a transparent diagnostic baseline before
counting, not a claim of deployment tracking accuracy. Production detector,
tracker, video pipeline, annotations, application CLI, configuration, defaults, and
all original 85 tests are unchanged. No dependencies/system changes, additional
media downloads, counting, TensorRT, camera integration, or dashboard work occurred.

## Existing tracker review

The class-aware greedy IoU tracker uses the last observed box, with no velocity,
appearance, Kalman filter, global assignment, or scene-cut detector. Same-class
candidates at/above the IoU threshold sort by descending overlap, then track ID,
then input detection index. Each track/detection matches once. Unmatched detections
create IDs; unmatched tracks expire when misses exceed the configured limit.
Consecutive observations confirm tentative tracks; confirmation stays sticky.
Only current detections are drawn, while missed tracks remain available for matching.

Strengths: dependency-free CPU association, deterministic input-order-aware ties,
session-local state, class separation, short-gap tolerance, and simple auditability.
Limits: ambiguous crossings, non-overlap motion, stale same-class box reuse,
category flips, detection errors, and fragmentation after expiration/cuts. A
confirmed track is a hypothesis, not proof of a real object. Full lifecycle and
Phase 3A evidence remain in [Phase 3A](phase3a.md).

## Evaluation architecture and provenance

- `evaluation_scenarios.py`: 17 original deterministic labelled detection sequences.
  Physical identities/visibility/categories are separate from detections.
- `tracking_evaluation.py`: independent standard-library CLI, parameter-grid
  replay, custom diagnostic scoring, optional manual-label loading, and JSON reports.
- Existing DEBUG traces: zero-based contiguous frame records, including empty
  frames. Saved IDs/statuses are discarded; only box/category/score enter a fresh
  instance of the unchanged tracker. Input labels never enter the tracker.
- Fixed factorial grid: IoU **0.1/0.3/0.5**, max missed **0/5/10**, minimum hits
  **1/2/3** = **27 configurations**. Each sequence/configuration uses **10 fresh-state
  repeats**. Assignment/count repeatability is checked; timing samples span all
  repeats, but diagnostic numerators/denominators describe one sequence, not ten
  duplicated copies.

Reports record tracker, evaluator, scenario-source, trace, annotation, and companion
report SHA256 hashes where applicable. A normalized detection-sequence hash excludes
old IDs/log timestamps and includes frame indices, boxes, categories, scores, and
explicit reset flags. Companion pipeline reports are checked against trace frame
and detection totals; that is a consistency check, not cryptographic authentication
that a report was generated with a trace. Preserve original evidence together.

The evaluator requires contiguous full traces; dropping empty or difficult frames
would otherwise change ageing and bias evaluation. Saved trace JSONL can contain
the structured logger envelope or direct frame objects. Malformed JSON, missing
frames, invalid labels, wrong annotation hashes, and reused output paths fail.
Only short offline traces are intended: the evaluator retains observation histories
and timing samples, unlike the incremental production pipeline.

## Diagnostic definitions (not standard MOT metrics)

No MOTChallenge, MOTA, IDF1, or HOTA compatibility is claimed. The custom definitions
below are deliberately small and unit-tested. Counts use **all current detections,
including tentative tracks**, not only confirmed ones. Confirmation statistics are
reported separately.

| Field | Definition |
| --- | --- |
| `identity_links` | Each later labelled observation of the same physical identity within a reset scope; gaps are included |
| `id_continuations` | Such links whose assigned track ID equals that identity's previous assigned ID |
| `id_switches` | Such links whose assigned ID differs; counts changes after gaps too |
| `persistence_ratio` | Continuations / identity links; null if no links. Conditional on observed detections, not overall tracking accuracy |
| `extra_identity_tracks` | Sum over physical identities of distinct assigned IDs minus one; a **fragmentation proxy**, not standard track-fragmentation counting |
| `identity_merges` | A track's successive non-null physical assignments change identity; reveals wrong inheritance even if neither identity has an ID switch |
| `missed_object_frames` | Reviewed visible objects without a linked detection; absent/offscreen objects do not count |
| `category_errors` | Linked detection class ID/name differs from the reviewed physical object's category |
| `false_positive_detections` | Reviewed detections explicitly labelled null, including duplicate detections |
| `false_tracks_created` | Track IDs whose first detection is labelled null (false-born tracks) |
| `false_tracks_confirmed` | Those false-born IDs that reach confirmed status; birth-based definition even if later linked to a real object |
| `false_positive_matches` | Null-labelled detections assigned an already existing ID |

Track creation/expiration, active state, confirmed observations, and ID frame spans
are separate observable lifecycle statistics. Expiration excludes tracks cleared
by an explicit reset. Reset scopes prevent numeric ID reuse from being mistaken
for an identity merge across independent sessions.

Detector and association diagnostics must be interpreted together. The crossing
case has perfect labelled detections and wrong associations: a tracker limitation.
The category-flip case has a known detector category error and consequent ID changes
under strict class matching: do not attribute it solely to association. Short
dropout/absence cases distinguish detector misses from true disappearance. False
detections can produce apparently persistent confirmed tracks. No single aggregate
"accuracy" score combines these different causes or the arbitrary toy-case mix.

## Labelled synthetic results at unchanged defaults

Default: IoU 0.3, max missed 5, min hits 2. These are measured deterministic replay
counts, checked against authored expected outcomes. **S** = ID changes,
**F** = extra distinct IDs/fragmentation proxy, **M** = identity-merge transitions.

| Scenario | S | F | M | Expected/observed behavior |
| --- | ---: | ---: | ---: | --- |
| One moving object | 0 | 0 | 0 | One ID across eight frames |
| Two independent moving objects | 0 | 0 | 0 | Two IDs despite alternating detection order |
| Same-class crossing | 2 | 2 | 2 | Identical crossing boxes/order change swap physical identities |
| Two temporary missed detections | 0 | 0 | 0 | Same ID returns; two visible object-frame misses |
| Object leaves permanently | 0 | 0 | 0 | Track expires; final active state empty |
| Object reappears after seven absent frames | 1 | 1 | 0 | Old track expired; true absence is not a detector miss |
| Category flip person → bird → person | 2 | 1 | 0 | One category error; ID 1 → 2 → 1 illustrates switches differing from extra distinct IDs |
| Sudden non-overlap jumps | 2 | 2 | 0 | Three observations create three IDs |
| Scene replacement without reset | 0 | 0 | 1 | Different object inherits stale ID in the same box; switches alone would miss the error |
| Explicit complete reset | 0 | 0 | 0 | Numeric ID restarts at 1 in a new scope; two tracks created across scopes |
| Identical overlapping boxes/reversed order | 2 | 2 | 2 | Tie-breaking cannot recover physical identity |
| Two-frame false detection burst | 0 | 0 | 0 | One false-born track, confirmed at default two hits |
| Three-frame persistent false detection | 0 | 0 | 0 | One false-born confirmed track |
| Box jitter (adjacent IoU ≈ 0.4286) | 0 | 0 | 0 | Default threshold tolerates alternating boxes |
| Fast motion (adjacent IoU 0.2) | 3 | 3 | 0 | Default threshold fails to associate four observations |
| Different object occupies old box after seven empty frames | 0 | 0 | 0 | Default expiry avoids stale identity inheritance |
| Twenty simultaneous independent objects | 0 | 0 | 0 | Twenty IDs stable over eight frames with reversed order |

Explicit scene reset is a controlled call to the **existing** `reset()` API. It
does not add automatic scene resetting to production video. Some expected outcomes
are intentional baseline failures; passing the regression test means they were
reproduced, not that crossings or jumps were tracked correctly.

## Parameter sensitivity and recommendation

Each comparison changes the stated setting on the identical labelled sequence.

| Parameter experiment | Measured benefit | Measured cost/limit |
| --- | --- | --- |
| IoU 0.1 versus 0.3 | Fast-motion ID changes 0 versus 3 | Lower overlap accepts weaker evidence; cannot fix zero-overlap jumps or ambiguous crossings |
| IoU 0.5 versus 0.3 | More restrictive matching, not a demonstrated accuracy benefit | Jitter case changes 3 versus 0; stricter is not automatically better |
| Max missed 0 versus 5 | Immediate stale-state removal | Two-frame detector dropout changes ID once versus zero |
| Max missed 10 versus 5 | Seven-frame reappearance changes ID 0 versus 1 | Different-object-after-gap case has one identity merge versus zero |
| Minimum hits 3 versus 2 | Two-frame false-born track confirmations 0 versus 1 | False track creation remains 1; tentative IDs are still displayed; genuine confirmation delayed |
| Minimum hits 1/2/3 | Controls status latency | A three-frame persistent false detection confirms for all three settings |

**Recommendation: retain 0.3 / 5 / 2 as the evaluation baseline. Defaults were not
changed.** This is an engineering recommendation to avoid unvalidated tuning, not
a claim these values are optimal. The synthetic grid demonstrates competing costs,
and the available clips cannot select an accuracy winner without reviewed labels.
Five misses correspond to five source-frame intervals (about 0.208 seconds at
24 FPS), not five processing-loop intervals. Longer retention and lower IoU need
scene-specific reviewed evidence. Higher min hits only marks confirmation and is
not a false-positive filter.

## Available local video review

A local project-tree inventory found only `data/sintel-sample.mp4` and
`data/sintel-trailer-480p.mp4` as source videos. Existing annotated outputs are
derivatives, not additional independent samples. No videos were downloaded.

1. Existing sample: trailer frames 360–407 (15–17 seconds), 48 frames, 854x480,
   24 FPS. Attribution/preparation remain in [Phase 2B](phase2b.md#real-gpu-verification-and-attribution).
2. Street section: derived locally from already available trailer frames 480–527
   (20–22 seconds), same geometry/FPS, re-encoded mp4v without audio. It supplies a
   smaller animated figure and background surfaces, but is **the same animated
   source**, not independent real-world evidence.

Street sample SHA256:
`888e98d088f762af9d07f809e61ca31049e499e7789f49501b4dcfe5965ad31c`.
Local attribution/derivation: `data/phase3b-sintel-street-attribution.json`.
Sintel trailer © Blender Foundation | durian.blender.org, CC BY 3.0; this phase
only excerpts the already local source and adds detector/tracker annotations.
No media, raw labels, private footage, or traces are added to Git.

### Unlabelled observations, not accuracy scores

Both clips were replayed over all 27 configurations on fixed detections. Identity,
fragmentation-proxy, category-error, and false-track accuracy fields are **null**
for these real traces; no reviewed ground truth was created or invented.

The main default trace reproduces #001 at frames 0–23, #002 at 36–43 and 46, and
#003/bird at 44–45. Visual review in Phase 3A showed the person-to-bird category
error and ID fragmentation across a shot/gap. The fresh 2026-10-18 trace has the
same observed spans. A class flip is detector-driven; recovering #002 demonstrates
short-gap retention, not a measured identity-accuracy rate.

The street trace has ten `bed` observations, with ID 1 at frames 4, 7, 8, 11 and
ID 2 at 24–27, 30–31. Inspected output frames 0, 8, 16, 24 show boxes over building
surfaces rather than beds, while the visible animated figure is not labelled as
person in those inspected frames. These are qualitative detector failure
observations, not an exhaustive false-positive or recall metric. Confirmation
can stabilize such detections without making them correct. Preview:
`outputs/phase3b-street-preview.jpg`.

| Replay configuration (one setting changed) | Main IDs created / confirmed observations | Street IDs created / confirmed observations |
| --- | --- | --- |
| Default 0.3 / 5 / 2 | 3 / 32 | 2 / 7 |
| IoU 0.1 | 3 / 32 | 2 / 7 |
| IoU 0.5 | 4 / 31 | 2 / 7 |
| Max missed 0 | 4 / 31 | 5 / 5 |
| Max missed 10 | 3 / 32 | 2 / 7 |
| Min hits 3 | 3 / 29 | 2 / 4 |

These are observable hypothesis/status counts, **not** physical-object counts,
ID-switch metrics, or a ranking of tracking accuracy. The main clip yields four
distinct ID observation patterns across the full grid; the labels do not tell us
which pattern is correct.

## Measured performance

Fresh sequential Orin CUDA runs on 2026-10-18 used the unchanged FP32 detector,
confidence 0.5, warm-up 3, mp4v, and **DEBUG logging for all three runs**. No concurrent
evaluation sweep ran during these video measurements. The main default/comparison
normalized detection hash is identical:
`3d4f75ce05feb7639ef70d7224e65308beff1f3d97f8bfeae77eff1373b9edeb`.
The comparison is an experiment (0.1 / 10 / 3), not a recommended new default.

| Measurement | Main default 0.3/5/2 | Main comparison 0.1/10/3 | Street default |
| --- | ---: | ---: | ---: |
| Frames / source FPS | 48 / 24 | 48 / 24 | 48 / 24 |
| Detection observations | 35 | 35 | 10 |
| Mean tracker update | 0.1139 ms | 0.1279 ms | 0.0700 ms |
| Median tracker update | 0.1162 ms | 0.1192 ms | 0.0843 ms |
| Mean synchronized model latency | 183.58 ms | 192.79 ms | 180.40 ms |
| Processing-loop throughput | 4.78 FPS | 4.60 FPS | 4.92 FPS |
| Full end-to-end throughput | 2.37 FPS | 2.11 FPS | 2.29 FPS |
| Total processing | 20,241.91 ms | 22,793.38 ms | 20,963.61 ms |

All outputs decoded all 48 frames at 854x480 and 24 FPS. DEBUG serialization/I/O is
included in loop throughput but outside association timing. Full totals include
model loading, warm-up, video processing, and decode verification. GPU inference
timing boundaries are unchanged from [Phase 2B](phase2b.md#performance-boundaries).
CPU frequency, power mode, clocks, thermal state, cache behavior, and background
work were not controlled/modified. A single sequential pair is not a statistically
isolated speed comparison; most of the total difference is outside the tiny CPU
association stage. No real-time or parameter-speedup claim is made. Historical
Phase 3A tracking-off/on measurements remain historical, not fresh Phase 3B runs.

CPU-only replay uses `perf_counter_ns` around the unchanged tracker update. The
default main trace measured **0.02070 ms/update** over 480 replay calls; the street
trace **0.01145 ms/update**. These differ from live-video tracker timing because
the execution context is different. Baseline synthetic update means were
**0.02496 ms** for one moving object, **0.05061 ms** for two independent objects,
and **1.63309 ms** for twenty simultaneous objects, illustrating pairwise CPU cost.
These tiny synthetic timings are not stable workload guarantees or peak-load tests.

`update_only_fps` divides call count by time inside tracker updates.
`replay_processing_fps` includes evaluator scoring/history bookkeeping but excludes
file loading, CUDA, decoding, annotation, and encoding. **Neither is video throughput**.
Labelled and unlabelled bookkeeping costs differ; do not compare replay FPS as if
it measured only tracking logic. Identity counts describe a single sequence; timing
call count includes all repeats.

## Reproduction commands and artifacts

Use the existing environment; no installation:

```bash
cd ~/jetson-projects/ai/projects/jetson-edge-vision
source ~/jetson-projects/ai/ai-env/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs

# Portable labelled suite only; works without media, weights, Torch, OpenCV, or GPU:
python -m jetson_edge_vision.tracking_evaluation --repeats 10 \
  --output outputs/synthetic-evaluation-repeat.json

# Complete synthetic + fixed main-video detection replay:
python -m jetson_edge_vision.tracking_evaluation --repeats 10 \
  --trace outputs/phase3b-sintel-default.jsonl \
  --video-report outputs/phase3b-sintel-default.json \
  --output outputs/main-evaluation-repeat.json

# Additional local clip replay, without rerunning synthetic cases:
python -m jetson_edge_vision.tracking_evaluation --trace-only --repeats 10 \
  --trace outputs/phase3b-street-default.jsonl \
  --video-report outputs/phase3b-street-default.json \
  --output outputs/street-evaluation-repeat.json
```

Use new report/video filenames on repeats. Omit `--output` to print JSON to stdout;
`--baseline-only` selects just the unchanged defaults. Existing report paths are
refused, including symlinks; shell redirection itself can overwrite files. Results
include source hashes, runtime, all per-case counts/observation spans, and timings.
Timing values will vary; labelled diagnostic counts should repeat exactly.

Fresh CUDA evidence can be reproduced with an existing DEBUG device/log config
(`outputs/phase3a-debug-config.json` in this workspace contains CUDA/DEBUG):

```bash
# If needed, create a local ignored debug config on a fresh checkout:
cat > outputs/evaluation-debug.json <<'JSON'
{"device": "cuda", "log_level": "DEBUG"}
JSON
python -m jetson_edge_vision --config outputs/evaluation-debug.json \
  --video data/sintel-sample.mp4 --output outputs/eval-default-repeat.mp4 \
  --confidence 0.5 --warmup 3 --track --track-iou 0.3 \
  --track-max-missed 5 --track-min-hits 2 \
  > outputs/eval-default-repeat.json 2> outputs/eval-default-repeat.jsonl
python -m jetson_edge_vision --config outputs/evaluation-debug.json \
  --video data/sintel-sample.mp4 --output outputs/eval-comparison-repeat.mp4 \
  --confidence 0.5 --warmup 3 --track --track-iou 0.1 \
  --track-max-missed 10 --track-min-hits 3 \
  > outputs/eval-comparison-repeat.json 2> outputs/eval-comparison-repeat.jsonl
```

For the additional clip, substitute `data/phase3b-sintel-street.mp4` and a new output
name with default parameters. The derived clip is local, not distributed in Git.
To reconstruct it from the already local trailer, use OpenCV `VideoCapture`, seek
to frame 480, decode/write exactly 48 frames with mp4v at reported 24 FPS and
854x480, and release both handles. No rescaling occurs; audio is omitted. Validate
decoded geometry/count/FPS and keep the attribution above. Encoded bytes may vary
with codec builds; the recorded sample hash identifies this run's exact input.

Measured local artifacts (all ignored by Git):

- `outputs/phase3b-final-evaluation.json`: complete synthetic grid + fresh main trace.
- `outputs/phase3b-street-evaluation.json`: complete street parameter grid replay.
- `outputs/phase3b-video-comparison.json`: fresh CUDA performance and sequence hashes.
- `outputs/phase3b-sintel-default.{mp4,json,jsonl}` and
  `outputs/phase3b-sintel-comparison.{mp4,json,jsonl}`: two main-clip runs.
- `outputs/phase3b-street-default.{mp4,json,jsonl}`: second local section.
- `outputs/environment-phase3b.json`: successful environment diagnostics.

Only original synthetic source, tests, and authored docs belong in a portfolio
commit. Media, private labels, raw logs, and JSON reports containing local paths
stay in ignored directories unless separately approved for publication.

## Small manual annotation format

This format reviews **detections already in an exact trace**, avoiding an automatic
box/identity matching heuristic inside the scorer. It can measure the custom
identity diagnostics above, category errors, and visible-object misses. It does
not provide independent localization ground truth, AP, or standard MOT scoring.

Example structure only, **not annotations for Sintel**:

```json
{
  "schema_version": 1,
  "reviewed": false,
  "trace_sha256": "<exact 64-character SHA256 of the reviewed JSONL>",
  "frames": [
    {
      "frame_index": 0,
      "reset_before": false,
      "visible_objects": {
        "person-A": {"class_id": 1, "class_name": "person"}
      },
      "truth_ids": ["person-A", null]
    }
  ]
}
```

Here the example frame has exactly two detections: the first belongs to person-A,
the second is a reviewed false/duplicate detection. Include **every** frame in
order, including empty frames. `visible_objects` lists observable physical objects
and their true categories, even when the detector misses them; absent/offscreen
objects are omitted. Preserve physical IDs across a justified disappearance, and
do not infer continuity through an ambiguous cut. Correct category labels must not
copy a known detector misclassification. Each object can label at most one detection
per frame; review duplicate detections as null. Null means reviewed false positive,
not "unknown". Uncertain labels must be resolved before setting `reviewed=true`.

`reset_before=true` explicitly tests a call to `tracker.reset()` and opens a new
identity scope. It is **not** merely a scene label, and it changes the replay's
tracker behavior. Leave it false when evaluating the existing production behavior;
scene reset is not automatic. Clearly identify reset experiments separately.

Keep labels under ignored `data/` or `outputs/`. Once fully reviewed:

```bash
python -m jetson_edge_vision.tracking_evaluation --trace-only \
  --trace outputs/reviewed-trace.jsonl --annotations data/reviewed-labels.json \
  --output outputs/reviewed-evaluation.json
```

The loader requires `schema_version=1`, `reviewed=true`, the exact trace hash,
complete frame coverage, aligned detection indices, type-checked categories, and
one-to-one labels. These checks prevent accidental mismatch; they do not establish
human-label correctness or inter-annotator agreement. Real labels remain absent
in this phase, so real-video accuracy diagnostics remain null.

## Verification, limitations, and next milestone

All **113 tests passed with GPU integrations enabled**, preserving the original
85 tests unchanged. Default discovery passes 109 and skips four GPU integrations.
The new 28 tests cover metric numerators/denominators, switches versus extra-ID
fragmentation, detector/association separation, parameter tradeoffs, scene scopes,
false-track confirmation, absent truth, label leakage prevention, frame coverage,
source-hash binding, report integrity, overwrite protection, no hardware imports,
and replay of a real CUDA trace.

```bash
python -m unittest discover -s tests -v
JEV_GPU_TESTS=1 JEV_TEST_IMAGE=data/dog.jpg JEV_EXPECT_CLASS=dog \
  JEV_TEST_VIDEO=data/sintel-sample.mp4 JEV_VIDEO_EXPECT_CLASS=person \
  JEV_TRACK_VIDEO=data/sintel-sample.mp4 \
  python -m unittest discover -s tests -v
python -m jetson_edge_vision --config configs/default.json --check-environment
```

The environment check again passed CUDA tensor sum 6.0, CUDA NMS `[0]`, OpenCV
resize `[4,4,3]`, and GStreamer support. GPU tests require host NVIDIA access,
existing media/checkpoint, and never download assets or fall back to CPU.

Limits: hand-designed scenarios are not a representative distribution; only two
short excerpts of one animated source were available; there is no reviewed real
identity ground truth, accuracy confidence interval, controlled thermal/power
benchmark, or standard MOT evaluation. Replay timing and video timing are different
measurements. A low error count on these cases does not make counting reliable.

Recommended next milestone, **not started**: obtain user-provided or explicitly
approved representative clips and review labels using the format above. Include
fixed-camera multi-person motion, same-class crossings/occlusion, entering/leaving
and returning, camera motion/cuts, small/distant objects, lighting changes, and
empty-background false positives. Retain source licensing/annotation provenance
privately. Then establish acceptance criteria and decide, separately, whether
motion prediction/global assignment, class stabilization, or detector improvements
are needed. Do not advance to object counting on the present evidence alone.
