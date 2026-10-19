# Phase 4A — Multi-object directional line counting

Phase 4A adds optional counting after the existing class-aware greedy IoU tracker.
The detector, weights, tracker, configuration schema, and tracking defaults
(**0.3 IoU / 5 maximum missed frames / 2 confirmation hits**) are unchanged.
No packages, media, drivers, CUDA libraries, or model weights were installed or
changed. Development stays on `feature/phase4a-line-counting`; no commit, merge,
or push is part of verification.

## Architecture and state

```text
OpenCV decode -> CUDA detector -> optional IoUTracker -> optional LineCounter
                                                            |
                                                   totals + event records
                                                            |
                                          box/ID + line/count overlay -> encode
                                                            |
                                              full output decode check + JSON
```

`counting.py` uses only the standard library and the existing immutable `Track`
record. It neither runs inference nor assigns IDs. `CountingConfig` validates
one line and its deadband. `LineCounter.update(tracker.tracks, frame_index, time)`
consumes the complete active snapshot, including missed tracks, once per frame.
Each video call creates its own tracker/counter; warm-up advances neither.
`counting_annotation.py` draws on the existing annotated frame copy. All previous
image, detection-only video, and tracking-only video options remain available.

Each ID has independent category/first-frame identity, previous observed center,
stable side, and finite-intersection state. Every eligible track is processed;
there is no configured object limit. Sorted IDs give deterministic event ordering
when several objects cross in the same frame. Runtime capacity still depends on
hardware, scene complexity, and detector/tracker behavior. The detector retains
its existing built-in postprocessing limits; counting adds no new cap.

## Geometry, direction, and duplicate suppression

Coordinates are normalized in `[0,1]`. Pixel endpoints use `x*(width-1)` and
`y*(height-1)` from the decoded frame, with integer dimensions at least two.
Coordinates must be finite numbers, endpoints must differ, and pixel hysteresis
must be finite and nonnegative. Default line: `(0.5,0.1)-(0.5,0.9)`; default
hysteresis: **2 pixels**, independent of the tracker settings.

For endpoints P and Q, signed perpendicular distance is
`cross(Q-P, center-P) / length(Q-P)`. **A is negative; B is positive.** Image y
increases downwards. For the default top-to-bottom vertical line, A is on the
right and B is on the left: right-to-left is `A->B`. For a left-to-right horizontal
line, top-to-bottom is `A->B`. Reversing endpoints reverses the direction labels.
These are geometric labels, not automatically entrance/exit or compass directions.

The center must reach strictly beyond the deadband on each side. Every pair of
consecutive observed centers is tested against the **finite closed line segment**;
endpoint touches are included, collinear travel alone is insufficient. A pending
intersection is carried through the deadband until a stable side is reached.
Returning to the original stable side clears it. This uses the observed polyline,
not a chord between distant stable observations that could falsely cross a line
when the object actually went around an endpoint.

An event is emitted when the opposite stable side is reached and a qualifying
intersection has occurred. Its position, frame, and time describe that confirming
observation, not an interpolated contact instant. The opposite stable side then
re-arms the track for a genuine reverse crossing. Same-side observations, stationary
centers, and deadband jitter do not repeatedly count. Small oscillations beyond
both deadband boundaries can still count; choose a suitable deadband for the scene.

Only currently observed **confirmed** tracks count. Tentative, stale, or missed
observations clear that ID's motion history. An absent ID removes its state;
changed category or first-frame identity starts fresh. Frame indices must increase
strictly; duplicates/out-of-order updates fail before mutation, and skipped updates
break continuity. Temporary disappearance preserves existing totals but **does not
infer a crossing through the gap**. Reset both tracker and counter when restarting
a session, because tracker reset reuses IDs and frame indices.

## Events, totals, and report fields

Each event contains `event_id` (line ID plus session sequence), `track_id`,
`class_id`, `class_name`, `frame_index` (zero based), `direction`, `line_id`,
`position` (pixel center), and `video_timestamp_seconds`. Multiple events may
share a frame/time; they never share an event ID within one counter session.
IDs are session-local, not globally unique identifiers.

Time is `frame_index / source_fps`, or null if source FPS is unavailable. It is
estimated video time, never wall-clock time; an output FPS override does not
change source-time estimates. Variable frame-rate timestamps are not preserved.

The JSON `counting` section contains enabled status, normalized/pixel line,
hysteresis, total crossings, `by_direction`, `by_category` (string COCO ID keys,
category name, total and both directions), every event, processing latency
(measured frames, total/mean/median/max ms), and known limitations. Disabled mode
reports `enabled: false` and zero measured counting frames. Existing inference,
tracking, detection-observation totals, and performance fields are preserved.
All detector-provided COCO categories are supported without a counter allowlist;
categories appear when first counted. Unseen categories implicitly have zero counts.

**Detection observations, active tracks, IDs created, and crossings are different
measurements.** One object can generate many detections, one track, and multiple
crossings after reversals. No unique-physical-object total is claimed.

The overlay displays boxes, scores, IDs/confirmation, line, total events, both
directions, and category-direction totals. Its panel is bounded to about one-third
of normal frame height. Crowded scenes show a category-overflow notice; JSON
retains every category. Tiny frames may show only aggregate totals. Line and box
parts under the panel may be obscured.

The pipeline retains no full video frames or GPU tensors as history. Active
counting state is pruned with track lifecycle. Requested event records grow with
crossing count, and exact-median latency arrays grow with processed frames.
Counting latency includes snapshot access, geometry, state, event/totals updates;
it excludes drawing and final report serialization. Loop FPS includes inference,
CPU transfers/filtering, tracking, counting, annotation, encoding and decoding of
following frames. End-to-end also includes startup, loading, warm-up, staging and
full output decode verification; JSON printing is outside that timer.

## Reproduction

Use the existing Jetson environment and local media. No download is needed.
Use new output names on subsequent runs. Shell `noclobber` protects saved JSON;
the application already rejects existing video destinations and publishes without
replacement after full verification. Reports remain stdout JSON, as in Phase 3B.

```bash
cd ~/jetson-projects/ai/projects/jetson-edge-vision
source ~/jetson-projects/ai/ai-env/bin/activate
export PYTHONPATH=src
set -o noclobber
mkdir -p outputs

# Detection only:
python -m jetson_edge_vision --video data/sintel-sample.mp4 \
  --output outputs/phase4a-detection.mp4 > outputs/phase4a-detection.json

# Tracking only, unchanged defaults:
python -m jetson_edge_vision --video data/sintel-sample.mp4 --track \
  --output outputs/phase4a-tracking.mp4 > outputs/phase4a-tracking.json

# Counting; --count uses the default line, --count-line also enables counting:
python -m jetson_edge_vision --video data/sintel-sample.mp4 --track \
  --count-line 0.4,0.1,0.4,0.9 --count-hysteresis 2 \
  --output outputs/phase4a-counting.mp4 > outputs/phase4a-counting.json

python -m unittest discover -s tests -p test_counting.py -v
python -m unittest discover -s tests -v
JEV_GPU_TESTS=1 JEV_TEST_IMAGE=data/dog.jpg JEV_EXPECT_CLASS=dog \
  JEV_TEST_VIDEO=data/sintel-sample.mp4 JEV_VIDEO_EXPECT_CLASS=person \
  JEV_TRACK_VIDEO=data/sintel-sample.mp4 \
  python -m unittest discover -s tests -v
python -m jetson_edge_vision --check-environment
```

Counting requires `--video --track`; `--count-hysteresis` requires `--count` or
`--count-line`. Invalid geometry is rejected before opening input/loading weights.
The JSON application configuration remains unchanged; these controls are CLI/API
options. Python API: `detect_video(..., tracking=True,
counting_config=CountingConfig(line=(.4,.1,.4,.9)))`.

## Automated verification

`tests/test_counting.py` adds 41 tests, including one opt-in CUDA test, to the
unchanged 113 prior tests. Known trajectories verify exact car/person totals,
three cars plus two people in one frame, overlapping independent IDs, opposite
directions, reversals, unrelated COCO categories, finite endpoints, polyline paths
around endpoints, jitter, tentative/confirmed states, disappearance, expiration,
ID changes, category changes, reset, duplicate frames, skipped frames, timestamps,
invalid input, and several resolutions/orientations.

A mocked detector plus real OpenCV/IoU tracker/video writer verifies all three
pipeline modes: **25 observations, five tracks, five events on frame 3**, with
three car and two person events, followed by successful five-frame output decode.
The mock is explicitly not CUDA inference. The separate CUDA test loads the real
model and processes three real local-video frames; it does not require fabricated
crossings in that short interval. Synthetic tests establish deterministic logic,
not real-world counting accuracy.

## Known limits and next milestone

The original tracker can switch IDs, inherit stale identities, fragment tracks,
or change categories. A new ID does not inherit crossing history, but switches
can still miss or duplicate physical crossings. Geometry cannot repair detector
misclassification or assign physical identities. Occlusion/gap policy intentionally
favors avoiding speculative events over recovering missed crossings. Centers
approximate objects; high speed, low frame rate, curved motion between frames,
shot cuts, endpoint placement and perspective can cause errors. Hysteresis is a
pixel setting requiring scene-specific evaluation.

Local animated trailer clips have limited object/category diversity and no
reviewed crossing ground truth. Measured events are software observations, not
validated entries/exits. No real-time, deployment accuracy, or camera readiness
claim follows from this phase. Prior phase limitations and benchmarks still apply.

Recommended next milestone: review representative local clips with independent
crossing labels, then measure directional per-category precision/recall and failure
cases before selecting further tracking or counting changes. Phase 4B, regions,
TensorRT, live CSI, databases, and a dashboard are not started here.

## Actual Jetson CUDA verification and benchmarks — 2026-10-19

The host environment diagnostic passed CUDA tensor sum 6.0, CUDA NMS `[0]`,
OpenCV resize `[4,4,3]` and GStreamer support. Inference used FP32 `cuda:0` on
`Orin`, PyTorch 2.8.0 / Torchvision 0.23.0 / CUDA 12.6 and the existing approved
checkpoint. Sandbox NVIDIA access failed; host execution was explicitly authorized.

Two sequential repetitions compared tracking against tracking plus counting on
each existing 48-frame clip, alternating order on repetition two. All calls used
confidence 0.5, three warm-ups, unchanged tracker defaults, the same resolution
and codec, and a 48-frame limit. Counting used line `(0.4,0.1)-(0.4,0.9)` and
2-pixel hysteresis. The line was selected by inspecting existing saved detections;
these are functional demonstrations, not a blind accuracy evaluation. No GPU
workloads from this verification ran concurrently. Clocks, thermals, cache state
and other system activity were not controlled.

| Run | Frames | Inference ms | Tracking ms | Counting ms | Loop FPS | End-to-end FPS | Crossings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sintel-1-tracking | 48 | 170.84 | 0.1120 | — | 5.18 | 2.64 | disabled |
| sintel-1-counting | 48 | 168.32 | 0.1108 | 0.0590 | 5.23 | 3.76 | 2 |
| sintel-2-counting | 48 | 171.22 | 0.1106 | 0.0607 | 5.16 | 3.60 | 2 |
| sintel-2-tracking | 48 | 166.56 | 0.1310 | — | 5.34 | 3.80 | disabled |
| street-1-tracking | 48 | 175.12 | 0.0680 | — | 5.09 | 3.74 | disabled |
| street-1-counting | 48 | 169.53 | 0.0672 | 0.0384 | 5.24 | 3.77 | 0 |
| street-2-counting | 48 | 171.39 | 0.0708 | 0.0425 | 5.17 | 3.86 | 0 |
| street-2-tracking | 48 | 172.70 | 0.0715 | — | 5.16 | 3.70 | disabled |

Per-frame times above are means. All eight encoded outputs fully decoded **48
frames at 854×480, 24 FPS**. Aggregate across these runs: 384 frames, with four
crossing records across the two separate main-clip counting sessions (two per
session). Do not add repeat runs to estimate physical-object counts. Tracking
runs had counting disabled, rather than a measured count of zero.

Counting updates cost **0.0384–0.0607 ms/frame** on these sparse clips, small
relative to **166.56–175.12 ms** synchronized GPU inference. Counting-on loop FPS
ranged **5.16–5.24**, versus **5.09–5.34** tracking-only. This overlap and the
inference/startup variation do not establish a speedup or a statistically reliable
net slowdown. The first tracking run had longer startup; it is retained rather
than discarded. These results do not support real-time 24 FPS operation or
performance extrapolation to crowded scenes. Overlay rendering and final event
serialization are outside the counting-update timer. The compact overlay was
subsequently moved from upper left to upper right to preserve the sample's track
labels; the comparison measures the original panel placement. Final CUDA output
verification also checks the revised placement.

Main input SHA256:
`25b46296472622e699959153fc79b021aecba83a5dc0472d4e0f45b3793dfba8`.
Street input SHA256:
`888e98d088f762af9d07f809e61ca31049e499e7789f49501b4dcfe5965ad31c`.
Both main counting sessions generated identical observed events:

| Event | Track | COCO category | Frame | Direction | Pixel center | Estimated video seconds |
| --- | ---: | --- | ---: | --- | --- | ---: |
| line-1:000001 | 1 | 1 / person | 12 | A->B | (336.6460, 235.7689) | 0.5 |
| line-1:000002 | 1 | 1 / person | 20 | B->A | (349.5010, 241.8312) | 0.833333 |

Thus the main report has person=2, A->B=1 and B->A=1. It contains 33 person and
two bird **detection observations**, with peak retained active tracks=2; those
figures are not two reviewed simultaneous physical objects. Street has zero
qualifying events. Neither clip proves multi-category real-video counting accuracy.

A separate **controlled synthetic CUDA stimulus** was created entirely from local
Sintel pixels: first-frame crop `[y=40:435,x=90:605]`, resized to 220×180, copied
at y=20 and y=270 on a 640×480 black frame. For frames 0..31, x=`320-6*frame`.
MJPG input runs at 12 FPS; both copies move right-to-left across the default line.
Real CUDA inference (no mock) produced **two independent person events on frame
18**, track IDs 1 and 2, both `A->B`, estimated time **1.5 seconds**. All 32 output
frames decoded. This exercises simultaneous events through actual detection,
tracking, counting, CLI JSON, and video encoding. The detector produced an
oversized box spanning both tiles and a short-lived extra track (three tracks
created, one expired, two confirmed/visible at the end). These artifacts prevent
interpreting the stimulus as reliable physical identity assessment. Initial run:
157.53 ms inference, 0.1834 ms tracking, 0.0845 ms counting, 5.72 loop FPS.
This artificial workload is separate from the paired natural-video benchmarks.

Generated evidence (all ignored by Git):

- `outputs/phase4a/comparison.json`: eight-run comparison.
- `outputs/phase4a/{sintel,street}-{1,2}-{tracking,counting}.{mp4,json}`:
  actual benchmark videos and complete reports, including input hashes.
- `outputs/phase4a/synthetic-two-person.{mp4,json}`: initial controlled CUDA run.
- `outputs/phase4a/final-tests.log`: final GPU-enabled test log.
- `outputs/phase4a_benchmark.py` and `outputs/phase4a_make_composite.py`:
  local verification helpers, intentionally outside versioned runtime code.

Exact paired-run reproduction (use a new directory if it already exists):

```bash
source ~/jetson-projects/ai/ai-env/bin/activate
export PYTHONPATH=src
set -o noclobber
mkdir outputs/phase4a-reproduction
for clip in sintel street; do
  input=data/sintel-sample.mp4
  if [ "$clip" = street ]; then input=data/phase3b-sintel-street.mp4; fi
  for repeat in 1 2; do
    modes="tracking counting"
    if [ "$repeat" = 2 ]; then modes="counting tracking"; fi
    for mode in $modes; do
      count_args=()
      if [ "$mode" = counting ]; then
        count_args=(--count-line 0.4,0.1,0.4,0.9 --count-hysteresis 2)
      fi
      python -m jetson_edge_vision --video "$input" --track \
        --confidence 0.5 --warmup 3 --max-frames 48 "${count_args[@]}" \
        --output "outputs/phase4a-reproduction/$clip-$repeat-$mode.mp4" \
        > "outputs/phase4a-reproduction/$clip-$repeat-$mode.json"
    done
  done
done
```

The initial measurements reused a process and loaded a new model for every call;
these CLI commands also import the Python application for each call. This adds
startup variation to end-to-end FPS; compare the separately bounded loop and
update measurements. Existing local verification helpers reproduce the original
single-process sequence when available.

Controlled two-object stimulus reproduction, using only the existing source:

```bash
python - <<'PY'
from pathlib import Path
import cv2
import numpy as np
path = Path('data/phase4a-synthetic-two-person.avi')
if path.exists():
    raise FileExistsError(path)
cap = cv2.VideoCapture('data/sintel-sample.mp4')
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError('Local source could not be decoded')
tile = cv2.resize(frame[40:435, 90:605], (220, 180))
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 12., (640, 480))
if not writer.isOpened():
    raise RuntimeError('MJPG writer unavailable')
try:
    for index in range(32):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        x = 320 - 6*index
        for y in (20, 270):
            canvas[y:y+180, x:x+220] = tile
        writer.write(canvas)
finally:
    writer.release()
PY
python -m jetson_edge_vision --video data/phase4a-synthetic-two-person.avi \
  --track --count --warmup 3 --output outputs/two-person-reproduction.mp4 \
  > outputs/two-person-reproduction.json
```

This derivative uses the same local Sintel attribution/terms as the input. It
contains duplicated image content and artificial translation; it is explicitly
separate from natural-video observations and field validation.

Final revision checks: **154 tests passed, zero skips, in 27.922 seconds** with
CUDA integrations enabled (`outputs/phase4a/final-tests.log`). Final CUDA renders
with the upper-right panel preserved both natural-clip events and both simultaneous
synthetic events. `final-sintel.mp4` decoded 48 frames at 854×480/24 FPS, measured
5.2340 loop FPS / 2.6479 end-to-end FPS and 0.0584 ms counting updates.
`final-two-person.mp4` decoded 32 frames at 640×480/12 FPS, measured 5.5848 loop
FPS / 2.2091 end-to-end FPS and 0.0822 ms counting updates. Their companion JSON
files passed event/direction/category-total consistency checks. Frame previews
`final-sintel-preview.jpg` and `final-two-person-preview.jpg` were decoded and
visually inspected; the revised panel preserves the sample track labels/scores.
These final renders are additional functional checks, separate from the eight-run
paired comparison. `git diff --check` also passed.
