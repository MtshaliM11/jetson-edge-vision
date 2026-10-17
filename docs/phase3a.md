# Phase 3A: optional multi-object tracking

Implemented and verified on 2026-10-17. This phase associates the existing model's
video detections with anonymous session-local IDs. It does not replace/retrain the
detector, identify people, implement biometrics, or add object counting. No new
packages, model weights, drivers, system settings, or installation changes were made.

## Algorithm evaluation

Resource comparisons below are architectural estimates, not Jetson benchmarks of
all four algorithms. Their accuracy depends on detections, scene, and configuration.

| Approach | Complexity, dependencies, CPU/memory | Reliability and integration | Reference licensing |
| --- | --- | --- | --- |
| Class-aware greedy IoU baseline | Small; standard-library pairwise overlap matching and candidate sort; last box/state per track, CPU only | Direct fit to existing detection dictionaries. No prediction, appearance, or camera-motion correction; weak during rapid motion, crossings, long gaps, or cuts | Original project code, no external tracker code copied; project license remains unselected |
| SORT | Moderate; Kalman state/covariance plus global assignment. Reference requirements include FilterPy, lap, scikit-image; small per-track state but more computation than last-box matching | Motion prediction helps smooth trajectories; still limited under occlusion. Can adapt current boxes without replacing detector, but requires additional algorithm/solver integration | Official reference implementation GPL-3.0 |
| ByteTrack | More lifecycle/matching logic; Kalman prediction and high/low-confidence association stages. Reference distribution includes lap, FilterPy, other training/evaluation dependencies | Uses low-score boxes to recover matches. Current detector API filters below the user threshold, so adopting this faithfully needs an input/threshold contract change, not just attaching IDs | Official reference implementation MIT; dependency terms separate |
| BoT-SORT | Largest scope: motion state, camera-motion compensation, optional appearance embeddings/ReID network; more CPU work and potentially model memory/GPU work | Targets harder camera-motion/association cases; more machinery and adaptation to COCO multi-class output. Reference environment includes SciPy, lap, FilterPy and other packages | Official reference implementation MIT; bundled components/weights/dependencies need separate review |

Primary sources: [SORT paper](https://arxiv.org/abs/1602.00763),
[SORT repository/license](https://github.com/abewley/sort),
[SORT requirements](https://github.com/abewley/sort/blob/master/requirements.txt),
[ByteTrack paper](https://arxiv.org/abs/2110.06864),
[ByteTrack license](https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE),
[ByteTrack requirements](https://github.com/FoundationVision/ByteTrack/blob/main/requirements.txt),
[BoT-SORT paper](https://arxiv.org/abs/2206.14651),
[BoT-SORT license](https://github.com/NirAharon/BoT-SORT/blob/main/LICENSE), and
[BoT-SORT requirements](https://github.com/NirAharon/BoT-SORT/blob/main/requirements.txt).
Full reference requirements include training/demo packages that a carefully isolated
tracker integration might not need. None were installed or imported here.

Selected: an original **class-aware greedy IoU tracker**, as the smallest baseline
compatible with the current API and dependency constraints. It is not SORT (no
Kalman/Hungarian assignment), ByteTrack (no low-score second association), or
BoT-SORT (no motion compensation/appearance model). Reference code was not copied.
The model/weight [licensing limitations](phase2a.md#licensing-limitations) and
project-license decision remain unchanged.

## Lifecycle and association contract

`tracking.py` is hardware-independent. One `IoUTracker` belongs to each enabled
video session, with immutable `TrackingConfig` and `Track` records. It accepts the
existing detection dictionaries: class ID/name, confidence, and original-frame
`box_xyxy`. Inputs are validated before changing state; malformed/nonfinite boxes,
nonpositive areas, invalid categories, and out-of-range scores fail clearly. The
input records are not mutated. No NumPy, Torch, or OpenCV imports are needed.

Each processed source frame, including one with no detections:

1. Compute continuous-coordinate intersection-over-union against each retained
   track's last observed box. Both COCO ID and class name must agree.
2. Keep candidate pairs with `IoU >= threshold`. Sort by descending IoU, then
   ascending track ID, then input detection index. Greedily consume each track and
   detection at most once. Matching is deterministic for the same ordered inputs,
   but does not solve a global assignment optimum.
3. Matched tracks keep their ID and update box, score, observation count, and last
   frame. A return after a miss begins a new consecutive-hit streak.
4. Unmatched tracks increment `missed`, reset the hit streak, and retain their last
   box. Delete them when `missed > max_missed`. With default 5, exactly five missed
   frames are tolerated and the following frame can still match; a sixth unmatched
   frame removes the track. With 0, the first unmatched frame removes it.
5. Unmatched detections receive new, monotonically increasing IDs starting at 1.
   IDs are never recycled within the session. `reset()` or a new session starts
   again at 1; equal integers in separate videos imply no identity connection.

Tracks start tentative unless `min_hits=1`. Confirmation needs `min_hits`
consecutive observations, including creation. Confirmation then persists through
allowed misses. Tentative tracks have the same inactivity expiry as confirmed
ones. Confirmation marks status; it does not hide detections or suppress their IDs.
All current detections remain visible, with a `?` suffix for tentative IDs.

Returned records preserve detector order and add `track_id`/`track_confirmed`.
Only current observations are returned/rendered; missed tracks are not predicted
or drawn. IDs use at least three digits; colors come from a deterministic six-color
palette (colors repeat after six IDs). There is no trajectory history or cross-video
state. Miss limits use processed source frames, not wall-clock time or processing FPS.

With T retained tracks, D detections, and K eligible pairs, matching costs
O(TD + K log K), with O(T + D + K) transient/state storage. The state retains only
last observations and counters, not frames, appearances, or GPU tensors. Video
statistics additionally retain one float64 per frame for exact association median,
alongside the existing inference timing samples. Large crowded scenes can cost
more CPU/memory; this clip is not a stress or peak-memory benchmark.

## CLI and configuration

```bash
cd ~/jetson-projects/ai/projects/jetson-edge-vision
source ~/jetson-projects/ai/ai-env/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs

# Tracking off (also the default):
python -m jetson_edge_vision --video data/sintel-sample.mp4 \
  --output outputs/detection-repeat.mp4 --confidence 0.5 --warmup 3 \
  --no-track > outputs/detection-repeat.json

# Same input/settings with tracking on:
python -m jetson_edge_vision --video data/sintel-sample.mp4 \
  --output outputs/tracking-repeat.mp4 --confidence 0.5 --warmup 3 \
  --track --track-iou 0.3 --track-max-missed 5 --track-min-hits 2 \
  > outputs/tracking-repeat.json
```

Select new filenames for repeats; existing videos are never overwritten. Shell
redirection can overwrite reports. `--max-frames N`, `--codec`, `--output-fps`, and
all existing video controls remain available. Tracking is video-only; `--track` and
`--no-track` are mutually exclusive. Tuning flags require `--track` so ignored
settings cannot mislead the user. Defaults and valid values:

| Flag | Default | Valid values |
| --- | --- | --- |
| `--track` / `--no-track` | disabled | Explicit switch, video only |
| `--track-iou` | 0.3 | Finite number in (0, 1] |
| `--track-max-missed` | 5 | Integer >= 0 |
| `--track-min-hits` | 2 | Integer >= 1 |

The original JSON config schema stays `device`/`log_level`; these tracking settings
are CLI options. Python callers pass `tracking=True` and an optional immutable
`TrackingConfig` to `detect_video`. Default-disabled calls do not instantiate a
tracker or perform matching. Detector/weight loading and warm-up are unchanged.

## Actual CUDA comparison

Sequential fresh-process runs used the same 48-frame `data/sintel-sample.mp4`,
854x480, 24 FPS, mp4v output, confidence 0.5, warm-up 3, default CUDA, INFO logging,
and the same checkpoint. Runtime: Orin `cuda:0`, Torch 2.8.0, Torchvision 0.23.0,
CUDA 12.6, FP32. No power, clock, thermal, or package settings were changed.
The commands above reproduce the settings with new output filenames.

| Measurement | Tracking off | Tracking on |
| --- | --- | --- |
| Frames | 48 | 48 |
| Detection observations | 35 | 35 |
| Average detections/frame | 0.7292 | 0.7292 |
| Mean inference | 168.75 ms | 164.93 ms |
| Median inference | 166.83 ms | 160.19 ms |
| Tracker update mean | Not run | 0.1035 ms/frame |
| Tracker update median | Not run | 0.1105 ms/frame |
| Tracker update maximum | Not run | 0.1936 ms/frame |
| Tracker update total | 0 | 4.9674 ms |
| Processing-loop FPS | 5.27 | 5.38 |
| Full end-to-end FPS | 2.67 | 2.66 |
| Total processing | 17,953.07 ms | 18,057.10 ms |
| Active retained tracks at end | 0 | 2 (both confirmed) |
| Visible tracks at last frame | 0 | 0 |
| Peak retained tracks | 0 | 2 |
| Track IDs created / expired | 0 / 0 | 3 / 1 |
| Matched observations | 0 | 32 |

Both videos were fully decoded before publication: 48 frames at the original
dimensions and FPS. Actual artifacts, all ignored by Git:

- Detection video: `outputs/phase3a-detection.mp4`; report: `outputs/phase3a-disabled.json`.
- Tracking video: `outputs/phase3a-tracked.mp4`; report: `outputs/phase3a-enabled.json`.
- Comparison: `outputs/phase3a-comparison.json`; visual inspection: `outputs/phase3a-preview.jpg`.

Association timing surrounds `tracker.update()`: validation/copying, overlap
matching, lifecycle updates, and output-record creation. It excludes active-state
summary collection, rendering, and logging. Full loop timing includes all of these.
It is CPU timing, separate from the detector's existing synchronized CUDA-call
latency. Streaming/full-throughput boundaries are unchanged from
[Phase 2B](phase2b.md#performance-boundaries). The full total includes startup,
warm-up, and output verification; neither metric is source playback FPS.

Tracking-on inference happened to be faster in this pair, despite added tracker
work. This does not establish a speedup: uncontrolled runtime variation is much
larger than the measured 4.97 ms of association. The 104.04 ms difference in full
totals is not an isolated tracking-cost measurement. Neither result supports
real-time processing of the 24 FPS source. Historical Phase 2A/2B baselines remain
unchanged, not replaced by these different runs.

## Trace verification and observed failures

A separate DEBUG run (not used for the performance comparison) emitted one
`Tracking frame: {JSON}` message per frame through the existing structured logger.
Only current observations are included. To reproduce with a local debug config:

```bash
cat > outputs/tracking-debug.json <<'JSON'
{"device": "cuda", "log_level": "DEBUG"}
JSON
python -m jetson_edge_vision --config outputs/tracking-debug.json \
  --video data/sintel-sample.mp4 --output outputs/tracking-debug-repeat.mp4 \
  --confidence 0.5 --warmup 3 --track \
  --track-iou 0.3 --track-max-missed 5 --track-min-hits 2 \
  > outputs/tracking-debug-repeat.json 2> outputs/tracking-debug-repeat.jsonl
```

The outer JSON log's `message` contains `Tracking frame: ` followed by a JSON
object with zero-based `frame_index` and current detection/ID/status records.
INFO logging omits this trace and its serialization cost. No track log is stored
unless stderr is redirected; the application retains no per-frame history.

Observed trace (`outputs/phase3a-track-trace.jsonl`, summarized locally in
`outputs/phase3a-track-summary.json`):

- `person #001`: visible frames 0–23, confirmed from frame 1; same ID throughout.
  It expires after missed frames 24–29. A different shot later creates a new ID.
- `person #002`: frames 36–43 and 46, confirmed from frame 37. It recovers the same
  ID at frame 46 after misses at 44–45; no stale box is drawn during those misses.
- `bird #003`: frames 44–45, confirmed at 45. Visual inspection shows the detector
  misclassifying the animated character as a bird; strict class matching therefore
  creates a different ID. At 46 the detector returns to person and matches #002.
- Frame 47 has no detections, but #002 and #003 remain in retained state. This
  explains `active_tracks=2` and `visible_tracks_last_frame=0`.

Decoded frames 0, 12, 36, 44, 45, and 46 were visually inspected: boxes, COCO
labels, scores, padded IDs, tentative markers, and stable colors are readable.
The recorded class/ID change at 44–45 is a real observed failure mode. Fragmentation
across the shot/gap is another limitation. No other within-segment ID switch was
observed in this short trace, but no identity ground truth or ID-switch metric
was established. A synthetic crossing test explicitly demonstrates possible ID
swapping when overlap evidence is ambiguous. A single animated sample cannot
establish tracking accuracy, robustness, or physical identity continuity.

Sample attribution/license and preparation are unchanged from
[Phase 2B](phase2b.md#real-gpu-verification-and-attribution): Sintel trailer,
© Blender Foundation, Creative Commons Attribution 3.0. The derived videos add
tracking annotations; sample/video/log/report files remain local and ignored.

## Detection, tracking, and counting

Detection outputs boxes, categories, and scores for individual frames. Tracking
adds associations and temporary session IDs over time. Counting would use tracks
and configured geometry/events to infer quantities such as line crossings or
entries/exits. Counting is not implemented. `tracks_created`, active-state counts,
and detection observations are diagnostic statistics, not counts of unique
physical people/objects. Duplicate hypotheses, misclassifications, fragmentation,
and ID switches prevent interpreting them that way.

## Tests and remaining limitations

All **85 tests passed with GPU testing enabled**, including all 57 original tests
unchanged. Default discovery passes 82 and skips three CUDA integrations. New
tests cover creation, state updates, two moving same-class objects over seven
frames with reversed detection order, empty frames, exact expiry boundaries,
return after misses, reset, class separation, threshold inclusivity, deterministic
ties, malformed inputs, confirmation, crossing ambiguity, annotations, disabled
tracking, per-session pipeline state, metrics, CLI validation, and DEBUG traces.
The actual-CUDA tracker integration decodes a three-frame output and checks a
shared ID across consecutive detected frames; it never mocks the detector.

```bash
python -m unittest discover -s tests -v
JEV_GPU_TESTS=1 JEV_TEST_IMAGE=data/dog.jpg JEV_EXPECT_CLASS=dog \
  JEV_TEST_VIDEO=data/sintel-sample.mp4 JEV_VIDEO_EXPECT_CLASS=person \
  JEV_TRACK_VIDEO=data/sintel-sample.mp4 \
  python -m unittest discover -s tests -v
python -m jetson_edge_vision --config configs/default.json --check-environment
```

GPU integration requires host NVIDIA access, the existing approved checkpoint,
and the local Sintel sample (or `JEV_TRACK_VIDEO` with consecutive detectable
objects). Explicit GPU opt-in fails rather than skipping missing media/CUDA or
falling back to CPU. The original tensor/NMS/OpenCV diagnostic passed again.

Limits: no motion prediction, global assignment optimum, appearance features,
camera-motion/scene-cut correction, or recovery after expiration. Fast movement
without sufficient overlap, occlusion, same-class crossings, duplicated detections,
and category flips can fragment or switch IDs. Retained state can match a different
same-class object occupying the old box. Confirmation is not proof of correctness.
Output/codec/audio/rotation/EOF restrictions from Phase 2B still apply.

Recommended next milestone: evaluate detector/tracker behavior on representative
annotated real-world clips, choose acceptable thresholds/ID continuity, and decide
whether motion prediction/global assignment is needed. Counting, camera capture,
TensorRT, and the dashboard require separate authorization and have not started.
