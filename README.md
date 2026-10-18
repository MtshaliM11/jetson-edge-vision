# Jetson Edge Vision

Local, GPU-accelerated computer vision for NVIDIA Jetson Orin Nano Super.
The intended system combines detection, tracking, counting, and a local monitoring
interface without cloud services for core processing.

**Status: Phase 3B reproducible tracking evaluation complete; tracker/defaults preserved.**

SSDLite320 MobileNetV3-Large COCO_V1 runs with the existing PyTorch/Torchvision
installation. It produces bounding boxes, COCO class names, confidence scores,
annotated images/videos, and JSON measurements. No additional packages are required.

## Evaluate tracking (no GPU required for replay)

Phase 3B adds a separate standard-library evaluator; production detection,
tracking, video processing, CLI, and defaults are unchanged. It runs **17 labelled
synthetic scenarios × 27 tracking configurations**, with fresh tracker state per
repeat. Reports distinguish detector errors from association diagnostics and
never infer accuracy from unlabelled clips.

After activating the environment and setting `PYTHONPATH` as below:

```bash
mkdir -p outputs
python -m jetson_edge_vision.tracking_evaluation \
  --repeats 10 --output outputs/tracking-evaluation-repeat.json

# Replay an existing saved trace, with its companion pipeline report:
python -m jetson_edge_vision.tracking_evaluation --trace-only \
  --trace outputs/phase3b-sintel-default.jsonl \
  --video-report outputs/phase3b-sintel-default.json \
  --repeats 10 --output outputs/trace-evaluation-repeat.json
```

Use new report filenames; existing outputs are protected. `--baseline-only`
selects the unchanged default configuration. Replay uses cached detection boxes,
strips saved track IDs, and imports no Torch/OpenCV. Its throughput is **CPU replay
throughput, not video FPS**. Synthetic labels never enter the tracker. Optional
reviewed manual annotations can enable labelled trace scoring; without them,
ID-switch, fragmentation-proxy, and false-track accuracy fields are `null`.

The synthetic baseline preserves IDs in ordinary motion, independent objects,
and short detection gaps. It exposes switches at ambiguous crossings, fragmentation
after long gaps/jumps, class-flip effects, stale-ID inheritance after scene changes,
and confirmation of persistent false detections. Two locally available Sintel
sections were also processed on CUDA; no media was downloaded. Both remain
unlabelled animated examples, not an accuracy dataset.

Keep **IoU 0.3 / max missed 5 / minimum hits 2** as the existing evaluation baseline.
Lower IoU, longer retention, and delayed confirmation each help specific scenarios
and hurt others; no general improvement is established. See the
[Phase 3B report](docs/phase3b.md) for definitions, parameter tables, measured
timings, real-video observations, annotation format, and exact reproduction steps.
Counting remains unimplemented.

## Track objects in a video (optional)

Tracking uses an original class-aware greedy IoU baseline on the existing
detections. It requires no new packages and does not replace or retrain the model.
Activate the environment and set `PYTHONPATH` as in the video example below, then:

```bash
mkdir -p outputs
python -m jetson_edge_vision --video data/sintel-sample.mp4 \
  --output outputs/tracked-rerun.mp4 --confidence 0.5 --warmup 3 \
  --track --track-iou 0.3 --track-max-missed 5 --track-min-hits 2 \
  > outputs/tracked-rerun.json

# Same source/options with tracking explicitly disabled:
python -m jetson_edge_vision --video data/sintel-sample.mp4 \
  --output outputs/untracked-rerun.mp4 --confidence 0.5 --warmup 3 \
  --no-track > outputs/untracked-rerun.json
```

Tracking is **off by default**. The existing detection-only command is preserved.
Tracking flags require `--video`; tuning flags also require `--track`.
`--track-iou` must be in (0, 1], `--track-max-missed` a nonnegative integer, and
`--track-min-hits` a positive integer. Defaults are 0.3, 5, and 2 respectively.
Settings are CLI options; the existing JSON device/log-level schema is unchanged.

IDs such as `person #001 0.97` are local to one video session. `#001?` means a
tentative track, confirmed after the configured number of consecutive observations.
IDs/colors remain consistent while associations succeed. Missed tracks retain
their last observed box for association but are not drawn as current detections;
they expire after more than the configured number of missed source frames.

On the 48-frame sample, #001 persisted over frames 0–23; #002 persisted over frames
36–43 and recovered at frame 46 after two misses. The detector incorrectly labelled
the character `bird` at frames 44–45, producing #003. A shot/gap also fragmented
the person into separate IDs. This baseline does not establish tracking accuracy,
handle all crossings/occlusions, or identify people.

Measured tracker updates averaged **0.1035 ms/frame**. The matched comparison
measured 5.27/5.38 processing FPS and 2.67/2.66 full end-to-end FPS with tracking
off/on; timing variation does not demonstrate a speedup. The report includes
average detections per frame, association times, and active-track statistics.
See [algorithm evaluation, lifecycle, benchmark, trace commands, and limits](docs/phase3a.md).

Detection finds/classifies boxes in each frame. Tracking associates those boxes
and assigns temporary IDs. Counting would derive events or quantities from tracks
and geometry; **object counting is not implemented**. Track creation and detection
totals in diagnostic reports are not counts of unique physical objects.

## Detect objects in a video (headless)

Use the existing environment and downloaded Phase 2A checkpoint:

```bash
cd ~/jetson-projects/ai/projects/jetson-edge-vision
source ~/jetson-projects/ai/ai-env/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs

python -m jetson_edge_vision --config configs/default.json \
  --video /path/to/input.mp4 --output outputs/video-detected.mp4 \
  --confidence 0.5 --warmup 3 > outputs/video-report.json

# Short test; select new filenames for each run:
python -m jetson_edge_vision --video /path/to/input.mov \
  --output outputs/short-test.avi --codec MJPG --max-frames 48 \
  --confidence 0.5 > outputs/short-test-report.json
```

MP4, AVI, and MOV inputs are supported when the installed OpenCV can decode their
particular codec. Actual write/decode probes passed for MP4/MOV with `mp4v` and
AVI with `MJPG`; `avc1` also passed an MP4 probe. Defaults are `mp4v` for MP4/MOV
and `MJPG` for AVI. `--codec FOURCC` selects an installed alternative; an
unavailable encoder produces an error suggesting AVI/MJPG, without installing
anything or silently changing formats.

The model loads once per video, warms up once (default: three passes), and performs
one measured inference per frame. Frames are read sequentially with no GUI,
preview, frame skipping, or audio output. Tracking is opt-in. Decoded dimensions and source
FPS are preserved. `--max-frames N` stops after N frames; without it, processing
continues until the decoder ends. `--runs` remains image-only. If source FPS is
invalid/missing, supply `--output-fps 24` (or another known correct value).
Overriding valid source FPS changes output playback duration.

The JSON report separates source playback FPS, synchronized inference latency,
processing-loop FPS, and full end-to-end FPS including initialization and output
verification. Total detections are repeated observations across frames, not unique
objects or a counting feature. The 48-frame, 24 FPS verification clip measured
**4.93 processing FPS**, **180.13 ms mean inference**, and **2.52 full end-to-end
FPS**. These measurements do not establish real-time processing.

Outputs are encoded in temporary storage, fully decoded to verify every frame,
then published without overwriting existing files. Failures before publication
and Ctrl+C discard the staged video and release readers/writers. Choose a new
output filename to repeat a run; shell redirection can still overwrite JSON reports.
Keep videos/reports under ignored `data/` and `outputs/`.

Known limits include omitted audio, constant-frame-rate output, backend-dependent
rotation metadata, and conservative handling of decoder failures. Odd dimensions
and known non-square pixels are rejected to avoid silently changing geometry.
See [Phase 2B architecture, codec evidence, troubleshooting, sample attribution,
and full benchmark](docs/phase2b.md).

## Detect objects in an image

```bash
cd ~/jetson-projects/ai/projects/jetson-edge-vision
source ~/jetson-projects/ai/ai-env/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Explicit download of the approved official checkpoint, needed only once:
python -m jetson_edge_vision --download-weights
mkdir -p outputs

# Choose an output that does not already exist:
python -m jetson_edge_vision --config configs/default.json \
  --image /path/to/image.jpg --output outputs/image-detected.jpg \
  --confidence 0.5 --warmup 3 --runs 10 > outputs/image-report.json
```

Detection never downloads weights automatically. `--download-weights` creates
`models/`, verifies the official SHA256 prefix, and exits. Existing checkpoints
are verified and reused; corrupt files are rejected without replacement.
Use `--weights /path/to/checkpoint.pth` to locate another copy of the same approved
checkpoint. The report records the complete SHA256. No package resolver or
installer is invoked.

JSON results go to stdout; logs go to stderr. Image output parent directories are
created automatically; create report directories before shell redirection. The
input image and existing output images are never overwritten. Image outputs
support JPEG, PNG, BMP, TIFF, and WebP subject to existing codecs. OpenCV decodes
inputs to BGR colour, dropping alpha and expanding grayscale; the detector converts
to RGB float32 in [0, 1]. Boxes are `[x1, y1, x2, y2]` in decoded input-image pixel
coordinates. Labels show scores on a 0–1 scale.

`--confidence` accepts finite values from 0 to 1. `--warmup` accepts zero or more
unmeasured passes; `--runs` requires at least one measured pass (defaults: 3 and
10). Each pass uses the same image. `device` in the existing config now also
selects inference device; the default is CUDA. A deliberately configured CPU run
is supported, but CUDA failures never trigger automatic CPU fallback.

The verified example detected **dog, confidence 0.7512**, with mean inference
**151.25 ms** and total processing **10,701.65 ms**. These are observed FP32
results, not guaranteed performance. See [model evaluation, timing boundaries,
licensing, and full verification evidence](docs/phase2a.md).

## Test image detection

```bash
# Offline tests; actual CUDA integration is skipped by default:
python -m unittest discover -s tests -v

# Requires host GPU access and the already downloaded checkpoint:
JEV_GPU_TESTS=1 python -m unittest discover -s tests -v

# Also verify a known class in your own example image:
JEV_GPU_TESTS=1 JEV_TEST_IMAGE=/path/to/dog.jpg JEV_EXPECT_CLASS=dog \
  python -m unittest discover -s tests -v
```

An explicitly enabled GPU test fails if CUDA or weights are unavailable; it does
not download assets or fall back to CPU. Optional `JEV_WEIGHTS` selects the
checkpoint location. All eight original foundation tests are preserved.

The complete Phase 2B suite passed **57 tests with GPU testing enabled**, including
all 28 existing tests. Default discovery passes 55 and skips two GPU integrations.
Synthetic clips verify video decoding/encoding and mocked frame inference without
the GPU. To also verify the local licensed Sintel excerpt:

```bash
JEV_GPU_TESTS=1 JEV_TEST_VIDEO=data/sintel-sample.mp4 \
  JEV_VIDEO_EXPECT_CLASS=person python -m unittest discover -s tests -v
```

The sample is already present in the verified workspace; tests never download it.
Without `JEV_TEST_VIDEO`, the video GPU test generates its own synthetic clip.

Phase 3A passed **85 tests with GPU testing enabled**, preserving the original 57.
Default discovery passes 82 and skips three CUDA integrations. Synthetic sequences
cover multiple moving objects, association, misses, expiration, reset, categories,
confirmation, deterministic ties, and the known crossing ambiguity. Tracking
integration also requires the existing local Sintel clip:

```bash
JEV_GPU_TESTS=1 JEV_TRACK_VIDEO=data/sintel-sample.mp4 \
  python -m unittest discover -s tests -v
```

That clip must contain overlapping detections across consecutive frames; the
tracking GPU test checks actual ID persistence and never downloads sample media.

Phase 3B passed **113 tests with GPU integrations enabled**, preserving all 85
previous tests unchanged. Default discovery passes 109 and skips four GPU tests.
The new tests verify diagnostic definitions, parameter tradeoffs, trace integrity,
manual-label binding, absent-ground-truth behavior, repeatability, and replay of
an actual CUDA-generated trace. Existing test commands also run the new tests.

## Run the foundation

Use the existing Jetson AI environment. No installation is required:

```bash
cd ~/jetson-projects/ai/projects/jetson-edge-vision
source ~/jetson-projects/ai/ai-env/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m jetson_edge_vision --config configs/default.json
python -m jetson_edge_vision --config configs/default.json --check-environment
python -m unittest discover -s tests -v
```

Startup validates configuration and writes a JSON readiness log to stderr, then
exits. It does not load a model, open a camera, or process media. The diagnostic
command additionally imports the existing AI libraries, runs a tensor sum and
Torchvision NMS on the configured device, checks OpenCV resizing, and prints JSON
to stdout. It returns a nonzero exit code on failure. GPU access must be available
to the process: restricted sandboxes can block Jetson memory devices even when
the host GPU works. CUDA failure never silently falls back to CPU.

Use `--help` for options and `--version` for the package version. Relative config
paths are resolved from the working directory. Omitting `--config` uses defaults.
A CPU diagnostic can be requested by using `"device": "cpu"` in a separate config;
it does not verify GPU functionality. Without an explicit command, startup still
avoids hardware imports, model loading, and image processing.

## Project structure

```text
configs/default.json                 Validated startup settings
src/jetson_edge_vision/
  __init__.py                        Package version
  __main__.py                        Module entry point
  cli.py                             Command orchestration and exit codes
  config.py                          Immutable settings and JSON validation
  logging_utils.py                   Application-scoped structured logging
  diagnostics.py                     Explicit environment checks
  detection.py                       Approved checkpoint and reusable detector
  image_pipeline.py                  Image decoding, annotation, output, timing
  video_pipeline.py                  Sequential video processing and verification
  tracking.py                        Session-local class-aware IoU tracker
  evaluation_scenarios.py            Original labelled synthetic detections
  tracking_evaluation.py             Independent CPU evaluation/report command
 tests/test_foundation.py             Configuration, logging, and CLI tests
 tests/test_detection.py              Offline and opt-in CUDA detection tests
 tests/test_video.py                  Video/codec tests and CUDA integration
 tests/test_tracking.py               Synthetic association and CUDA tracking tests
 tests/test_tracking_evaluation.py    Scoring, replay, annotations, CUDA trace contract
 docs/architecture.md                Module boundaries and phased roadmap
 docs/environment.md                 Dependencies and verification evidence
 docs/phase2a.md                      Model evaluation and GPU measurements
 docs/phase2b.md                      Video behavior, codecs, and GPU measurements
 docs/phase3a.md                      Tracker evaluation, lifecycle, and measurements
 docs/phase3b.md                      Reproducible evaluation methodology and results
pyproject.toml                       Package metadata and optional CLI entry point
```

The source layout makes imports explicit and keeps test files outside the package.
The commands above use `PYTHONPATH` to avoid modifying the existing environment.
The packaged entry point is declared for future installation; this workflow uses
`python -m jetson_edge_vision` and does not require an editable install.

## Configuration and logging

Configuration currently accepts only `device` (`cuda` or `cpu`) and `log_level`
(`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Unknown keys, invalid values,
unreadable files, and malformed JSON fail with informative messages. JSON uses
Python's standard library, avoiding an additional configuration dependency.

Logs contain UTC `timestamp`, `level`, `logger`, and `message`; exception records
also include an `exception` field. Only the application logger is configured.
Repeated initialization does not duplicate output. Persistent logs and event
storage will be added in later phases.

## Environment safeguards

Do not replace the working ARM64 CUDA PyTorch or Torchvision builds. Do not install
`opencv-python` or `opencv-python-headless`: the existing OpenCV preserves
GStreamer integration. No packages or system settings were changed for Phase 1.
See [environment verification](docs/environment.md) for measured checks.

The CSI camera has an unresolved purple/magenta cast. Phase 1 does not open it.
Phase 2A processes images and Phase 2B local video files; camera integration is Phase 5.
The verified `hello-gpu` and `camera-gpu` projects remain untouched.

## Roadmap

1. Foundation: configuration, logs, diagnostics, tests — verified.
2. Phases 2A/2B: PyTorch CUDA image/video-file detection, measured latency,
   annotated output, and tests — verified.
3. Phase 3A: optional IoU tracking — verified. Phase 3B: controlled evaluation and
   parameter replay — complete; representative labelled real-video accuracy remains unestablished.
4. Add tested line/region counting and stored events.
5. Integrate CSI capture and document camera limitations.
6. Add a local API and simple dashboard.
7. Benchmark and evaluate TensorRT/FP16 using measured results.
8. Prepare documentation, licensing, examples, and release checks for GitHub.

See [architecture](docs/architecture.md), [Phase 2A evidence](docs/phase2a.md), and
[Phase 2B evidence](docs/phase2b.md), [Phase 3A evidence](docs/phase3a.md), and
[Phase 3B evaluation](docs/phase3b.md).
Counting and subsequent phases have not been started.

## Repository hygiene and licensing

Git is initialized locally on `main`; no remote, commit, or push is created by
Phase 1. `.gitignore` excludes environment files, caches, typical model weights,
video recordings, logs, local data, and database outputs. Review `git diff` and
`git status` before staging: ignore rules cannot detect every credential or
private image. Keep recordings and downloaded assets under ignored `data/` and
weights under ignored `models/`.

An open-source license has not yet been selected; this is not yet an open-source
release. Select a project license and review model, weight, and dependency
licenses before publication. Torchvision code is BSD-3-Clause, but its pretrained
weights have a separate dataset-derived licensing caveat: code licensing alone
does not establish unrestricted rights to COCO-trained weights. See the
[licensing limitations](docs/phase2a.md#licensing-limitations). Weights, example
images, reports, and annotated images remain local under ignored directories.
