# Jetson Edge Vision

Local, GPU-accelerated computer vision for NVIDIA Jetson Orin Nano Super.
The intended system combines detection, tracking, counting, and a local monitoring
interface without cloud services for core processing.

**Status: Phase 2A image detection verified on the Orin GPU.**

SSDLite320 MobileNetV3-Large COCO_V1 runs with the existing PyTorch/Torchvision
installation. It produces bounding boxes, COCO class names, confidence scores,
an annotated image, and JSON measurements. No additional packages are required.

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
 tests/test_foundation.py             Configuration, logging, and CLI tests
 tests/test_detection.py              Offline and opt-in CUDA detection tests
 docs/architecture.md                Module boundaries and phased roadmap
 docs/environment.md                 Dependencies and verification evidence
 docs/phase2a.md                      Model evaluation and GPU measurements
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
Phase 2A processes image files only; camera integration is Phase 5.
The verified `hello-gpu` and `camera-gpu` projects remain untouched.

## Roadmap

1. Foundation: configuration, logs, diagnostics, tests — verified.
2. Phase 2A: PyTorch CUDA image detection, measured latency, annotated output,
   and tests — verified. Video-file processing remains a separate future increment.
3. Add multi-object tracking with temporary, session-scoped IDs.
4. Add tested line/region counting and stored events.
5. Integrate CSI capture and document camera limitations.
6. Add a local API and simple dashboard.
7. Benchmark and evaluate TensorRT/FP16 using measured results.
8. Prepare documentation, licensing, examples, and release checks for GitHub.

See [architecture](docs/architecture.md) and [Phase 2A evidence](docs/phase2a.md).

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
