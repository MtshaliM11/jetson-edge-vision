# Jetson Edge Vision

Local, GPU-accelerated computer vision for NVIDIA Jetson Orin Nano Super.
The intended system combines detection, tracking, counting, and a local monitoring
interface without cloud services for core processing.

**Status: Phase 1 foundation verified. Detection is not implemented yet.**

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
it does not verify GPU functionality. `device` is a diagnostic setting in Phase 1.

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
 tests/test_foundation.py             Configuration, logging, and CLI tests
 docs/architecture.md                Module boundaries and phased roadmap
 docs/environment.md                 Dependencies and verification evidence
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
Image and local video inference come first; camera integration is Phase 5.
The verified `hello-gpu` and `camera-gpu` projects remain untouched.

## Roadmap

1. Foundation: configuration, logs, diagnostics, tests — verified.
2. Evaluate detector compatibility and licensing; implement PyTorch CUDA inference
   for images and local videos with measured latency and annotated output.
3. Add multi-object tracking with temporary, session-scoped IDs.
4. Add tested line/region counting and stored events.
5. Integrate CSI capture and document camera limitations.
6. Add a local API and simple dashboard.
7. Benchmark and evaluate TensorRT/FP16 using measured results.
8. Prepare documentation, licensing, examples, and release checks for GitHub.

See [architecture](docs/architecture.md). No inference FPS or latency claims are
available: the Phase 1 checks are functional checks, not benchmarks.

## Repository hygiene and licensing

Git is initialized locally on `main`; no remote, commit, or push is created by
Phase 1. `.gitignore` excludes environment files, caches, typical model weights,
video recordings, logs, local data, and database outputs. Review `git diff` and
`git status` before staging: ignore rules cannot detect every credential or
private image. Keep recordings and downloaded assets under ignored `data/` and
weights under ignored `models/`.

An open-source license has not yet been selected; this is not yet an open-source
release. Select a project license and review model, weight, and dependency
licenses before publication. No model dependencies or weights have been added.
