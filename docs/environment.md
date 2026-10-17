# Environment and dependencies

Verified on 2026-10-16 using `/home/mbongeni/jetson-projects/ai/ai-env/bin/python`.
The working directory is `/home/mbongeni/jetson-projects/ai/projects/jetson-edge-vision`.
The shell already selected the existing AI virtual environment.

| Component | Observed |
| --- | --- |
| Architecture | aarch64 |
| Python | 3.10.12, existing ai-env |
| PyTorch | 2.8.0, CUDA build 12.6 |
| Torchvision | 0.23.0 |
| GPU | Orin, compute capability 8.7 |
| NumPy | 1.26.4 |
| OpenCV | 4.8.0; GStreamer YES (1.20.3) |
| GStreamer CLI | 1.20.3 |
| CUDA compiler | 12.6, V12.6.68 |
| Git | 2.34.1 |
| setuptools | 84.0.0 |
| pip | 26.2.1 |
| pytest | Not installed; not needed for Phase 1 |

## Executed checks

- Imported Torch, Torchvision, NumPy, and OpenCV successfully.
- CUDA tensor sum of `[1, 2, 3]` returned `6.0`.
- Torchvision CUDA NMS suppressed identical overlapping boxes, returning `[0]`.
- OpenCV resized a synthetic 8×8 BGR frame to shape `(4, 4, 3)`.
- OpenCV build information reported GStreamer integration; the GStreamer CLI ran.
- Eight unittest tests passed for defaults, example config, invalid config,
  missing files, JSON exception logging, repeated logging setup, startup, and
  startup failure exit status.
- The module CLI ran with the default configuration and its hardware diagnostic
  completed successfully against the GPU.

The initial sandboxed CUDA check failed with NVIDIA memory-device access errors.
The same check and the final CLI diagnostic succeeded outside the sandbox with
approval. This did not require an environment repair or package replacement.
Git initialization also required access outside the sandbox's read-only `.git`.

No camera was opened and no GStreamer video pipeline was executed. Build support
is not proof of camera correctness. Camera colour quality, decoding representative
media, inference, throughput, and TensorRT have not been tested in this phase.
Other hardware/system versions in the project brief remain user-reported unless
listed above; this is not a full system inventory.

## Dependency policy

Configuration, CLI startup, structured logging, and tests use the standard library.
The optional environment diagnostic uses the already installed Torch,
Torchvision, NumPy, and OpenCV. `pyproject.toml` intentionally has no automatically
resolved AI dependencies: generic wheels can replace Jetson-compatible builds.
Build metadata uses setuptools, but building/installing the package is unnecessary
for the documented source-tree workflow.

Do not use this inventory as a generic pip requirements file. A future dependency
addition must check installed versions, ARM64 wheels, CUDA compatibility,
transitive requirements, and licenses before any installation. Preserve the
existing OpenCV/system integration and CUDA PyTorch installation. This phase
installed nothing, downloaded no weights, and modified no drivers or system tools.

## Phase 2A verification (2026-10-16)

The same existing interpreter and library versions were reused. Read-only checks
also confirmed Ubuntu 22.04.5 LTS and Torchvision's CUDA build value `12060`,
matching CUDA 12.6. The NVIDIA release file reports L4T R36 revision 4.7; no
JetPack, driver, CUDA, or library changes were made. No packages were installed.

The user approved the official SSDLite320 MobileNetV3-Large COCO_V1 checkpoint.
It was downloaded into ignored `models/`, verified against the official SHA256
prefix, and loaded with `weights_only=True`. The example image, annotations,
and JSON reports are under ignored `data/` and `outputs/`.

Sandboxed network downloads failed DNS resolution; approved host execution
resolved this. Sandboxed CUDA initialization failed with `NvRmMemInitNvmap` and
NVIDIA memory-manager access errors. The same detector succeeded outside the
sandbox on `cuda:0` (GPU name `Orin`), without any environment repair. CUDA failure
never triggered CPU fallback.

The actual image test detected a dog and saved a visually checked annotated image.
The complete 28-test suite, including all eight original tests and the opt-in
CUDA/known-class test, passed. The Phase 1 GPU diagnostic was rerun successfully
after implementation. See [Phase 2A evidence](phase2a.md) for measured timings,
reproduction commands, model limitations, and licensing notes.

## Phase 2B verification (2026-10-17)

Reused the existing interpreter, CUDA checkpoint, and all installed libraries.
No Python/system packages, drivers, camera settings, CUDA, or JetPack were changed.
MP4/mp4v, MOV/mp4v, AVI/MJPG, and MP4/avc1 probe files each encoded and decoded
three frames successfully through the existing OpenCV FFmpeg backend. GStreamer
support remains intact; this does not imply GPU-accelerated video decoding/encoding.

A licensed Sintel trailer excerpt (48 frames, 854x480, 24 FPS) was processed with
the unchanged detector on `cuda:0`, FP32. All 48 annotated output frames decoded
successfully, with the same dimensions/FPS. Processing-loop throughput was 4.93
FPS, mean synchronized inference 180.13 ms, and full end-to-end throughput 2.52
FPS. See [Phase 2B](phase2b.md) for the exact boundaries and sample attribution.

All 57 tests passed with GPU testing enabled, including the unchanged Phase 1/2A
tests and a separate CUDA video integration. The original environment diagnostic
also passed again. The sandbox still blocks NVIDIA memory-device access; approved
host execution succeeded without repairs. No weights were redownloaded. Media,
codec evidence, and reports remain local under ignored `data/` and `outputs/`.

## Phase 3A verification (2026-10-17)

Added an original standard-library IoU tracker without installing packages or
changing the model, checkpoint, drivers, CUDA, JetPack, or AI library versions.
The existing CUDA detector processes video frames as before. The sandbox still
blocks NVIDIA memory-device access; approved host execution succeeded without
environment repair.

Matched 48-frame runs with tracking disabled/enabled used the same Sintel source,
confidence 0.5, three warm-ups, mp4v output, and INFO logging. Tracker updates
averaged 0.1035 ms/frame (4.9674 ms total). Mean model latency was 168.75/164.93 ms,
processing FPS 5.27/5.38, and full end-to-end FPS 2.67/2.66 off/on. These noisy runs
do not establish a speedup or real-time performance. Both outputs decoded all 48
frames at 854x480 and 24 FPS. A separate DEBUG run verified ID persistence and
revealed a class flip and fragmentation across shots; see [Phase 3A](phase3a.md).

All 85 tests passed with actual GPU integrations enabled, including all 57 prior
tests unchanged. The Phase 1 CUDA tensor/NMS/OpenCV diagnostic passed again.
Tracking output, logs, comparisons, and previews remain under ignored `outputs/`.
