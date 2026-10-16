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
