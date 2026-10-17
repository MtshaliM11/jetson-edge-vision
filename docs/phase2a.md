# Phase 2A: GPU image object detection

Implemented and verified on 2026-10-16. Scope is image files only. Video, tracking,
counting, TensorRT, camera capture, and dashboard work remain future phases.

## Model decision

The user approved Torchvision SSDLite320 MobileNetV3-Large `COCO_V1` and its
official checkpoint download. It fits the existing PyTorch 2.8.0 / Torchvision
0.23.0 / CUDA 12.6 / Python 3.10 / ARM64 environment and needs no new dependencies.
Local construction without weights was checked before approval; pretrained CUDA
inference was verified afterwards. Ubuntu is 22.04.5 LTS, GPU is reported as Orin,
and OpenCV 4.8.0 retains GStreamer support.

| Candidate | Parameters | Published COCO box AP | Decision |
| --- | --- | --- | --- |
| SSDLite320 MobileNetV3-Large COCO_V1 | 3.44M | 21.3 | Selected: existing API, compact 13.4 MB checkpoint |
| Faster R-CNN MobileNetV3-Large 320 FPN | 19.39M | 22.8 | Small AP improvement with a 74.2 MB checkpoint |
| Ultralytics YOLO11n, 640 input | 2.6M | 39.5 | Higher published AP, but another framework and AGPL/Enterprise considerations |

Sources: [Torchvision SSDLite 0.23](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html),
[Torchvision Faster R-CNN 0.23](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html),
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/), and
[Ultralytics licensing](https://www.ultralytics.com/license).
AP is averaged over IoU 0.50–0.95; it is not a percentage of individual detections
that are correct. Different input resolutions/training/evaluation pipelines limit
direct comparison. These published numbers do not establish Jetson latency.

SSDLite's 320x320 internal input limits small/distant-object detail. Crowding,
occlusion, poor lighting, the existing camera colour issue, and deployment-domain
differences can reduce accuracy. The model recognizes 80 COCO categories, not
arbitrary custom labels. Its metadata has 91 slots including background and
unused entries; names are indexed directly from that metadata. Confidence is a
model score, not a calibrated probability. A successful dog example demonstrates
the pipeline, not deployment accuracy. Representative labelled images are still
needed for an accuracy assessment. Torchvision marks its detection API as beta;
this implementation targets the installed 0.23 API.

## Checkpoint and reproducibility

Official source:
[PyTorch COCO_V1 checkpoint](https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth).

Local path: `models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth`.
The loader checks the official filename hash prefix `a79551df` before deserializing
with `torch.load(..., weights_only=True, map_location="cpu")` and strict state loading.
Downloaded full SHA256:

```text
a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2
```

Example source: [PyTorch Hub dog image](https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg),
stored locally as `data/dog.jpg` (1546x1213). Image SHA256:

```text
f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a
```

The sample is a local verification asset, not redistributed in Git; its availability
from that repository does not establish unrestricted image redistribution rights.

Commands used, after activating the environment and setting `PYTHONPATH` as in
the README:

```bash
python -m jetson_edge_vision --download-weights
mkdir -p data outputs
curl --fail --location --max-time 60 \
  https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg \
  --output data/dog.jpg
python -m jetson_edge_vision --image data/dog.jpg \
  --output outputs/dog-detected.jpg --confidence 0.5 --warmup 3 --runs 10 \
  > outputs/dog-report.json
```

The assets already exist in the verified workspace. For another run, select a
new annotation/report filename; existing annotated images are rejected. Shell
redirection itself can overwrite reports, so choose those names deliberately.
Paths resolve from the current working directory. Inference never downloads
weights. `--download-weights` alone is the only network operation in application
code. No pip, apt, build, or installation command is needed.

## Actual GPU results

Runtime report: Torch 2.8.0, Torchvision 0.23.0, CUDA build 12.6, GPU `Orin`, model
parameters on `cuda:0`, FP32, evaluation/inference mode. CUDA availability and
successful execution were verified, rather than inferred from a CUDA build string.

| Measurement | Observed |
| --- | --- |
| Category / COCO ID | dog / 18 |
| Confidence | 0.7511516809 |
| Box `[x1, y1, x2, y2]`, original pixels | [146.88, 41.31, 1472.18, 1144.44] |
| Threshold | 0.5 |
| Warm-up / measured passes | 3 / 10 |
| Mean inference | 151.25 ms |
| Median inference | 146.20 ms |
| Minimum / maximum inference | 103.67 / 207.64 ms |
| Input conversion and transfer | 117.06 ms |
| Model loading and initialization | 7,562.22 ms |
| Total image processing | 10,701.65 ms |

Inference samples in milliseconds: 111.24, 103.67, 108.72, 207.64, 177.15,
188.27, 200.71, 165.94, 126.47, 122.70.

`outputs/dog-detected.jpg` was saved and visually inspected: the dog has a green
box and a `dog 0.75` label. `outputs/dog-report.json` preserves the full-precision
measurements, coordinates, runtime versions, and checkpoint digest. These assets
are ignored by Git. No additional categories passed threshold in this image.

## Measurement boundaries

- `preprocessing_ms`: BGR-to-RGB conversion, tensor conversion to RGB CHW float32
  [0, 1], transfer to the requested device, and synchronization.
- `inference`: wall-clock `perf_counter` around each model call with explicit
  CUDA synchronization before and after. Includes model-internal 320x320 resize,
  normalization, network execution, box decoding, built-in NMS, and box rescaling.
  Excludes input conversion/transfer and output-to-CPU transfer/application filtering.
  This includes Python/host overhead; it is not isolated GPU-kernel time.
- `model_load_ms`: detector construction, including lazy Torch/Torchvision imports,
  CUDA initialization/checks, checksum verification, checkpoint loading, model
  construction, device transfer, and final synchronization. Includes random model
  initialization before pretrained weights are loaded.
- `total_processing_ms`: entry to `detect_image` through successful annotated-image
  writing. Includes remaining lazy imports, validation, decode, model load, input
  preparation, all warm-up and measured passes, result transfer/filtering, annotation,
  and image encoding/writing. Excludes interpreter/CLI startup, weight download,
  and JSON serialization/stdout writing. It is not a one-pass steady-state latency.

The final measured pass supplies detections. Runs reuse one image and one loaded
model; no video throughput is claimed. Jetson power mode, clocks, thermal state,
and other workloads were not controlled or changed. Variation is reported rather
than hidden; optimization is deferred. Peak memory was not benchmarked.

## Automated and regression checks

The final host-GPU run passed **28 tests**, including the eight unchanged Phase 1
tests. Default discovery runs 27 tests and skips the opt-in CUDA test. Coverage
includes confidence validation, sparse COCO label IDs, empty results, timing
statistics, BGR-to-RGB conversion and normalization, inference mode, annotation
without source mutation, file protection, decode/write failures, CLI routing and
errors, lazy hardware imports, missing/corrupt checkpoints, and no CPU fallback.

The opt-in integration test executes real pretrained CUDA inference on a blank
image and checks timing/output. With the dog example enabled, it also runs the
complete image pipeline, decodes the annotation, and checks the expected class.
Missing CUDA/weights are failures when integration is explicitly enabled.

```bash
python -m unittest discover -s tests -v
JEV_GPU_TESTS=1 JEV_TEST_IMAGE=data/dog.jpg JEV_EXPECT_CLASS=dog \
  python -m unittest discover -s tests -v
python -m jetson_edge_vision --config configs/default.json --check-environment
```

The original GPU diagnostic passed again: tensor sum 6.0, CUDA NMS indices `[0]`,
OpenCV resize `[4, 4, 3]`, GStreamer enabled. Sandboxed CUDA initially failed with
NVIDIA memory-manager access errors, then succeeded with approved host execution.
This was an execution-permission issue; no CUDA or package changes were needed.

## Licensing limitations

Torchvision's [BSD-3-Clause code license](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE)
permits reuse subject to its notice and other conditions. Its
[pretrained-model disclaimer](https://github.com/pytorch/vision/blob/v0.23.0/README.md#pre-trained-model-license)
separately warns that weights may have dataset-derived licenses or terms and
places responsibility on the user to determine permitted use. The BSD code
license alone does not establish unrestricted rights to these COCO-trained weights.

No unconditional commercial-use or redistribution clearance is asserted for this
checkpoint or its training images. Review the applicable weight/dataset terms
for the intended release or deployment. The project's own open-source license
is still unselected. No license was silently selected, and no weights or example
images are included in Git. The approved local evaluation does not resolve those
future distribution questions.
