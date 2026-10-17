# Architecture

Phase 1 provides startup configuration, logging, and diagnostics. The CLI
coordinates these small modules; configuration and logging require only Python's
standard library. Hardware dependencies are imported only by explicit diagnostics.
Phase 2A also loads hardware libraries only for explicit download/detection
commands. This keeps ordinary startup and foundation tests independent of GPU
availability. The original config schema, logging, and diagnostics remain intact.

## Phase 2A implemented path

```text
CLI --image -> image_pipeline.detect_image -> detection.SSDLiteDetector
                    |                           |
              OpenCV decode                RGB float32 tensor
                    |                      CUDA model + built-in NMS
                    |                           |
              OpenCV annotation <--- plain detection records
                    |
             image file + JSON report
```

`detection.py` owns the approved model identity, explicit checkpoint download,
hash verification, model loading, input conversion, inference, COCO label mapping,
and per-pass latency. The reusable detector consumes a BGR uint8 frame and returns
plain records; it does not own file I/O or capture. The frozen `Detection` record
contains class ID/name, confidence, and original-image pixel coordinates.

`image_pipeline.py` validates file paths and options before loading the model,
decodes the image, calls the detector, draws annotations on a copy, saves the
output, and reports total elapsed time. Existing output files and the source
image are protected against overwrite. JSON reports are printed by the CLI;
shell redirection controls whether they are saved.

`--download-weights` is a separate command. Inference requires a verified local
checkpoint and initializes both model and backbone without automatic downloads.
CUDA is explicit, FP32, with `eval()` and `inference_mode()`. CPU is available only
through explicit configuration. CUDA failures propagate instead of selecting CPU.
Torchvision owns internal 320x320 resize/normalization, box rescaling, and NMS;
the application does not apply these operations twice.

Inference timers synchronize CUDA before/after every measured model call. They
include internal transforms and NMS but exclude image decoding, BGR conversion,
initial transfer, model loading, output transfer/filtering, annotation, and writing.
Total pipeline time includes all of these plus warm-up and every measured pass.
See [Phase 2A](phase2a.md) for exact measurement boundaries and observed results.

## Planned processing boundaries

```text
image / video / CSI source -> detector -> tracker -> counter -> analytics
                                 |          |          |
                                 +---- performance ----+
                                                       |
                                                persistence / API
                                                       |
                                                   dashboard
```

Acquisition will own capture handles, frame timestamps, end-of-stream handling,
and cleanup. The detector will consume frames and return class, confidence, and
bounding-box data without owning camera capture. The tracker will consume
per-frame detections and maintain temporary identities. Counting will consume
tracks and configured geometry; synthetic trajectories will verify crossings,
entry/exit events, and duplicate suppression independently of inference.

Persistence will store events rather than sit inside detector code. The API will
expose application state; dashboard code will not own inference or counting.
Shared frame/detection/track/event types will be introduced alongside the first
consumer, avoiding empty speculative modules now. Start with a synchronous
pipeline so ordering and latency are understandable; introduce concurrency only
when measurements justify its memory and synchronization costs.

Future streaming measurements will distinguish acquisition, inference, and
end-to-end throughput. Phase 2A measures only synchronous single-image processing;
its repeated passes do not establish video FPS.

The initial detector evaluation and CUDA image path are complete without new
dependencies. Video-file processing, tracking, counting, TensorRT, camera capture,
and the dashboard remain future work.
