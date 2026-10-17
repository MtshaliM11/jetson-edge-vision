# Architecture

Phase 1 provides startup configuration, logging, and diagnostics. The CLI
coordinates these small modules; configuration and logging require only Python's
standard library. Hardware dependencies are imported only by explicit diagnostics.
Phases 2A/2B/3A also load hardware libraries only for explicit download/detection
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

## Phase 2B implemented path

```text
CLI --video -> OpenCV VideoCapture -> SSDLiteDetector.detect -> annotate_image
                      ^                       |                       |
                      +---- next frame -------+                OpenCV VideoWriter
                                                                     |
                                                      temporary encoded video
                                                                     |
                                                full decode/count/geometry/FPS check
                                                                     |
                                                     publish new output + JSON
```

`video_pipeline.py` owns local-file validation, source metadata, capture/writer
lifetime, one reusable detector, per-video warm-up, a sequential frame loop, and
performance aggregation. It reuses `SSDLiteDetector.detect(..., warmup=0, runs=1)`
and the existing `image_pipeline.annotate_image`. Phase 3A extends the annotation
function only when tracking fields are present; ordinary detection rendering is preserved.
The warm-up uses the same detector API on the first decoded frame, before loop
timing. Configuration remains unchanged; video-specific controls are CLI arguments.

The pipeline keeps only current-frame data, not a video-sized frame/detection
history. Exact median inference timing retains one float64 latency per frame
(eight bytes each, plus temporary statistics overhead). Model tensors and
predictions are released by the existing detector's inference-mode path. GPU
memory does not grow through retained frame tensors; peak usage is not benchmarked.

Decoded dimensions determine writer size, with no resizing/stretching. Output
FPS defaults to reported source FPS, or requires an explicit override if unknown.
Audio and variable timestamp spacing are not carried through this OpenCV path.
Writer output is staged next to the destination, finalized, fully decoded, and
published with a no-replace hard link. Readers/writers close in `finally`/cleanup
handlers, including model-load failures, warm-up failures, and Ctrl+C. The temporary
directory is removed automatically. See [Phase 2B](phase2b.md) for error policies,
codec probes, publication/filesystem limits, and measurement boundaries.

## Phase 3A implemented path

```text
CUDA detector -> current detection records -> optional IoUTracker.update
                                                       |
                                   current records + track_id/track_confirmed
                                                       |
                                                annotate_image -> writer
```

`tracking.py` uses only the standard library. `TrackingConfig` is immutable and
validated; `Track` is an immutable last-observation state record. One `IoUTracker`
is constructed per enabled video call. `update()` is called exactly once per
decoded frame, including empty detections, after inference and before annotation.
Warm-up does not advance tracking. State never crosses video sessions; explicit
`reset()` restarts IDs at 1. No global ID counter or biometric identity exists.

Greedy candidate matching is class-aware and one-to-one, sorted by descending
IoU, ascending track ID, then detection index. Matched tracks update their last
box/score; unmatched detections create monotonic IDs; unmatched tracks accrue
misses and expire when `missed > max_missed`. Tentative tracks need consecutive
hits to confirm, and confirmation persists until expiration. Output contains only
current detections, preserving detector order; retained missed boxes are never
rendered as observations. See [Phase 3A](phase3a.md) for exact boundary semantics.

Default-disabled integration avoids creating a tracker or running association.
Enabled integration adds IDs, status, deterministic palette colors, association
timing, active-state summaries, and optional DEBUG per-frame JSON messages via
the existing structured logger. No frame/detection history is accumulated by the
tracker. Median association timing adds one float64 per processed frame, parallel
to existing inference samples. Aggregate track statistics are diagnostic state,
not object counting. The detector, configuration schema, and original tests remain
unchanged. Tracking controls are CLI flags rather than new JSON config keys.

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
bounding-box data without owning camera capture. The tracker consumes
per-frame detections and maintains temporary session IDs. Counting will consume
tracks and configured geometry; synthetic trajectories will verify crossings,
entry/exit events, and duplicate suppression independently of inference.

Persistence will store events rather than sit inside detector code. The API will
expose application state; dashboard code will not own inference or counting.
Shared frame/detection/track/event types will be introduced alongside the first
consumer, avoiding empty speculative modules now. Start with a synchronous
pipeline so ordering and latency are understandable; introduce concurrency only
when measurements justify its memory and synchronization costs.

Phase 2B distinguishes synchronized per-frame inference, processing-loop throughput,
and full end-to-end throughput. These measurements do not establish real-time
streaming capability. Phase 2A's repeated image passes remain a separate baseline.

The initial detector evaluation, CUDA image/video paths, and baseline IoU tracking
are complete without new dependencies. Further tracking evaluation, counting,
TensorRT, camera capture, and the dashboard remain future work.
