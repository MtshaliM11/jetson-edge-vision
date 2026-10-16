# Architecture

Phase 1 implements only startup configuration, logging, and diagnostics. The CLI
coordinates these small modules; configuration and logging require only Python's
standard library. Hardware dependencies are imported only by explicit diagnostics.
This keeps ordinary startup and unit tests independent of GPU availability.

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

Performance measurements will distinguish acquisition, model inference, and
end-to-end throughput. CUDA execution is asynchronous, so future inference timing
must use synchronization or CUDA events with explicit measurement boundaries.
No timing instrumentation or benchmark results are claimed in Phase 1.

The next increment is a detector evaluation including ARM64 support, installed
PyTorch compatibility, memory, accuracy, maintenance, and code/weight licenses.
Only then add the minimal dependencies and implement image inference followed by
video-file processing. TensorRT, camera capture, and the dashboard remain later
milestones.
