# Phase 2B: GPU video-file object detection

Implemented and verified on 2026-10-17 with the existing AI environment. The
Phase 2A detector and annotation functions are reused unchanged. No packages,
system settings, drivers, weights, or model choice were changed. No camera,
tracking, unique-object counting, TensorRT, dashboard, or Phase 3 work was added.

## Run headlessly

```bash
cd ~/jetson-projects/ai/projects/jetson-edge-vision
source ~/jetson-projects/ai/ai-env/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs

python -m jetson_edge_vision --config configs/default.json \
  --video data/sintel-sample.mp4 --output outputs/sintel-run2.mp4 \
  --confidence 0.5 --warmup 3 > outputs/sintel-run2-report.json

# Optional short run and AVI/MJPEG output:
python -m jetson_edge_vision --video /path/to/input.mov \
  --output outputs/test.avi --codec MJPG --max-frames 24 \
  --confidence 0.6 > outputs/test-report.json
```

Choose an output filename that does not exist. Annotation output parents are
created automatically; shell report redirection requires an existing parent and
can overwrite old JSON reports. Results go to stdout, structured logs to stderr.
No windows are opened. Input must be a local file, not a camera, URL, or pipeline.
GPU access must be available to the process; CUDA never falls back to CPU.

| Option | Video behavior |
| --- | --- |
| `--video PATH` | Local MP4/AVI/MOV, mutually exclusive with image/download/diagnostic commands |
| `--output PATH` | Required new .mp4, .avi, or .mov output |
| `--confidence FLOAT` | Finite score threshold in [0, 1], default 0.5 |
| `--warmup N` | Nonnegative passes on first frame, once per video; default 3 |
| `--max-frames N` | Positive frame limit; default processes until decoder ends |
| `--codec FOURCC` | Four ASCII alphanumeric characters; defaults below |
| `--output-fps FLOAT` | Positive finite override; required if source FPS is unknown |
| `--weights PATH` | Existing approved COCO_V1 checkpoint; no download |
| `--config PATH` | Existing device/log-level config; default device CUDA |

`--runs` is image-only; video has exactly one measured inference per frame.
Using video-only flags without `--video` is an argument error. A CPU configuration
is explicit and available for debugging, but cannot verify CUDA performance.

## Architecture and resource lifetime

`video_pipeline.py` validates the source/output/options, opens a capture, reads
metadata and the first frame, and opens a writer with decoded geometry. The
SSDLite detector is constructed once, then warmed up through its existing API.
Each sequential frame passes through the detector's existing RGB conversion,
FP32 CUDA transfer, `torch.inference_mode()`, forward/NMS, CPU result extraction,
and confidence filtering. Existing annotation code draws boxes, COCO names, and
scores. No frame skipping, asynchronous queues, or inference duplication is used.

Only current-frame arrays/results exist during the loop. One eight-byte latency
sample per frame is retained for exact median calculation; no full-video frame,
detection-history, or GPU-tensor list is accumulated. Median calculation adds
temporary scalar-statistics memory, so CPU statistics storage grows with duration.
Peak memory was not measured. Torch's allocator may retain reusable GPU memory.

Reader/writer cleanup handles successful EOF, exceptions, failed initialization,
warm-up failure, and KeyboardInterrupt. Ctrl+C returns CLI exit status 130; other
handled failures return 1, argparse usage errors 2, success 0. Staged output is
finalized and fully decoded to check frame count, BGR geometry, and FPS before
publication. OpenCV can fail to encode silently, so checking `isOpened`/`open`
alone is insufficient. Explicit OpenCV exceptions and writer false returns also
fail the run.

Publication uses an atomic no-replace hard link from a temporary directory on
the destination filesystem. An output created by another process during inference
is preserved. This requires a filesystem that supports hard links (the verified
local filesystem does); for unsupported destinations, choose local storage first.
Failures before publication remove staging files and do not create a final video.
Uncatchable process termination/power loss may leave a `.jev-video-*` directory;
automatic recovery from that case is not implemented. No overwrite option exists.

## Codec and metadata policies

Actual three-frame encode/decode probes on OpenCV 4.8.0:

| Container | Encoder FourCC | Result | Backend |
| --- | --- | --- | --- |
| MP4 | mp4v (default) | 3 frames decoded | FFMPEG |
| MOV | mp4v (default) | 3 frames decoded | FFMPEG |
| AVI | MJPG (default) | 3 frames decoded | FFMPEG |
| MP4 | avc1 (optional) | 3 frames decoded | FFMPEG |

Evidence: local `outputs/phase2b-codecs.json`. Automated MP4/MOV/AVI round trips
also passed with synthetic frames and mocked inference. Codec availability can
vary by build, profile, pixel format, resolution, and media content. A container
extension alone never guarantees decode/encode support. If an encoder fails, the
error proposes a new `.avi` output with `--codec MJPG`; it never installs codecs
or silently changes the requested format. Model inference is GPU accelerated;
hardware-accelerated video decode/encode is not claimed.

Source metadata reports width, height, FPS, frame count where available, rotation,
sample aspect ratio, and backend. Decoded frame dimensions determine writer size;
reported width/height may differ after backend auto-rotation. No resizing occurs.
Known non-square source pixels and odd dimensions are rejected, rather than
stretching or allowing encoder truncation. A midstream size change also fails.

Auto-rotation is requested from the backend. Known nonzero rotation with unavailable
auto-rotation is rejected; absent/unsupported metadata can appear as zero, so this
cannot detect every orientation issue. Output writes decoded orientation, not
the original rotation metadata. Check portrait sources visually. OpenCV documents
backend-dependent [video properties and orientation support](https://docs.opencv.org/4.8.0/d4/d15/group__videoio__flags__base.html).

Source FPS is preserved unless explicitly overridden. Missing/nonfinite/zero FPS
requires `--output-fps`; no arbitrary rate is guessed. All decoded frames are
written at a constant FPS, without audio, subtitles, or source timestamps. Variable
frame rate timing is not preserved; an override can change playback duration.

## Errors and troubleshooting

- Missing files, unsupported extensions, unreadable media, or an empty/undecodable
  first frame fail before model inference.
- Invalid confidence, warm-up, frame limit, codec string, or FPS fail clearly.
- A malformed decoded frame or size change fails immediately, without silently
  skipping frames and altering duration.
- OpenCV's `read()` false result can mean EOF or decode failure. If the decoder
  stops before a known reported frame count, the run fails with a possible damage
  or inaccurate-metadata message. This conservative policy can reject usable media
  with an overestimated count. If count is unknown, termination is reported as
  `decoder_end` with a warning that EOF and decoder failure cannot be distinguished.
  Even with counts, a decoder that conceals damage cannot be reliably detected.
- Unavailable CUDA raises the existing detector error. Sandbox `NvRmMemInitNvmap`
  failures reproduced Phase 2A; approved host execution worked without repairs.
- Output permission, directory, disk, codec, and decode-verification errors stop
  publication. Try writable local storage or AVI/MJPG as appropriate. Do not
  install codecs or replace the working OpenCV to address an unverified cause.
- OpenCV/FFmpeg/GStreamer may write their own diagnostic lines to stderr; an
  intentionally corrupt-media test emits these even when the test passes.

## Real GPU verification and attribution

Sample: excerpt from **Sintel trailer**, © copyright Blender Foundation |
durian.blender.org, [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/).
Sources: [Blender project description](https://durian.blender.org/about/),
[Creative Commons interview confirming CC BY 3.0](https://creativecommons.org/2010/10/27/ton-roosendaal-sintel-producer-and-head-of-blender-institute/),
and [official trailer MP4](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4).
No dataset was introduced. This licensed movie excerpt is a functional test, not
an accuracy evaluation. The Phase 2A checkpoint and its
[licensing limitations](phase2a.md#licensing-limitations) are unchanged.

Local preparation: decoded frames 360–407 (15–17 seconds at 24 FPS), re-encoded
to `mp4v`, audio omitted, no geometry change. Detection output adds annotations.
Source and sample are under ignored `data/`; attribution is also saved locally in
`data/sintel-attribution.txt`. No media is added to Git.

```text
Trailer SHA256: b670602fa00934ca27c4351bb0efe7ea7a07fae57284e44226025eeed7c51254
Excerpt SHA256: 25b46296472622e699959153fc79b021aecba83a5dc0472d4e0f45b3793dfba8
```

Measured command: `python -m jetson_edge_vision --video data/sintel-sample.mp4
--output outputs/sintel-detected.mp4 --confidence 0.5 --warmup 3`, with stdout saved
to `outputs/sintel-video-report.json`. Run new output filenames to reproduce.

| Measurement | Actual result |
| --- | --- |
| Runtime | Orin, cuda:0, FP32; Torch 2.8.0 / Torchvision 0.23.0 / CUDA 12.6 |
| Source / output | MP4, 854x480, 24 FPS, 48 frames (2 seconds) |
| Output codec / backend | mp4v / FFMPEG |
| Measured frames / warm-up passes | 48 / 3 |
| Total detection observations | 35: person 33, bird 2 |
| Mean / median inference | 180.13 / 170.00 ms |
| Minimum / maximum inference | 125.90 / 254.14 ms |
| Streaming loop including writer finalization | 9,732.30 ms |
| Processing-loop throughput | 4.93 FPS |
| Model loading | 7,434.10 ms |
| Warm-up stage | 1,545.00 ms |
| Full output decode verification | 81.44 ms |
| Total processing | 19,040.50 ms |
| Full end-to-end throughput | 2.52 FPS |
| Output verification | All 48 frames decoded at 854x480, 24 FPS |

Output: `outputs/sintel-detected.mp4`; machine-readable report:
`outputs/sintel-video-report.json`. Decoded frames 0, 12, 24, and 36 were inspected
visually; boxes, names, and scores are visible. Local preview:
`outputs/sintel-detected-preview.jpg`. The `bird` label is a model prediction,
not validated ground truth. Repeated observations are not unique-object counts.

## Performance boundaries

- `source.fps` / `output_fps`: playback rates, independent of processing speed.
- `inference.*_ms`: one synchronized model call per frame through the unchanged
  detector. Includes internal transforms, network, NMS, box rescaling, and host
  overhead; excludes external BGR conversion, initial transfer, CPU result transfer,
  annotation, and video I/O. This is not pure GPU-kernel timing.
- `stream_processing_ms`: timed loop from the first measured frame through writer
  finalization. Includes conversion/transfer, inference, result extraction,
  annotation, subsequent frame reads, writes, and encoding flush. Excludes model
  load, warm-up, initial capture/metadata/first-frame read, and verification.
  `processing_fps = frames_processed / stream_seconds`.
- `total_processing_ms`: function entry through output verification/publication
  and staging cleanup; includes imports, validation, initialization, model load,
  warm-up, all frame processing, source release, and full verification decode.
  Excludes interpreter/CLI startup, downloads/sample preparation, and JSON printing.
  `end_to_end_fps = frames_processed / total_seconds`.

The 24 FPS output preserves playback timing despite processing more slowly. Neither
4.93 nor 2.52 FPS supports a real-time claim for this source. The original 151.25 ms
image baseline remains historical evidence, not a replacement benchmark: input,
workload, power/clock/thermal state, and background load differ. These conditions
were not controlled or modified. No optimization or guaranteed throughput is claimed.

## Tests and next milestone

All **57 tests passed with GPU testing enabled**, including all 28 original tests,
synthetic codec round trips, mock-inference video tests, and separate actual-CUDA
image/video integrations. Default discovery passes 55 and skips the two CUDA tests.

```bash
python -m unittest discover -s tests -v
JEV_GPU_TESTS=1 JEV_TEST_IMAGE=data/dog.jpg JEV_EXPECT_CLASS=dog \
  JEV_TEST_VIDEO=data/sintel-sample.mp4 JEV_VIDEO_EXPECT_CLASS=person \
  python -m unittest discover -s tests -v
python -m jetson_edge_vision --config configs/default.json --check-environment
```

With no optional media variables, CUDA tests use synthetic frames/videos and never
download anything. With them, known labels are also checked. The Phase 1 diagnostic
passed again: CUDA tensor sum 6.0, NMS `[0]`, OpenCV resize `[4, 4, 3]`, GStreamer
enabled. No Phase 1 or Phase 2A tests were modified.

Recommended next milestone: review representative real-world clips for detector
accuracy and threshold selection, then explicitly scope Phase 3 tracking. Keep
counting, TensorRT, camera integration, and dashboard work separate. Phase 3 was
not started automatically.
