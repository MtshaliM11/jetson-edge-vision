"""Opt-in hardware checks; never install packages or open the camera."""
import platform
import sys


def check_environment(device: str) -> dict:
    import cv2
    import numpy as np
    import torch
    import torchvision

    cuda_available = torch.cuda.is_available()
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA unavailable; verify NVIDIA device access and the existing AI environment")
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
    if tensor.sum().item() != 6.0:
        raise RuntimeError("Tensor computation failed")
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device=device)
    kept = torchvision.ops.nms(boxes, torch.tensor([.9, .8], device=device), .5).tolist()
    if kept != [0]:
        raise RuntimeError("Torchvision NMS check failed")
    frame = cv2.resize(np.zeros((8, 8, 3), dtype=np.uint8), (4, 4))
    if frame.shape != (4, 4, 3):
        raise RuntimeError("OpenCV resize check failed")
    gst = next((line.strip() for line in cv2.getBuildInformation().splitlines()
                if "GStreamer:" in line), "GStreamer: unknown")
    return {
        "python": sys.version.split()[0], "interpreter": sys.executable,
        "virtual_environment": sys.prefix != sys.base_prefix,
        "architecture": platform.machine(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "opencv": cv2.__version__, "gstreamer_build": gst,
        "cuda_build": torch.version.cuda, "cuda_available": cuda_available,
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "tensor_sum": 6.0, "nms_indices": kept, "opencv_resize": list(frame.shape),
    }
