import math
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
import prt_nn.common.utils as utils

class YoloDetector:
    """
    Inference-only adapter for Ultralytics YOLO (v8/v9 family).

    This class wraps an Ultralytics YOLO model to run **batched** object-detection
    inference on PyTorch tensors and returns results in a **canonical** format.

    Expected image batch (to ``predict``)
    -------------------------------------
    images : ``torch.Tensor`` of shape ``(N, C, H, W)``
        - Color order: **RGB**
        - Dtype: ``uint8`` in ``[0, 255]`` **or** ``float32`` (Ultralytics
          will internally normalize; both work in practice).
        - Device: CPU or CUDA. The YOLO model will run on its internal device.
          If you need a specific device, move the model via
          ``detector.model.to(device)`` and send the batch to the same device.

    Returned predictions
    --------------------
    A ``list`` (length ``N``) of dicts, one per image:

    - **"boxes"**  → ``FloatTensor [M, 4]`` in **XYXY** pixel coords
    - **"scores"** → ``FloatTensor [M]`` (confidence in ``[0, 1]``)
    - **"labels"** → ``LongTensor   [M]`` (contiguous class indices)

    When there are no detections for an image, ``M = 0`` and the tensors are
    zero-length. Class-name mapping is available via ``self.model.names``
    (a list/dict provided by Ultralytics).

    Args:
        weights : pathlib.Path | str | None, default: None
            Path or model name for YOLO weights (e.g., ``"yolov8n.pt"``).
            If ``None``, this class looks under your resolved data root
            (see ``prt_datasets.common.utils.resolve_root(None) / "yolov8n.pt"``).
            Pass a model name (``"yolov8n.pt"``/``"yolov8s.pt"``/...) to let
            Ultralytics download automatically if missing, or pass a local file path.
        confidence : float, default: 0.25
            Score threshold applied **before** NMS (Ultralytics ``conf``).
        iou : float, default: 0.70
            IoU threshold used by NMS (Ultralytics ``iou``).
        max_det : int, default: 300
            Maximum detections per image after NMS.    
        device : torch.device | None, default: None
            Device to run the model on. If ``None``, uses CUDA if available, else CPU.

    Example:
        .. code-block:: python

            import torch
            import numpy as np
            from PIL import Image
            from pathlib import Path

            # 1) Construct the detector (downloads model if needed when using a name)
            det = YoloDetector(weights="yolov8n.pt", confidence=0.35, iou=0.6, max_det=200)

            # 2) Prepare a single RGB image as (1, 3, H, W) uint8
            img = Image.open("example.jpg").convert("RGB")
            x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).unsqueeze(0)

            # 3) Inference
            preds = det.predict(x)         # list of length 1
            p0 = preds[0]
            print(p0["boxes"].shape, p0["scores"].shape, p0["labels"].shape)

            # 4) Map label indices to names
            names = det.model.names
            top_names = [names[int(i)] for i in p0["labels"][:5].tolist()]
            print(top_names)

    Notes:
        - **Preprocessing**: Ultralytics will handle resizing/letterboxing and
        normalization internally. If you need a fixed input size, you can extend
        this wrapper to pass ``imgsz=...`` into the model call.
        - **Throughput**: Batch multiple images together for better GPU utilization.
        - **Precision**: You can put the model in half precision on CUDA with
        ``det.model.to(torch.float16)`` (ensure inputs are compatible).
        - **Device of outputs**: Outputs are returned on the model's device. If you
        require a specific device for downstream code, move them via ``.to(device)``.
        - **Dependencies**: Requires ``ultralytics`` (install with ``pip install ultralytics``).

    Raises:
        ImportError
            If ``ultralytics`` is not installed.
        FileNotFoundError / RuntimeError
            If a local ``weights`` path is provided but cannot be loaded.
    """
    def __init__(self, 
                 weights: Path | None = None, 
                 confidence: float = 0.25, 
                 iou: float = 0.7, 
                 max_det: int = 300,
                 device: torch.device | None = None
                 ) -> None:
        try:
            from ultralytics import YOLO  # lazy import so dependency is optional
        except Exception as e:
            raise ImportError("ultralytics is required for YoloDetector: pip install ultralytics") from e
        
        # If no pretrained weights are specified, use the default YOLOv8n
        if weights is None:
            weights = utils.resolve_root(None) / "yolo" / "yolov8n.pt"

        self._YOLO = YOLO
        self.model = self._YOLO(weights)

        # Try to use GPU if available and no device specified
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.confidence, self.iou, self.max_det = confidence, iou, max_det

    @torch.inference_mode()
    def predict(self, images: Tensor) -> List[Dict[str, Tensor]]:
        """
        Perform inference on a batch of images (BCHW). Handles stride padding internally.

        Args:
            images: (N, C, H, W) uint8 [0,255] or float32 [0,1]

        Returns:
            List[Dict]: per-image dicts with keys:
              - "boxes":  (M, 4) xyxy in **original** image coords (padded area removed, clipped)
              - "scores": (M,)
              - "labels": (M,)
        """
        # 1) pad to stride-multiple, keep top-left anchored
        images, meta = self._preprocess(images, stride=32, pad_value=114.0 / 255.0)
        orig_h, orig_w = meta["orig_hw"]

        # 2) run model (Ultralytics accepts BCHW float in [0,1])
        results = self.model(images, conf=self.confidence, iou=self.iou, max_det=self.max_det, verbose=False)

        # 3) gather outputs, clip to original H,W (no translation needed; we padded bottom/right only)
        out: List[Dict[str, Tensor]] = []
        device = images.device
        for r in results:
            b = r.boxes
            if b is None or b.xyxy.shape[0] == 0:
                out.append({
                    "boxes":  torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "scores": torch.zeros((0,),   device=device, dtype=torch.float32),
                    "labels": torch.zeros((0,),   device=device, dtype=torch.long),
                })
                continue

            boxes = b.xyxy.detach()  # (M,4)
            # clip against original dims to drop any padding region
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(min=0, max=orig_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(min=0, max=orig_h)

            out.append({
                "boxes":  boxes,
                "scores": b.conf.detach(),
                "labels": b.cls.to(torch.int64).detach(),
            })
        return out
    
    def _preprocess(
        self,
        images: Tensor,
        stride: int = 32,
        pad_value: float = 114.0 / 255.0,
    ) -> Tuple[Tensor, dict]:
        """
        Make a BCHW batch stride-compatible by padding only on the **right** and **bottom**.
        Keeps the top-left corner at (0,0), so no coordinate shift is needed for boxes.

        Args:
            images: (N, C, H, W) uint8 in [0,255] or float in [0,1]
            stride: model stride (YOLO default is 32)
            pad_value: padding value in [0,1] (Ultralytics uses ~114/255)

        Returns:
            images_out: (N, C, H', W') float32 in [0,1], with H',W' % stride == 0
            meta: {"orig_hw": (H, W), "padded_hw": (H', W'), "pad": (0, pad_right, 0, pad_bottom)}
        """
        assert images.ndim == 4, "Expected BCHW"
        n, c, h, w = images.shape

        # to float32 in [0,1]
        if images.dtype == torch.uint8:
            images = images.to(torch.float32) / 255.0
        elif images.dtype != torch.float32:
            images = images.to(torch.float32)

        new_h = math.ceil(h / stride) * stride
        new_w = math.ceil(w / stride) * stride
        pad_h = new_h - h
        pad_w = new_w - w

        if pad_h == 0 and pad_w == 0:
            return images, {"orig_hw": (h, w), "padded_hw": (h, w), "pad": (0, 0, 0, 0)}

        # F.pad pads in (left, right, top, bottom) for 4D tensors
        images = F.pad(images, (0, pad_w, 0, pad_h), value=pad_value)
        return images, {"orig_hw": (h, w), "padded_hw": (new_h, new_w), "pad": (0, pad_w, 0, pad_h)}

