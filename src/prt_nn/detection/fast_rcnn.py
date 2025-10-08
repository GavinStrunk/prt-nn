from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.hub
import prt_nn.common.utils as utils

class FastRCNNDetector:
    """
    Inference-only adapter for Faster R-CNN (ResNet-50 FPN) via torchvision.

    This class runs **batched** object detection and returns predictions in the
    canonical `{boxes, scores, labels}` format.

    Expected batch (to `predict`)
    -----------------------------
    images : torch.Tensor of shape (N, C, H, W)
        - RGB, dtype uint8 in [0,255] **or** float32.
        - Will be converted to float32 in [0,1] for the model.
        - May reside on CPU or CUDA; the tensor is moved to the model device.

    Returns
    -------
    List[Prediction]
        For each image:
        - "boxes"  : FloatTensor [M,4] in XYXY pixel coords
        - "scores" : FloatTensor [M]
        - "labels" : LongTensor   [M]  (contiguous class indices)

    Args:
        weights : Path | None, default: None
            - None  -> use torchvision COCO-pretrained weights.
            - Path  -> load a local state_dict checkpoint (e.g., .pt/.pth).
                    (Keys `state_dict` or `model` will be unwrapped if present.)
        confidence : float, default: 0.25
            Score threshold applied **after** model output.
        max_det : int, default: 300
            Keep at most this many detections per image after thresholding.
        device : torch.device | None, default: auto
            If None, chooses `"cuda"` if available else `"cpu"`.

    Example:
        .. code-block:: python

            import torch, numpy as np
            from PIL import Image

            det = FastRCNNDetector(confidence=0.5)
            img = Image.open("example.jpg").convert("RGB")
            x = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2,0,1).unsqueeze(0)  # (1,3,H,W) uint8
            preds = det.predict(x)
            print(preds[0]["boxes"].shape, preds[0]["scores"][:5], preds[0]["labels"][:5])

    Notes:
        - Uses torchvision postprocessing (resizing/letterboxing handled internally).
        - COCO pretrained model outputs 91 classes (including background conventions
        handled internally by torchvision). Labels are contiguous class indices.
        - To force GPU: `det.model.to("cuda")` or pass `device=torch.device("cuda")`.
    """

    def __init__(
        self,
        *,
        weights: Optional[Path] = None,
        confidence: float = 0.25,
        max_det: int = 300,
        device: Optional[torch.device] = None,
    ) -> None:
        try:
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        except Exception as e:
            raise ImportError("torchvision>=0.14 is required for FastRCNNDetector") from e

        # Device selection
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Get the pretrained model directory
        if weights is None:
            root = utils.resolve_root(None, create=True) / "fastrcnn"
            torch.hub.set_dir(str(root))

        # Build model
        if weights is None:
            # Official COCO-pretrained weights
            self.model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
        else:
            # Create arch and load external state_dict
            self.model = fasterrcnn_resnet50_fpn(weights=None)
            sd = torch.load(str(weights), map_location="cpu")

            if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model")):
                sd = sd.get("state_dict", sd.get("model", sd))

            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[FastRCNNDetector] Missing keys: {len(missing)} (showing first 5) {missing[:5]}")
            if unexpected:
                print(f"[FastRCNNDetector] Unexpected keys: {len(unexpected)} (showing first 5) {unexpected[:5]}")

        self.model.to(self.device).eval()
        self.confidence = float(confidence)
        self.max_det = int(max_det)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Perform inference on a batch of images.
        
        Args:
            images: A batch of images as a tensor of shape (N, C, H, W) and dtype uint8 [0, 255].
        Returns:
            A list of predictions, one per image. Each prediction is a dict with keys:
                - "boxes": Tensor of shape (num_boxes, 4) in xyxy format
                - "scores": Tensor of shape (num_boxes,) with confidence scores
                - "labels": Tensor of shape (num_boxes,) with class labels
        """ 
        # Validate input
        if images.ndim != 4:
            raise ValueError(f"images must be (N,C,H,W); got shape {tuple(images.shape)}")
        if images.size(1) != 3:
            raise ValueError(f"images must have 3 channels (RGB); got C={images.size(1)}")

        # Convert to list of float32 [0,1] on the model device
        imgs_list: List[torch.Tensor] = []
        for img in images:
            if img.dtype == torch.uint8:
                t = img.to(self.device, non_blocking=True).float() / 255.0
            else:
                t = img.to(self.device, non_blocking=True).float()
                # assume already scaled reasonably; Faster R-CNN is robust
            imgs_list.append(t)

        # Forward
        raw_outputs = self.model(imgs_list)  # list of dicts with boxes/scores/labels on device

        # Post-filter by score and cap to max_det
        out: List[Dict[str, torch.Tensor]] = []
        for det in raw_outputs:
            boxes: torch.Tensor = det.get("boxes", torch.empty(0, 4, device=self.device))
            scores: torch.Tensor = det.get("scores", torch.empty(0, device=self.device))
            labels: torch.Tensor = det.get("labels", torch.empty(0, dtype=torch.long, device=self.device))

            if boxes.numel() == 0:
                out.append({
                    "boxes":  boxes.detach(),
                    "scores": scores.detach(),
                    "labels": labels.detach(),
                })
                continue

            keep = scores >= self.confidence
            if keep.any():
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            else:
                boxes = boxes[:0]; scores = scores[:0]; labels = labels[:0]

            if boxes.shape[0] > self.max_det:
                boxes  = boxes[: self.max_det]
                scores = scores[: self.max_det]
                labels = labels[: self.max_det]

            out.append({
                "boxes":  boxes.detach(),
                "scores": scores.detach(),
                "labels": labels.to(torch.int64).detach(),
            })

        return out
