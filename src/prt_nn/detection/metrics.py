from dataclasses import dataclass
import torch
from typing import List, Dict, Optional, Sequence, Any
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@dataclass
class DetectionMetrics:
    """
    A dataclass representing detection metrics.
    
    Attributes:
    map: float
        Mean Average Precision (mAP) score.
    """
    map_50_95: float               # mAP @[0.50:0.95]
    map_50: float                  # AP @0.50
    map_75: float                  # AP @0.75
    map_small: float               # AP for small objects
    map_medium: float              # AP for medium objects
    map_large: float               # AP for large objects
    mar_1: float                   # AR @1 detection per image
    mar_10: float                  # AR @10 detections per image
    mar_100: float                 # AR @100 detections per image
    mar_small: float               # AR for small objects
    mar_medium: float              # AR for medium objects
    mar_large: float               # AR for large objects
    map_per_class: Optional[List[float]] = None     # per-class AP @[0.50:0.95]
    mar_100_per_class: Optional[List[float]] = None # per-class AR @100
    classes: Optional[List[int]] = None             # class ids (if available)
    class_names: Optional[List[str]] = None         # names aligned with classes (optional)


class DetectionEvaluator:
    """
    A class to compute and store detection metrics.
    
    Methods:
    update(predictions: List[Dict[str, Tensor]], targets: List[Dict[str, Tensor]]) -> None
        Update the metrics with new predictions and targets.
    
    compute() -> DetectionMetrics
        Compute and return the current metrics.
    
    reset() -> None
        Reset the metrics to their initial state.
    """
    def __init__(
        self,
        box_format: str = "xyxy",
        iou_thresholds: Optional[Sequence[float]] = None,
        class_metrics: bool = True,
        class_names: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            box_format: 'xyxy' | 'xywh' | 'cxcywh'
            iou_thresholds: list of IoU thresholds (default is 0.50:0.95 step 0.05)
            class_metrics: if True, compute per-class AP/AR
            class_names: optional list of human-readable class names
        """
        self.metric = MeanAveragePrecision(
            box_format=box_format,
            iou_type="bbox",
            iou_thresholds=iou_thresholds,
            class_metrics=class_metrics,
        )
        self._class_names = list(class_names) if class_names is not None else None

    # ---- convenience one-shot API ----
    def evaluate(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> DetectionMetrics:
        self.reset()
        self.update(predictions, targets)
        return self.compute()

    # ---- streaming API ----
    def update(self, predictions: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> None:
        """
        Update internal accumulators with a batch of predictions/targets.
        """
        self.metric.update(predictions, targets)

    def compute(self) -> DetectionMetrics:
        """
        Compute COCO-style metrics over all updates so far and return a dataclass.
        """
        m: Dict[str, Any] = self.metric.compute()  # tensors dict

        def _to_float(x: Any) -> float:
            # Handles torch.Tensor, Python floats, and Nones -> NaN
            if x is None:
                return float("nan")
            if isinstance(x, torch.Tensor):
                return float(x.item())
            return float(x)

        def _to_list(x: Optional[torch.Tensor]) -> Optional[List[float]]:
            if x is None:
                return None
            return [float(v) for v in x.tolist()]

        classes_list: Optional[List[int]] = None
        class_names: Optional[List[str]] = None

        # TorchMetrics may expose 'classes' when class_metrics=True
        cls_tensor: Optional[torch.Tensor] = m.get("classes", None)
        if cls_tensor is not None:
            classes_list = [int(c) for c in cls_tensor.tolist()]
            if self._class_names is not None:
                # Map ids to names when possible; fall back to str(id) if out of range
                class_names = [
                    self._class_names[c] if 0 <= c < len(self._class_names) else str(c)
                    for c in classes_list
                ]

        return DetectionMetrics(
            map_50_95=_to_float(m.get("map")),
            map_50=_to_float(m.get("map_50")),
            map_75=_to_float(m.get("map_75")),
            map_small=_to_float(m.get("map_small")),
            map_medium=_to_float(m.get("map_medium")),
            map_large=_to_float(m.get("map_large")),
            mar_1=_to_float(m.get("mar_1")),
            mar_10=_to_float(m.get("mar_10")),
            mar_100=_to_float(m.get("mar_100")),
            mar_small=_to_float(m.get("mar_small")),
            mar_medium=_to_float(m.get("mar_medium")),
            mar_large=_to_float(m.get("mar_large")),
            map_per_class=_to_list(m.get("map_per_class", None)),
            mar_100_per_class=_to_list(m.get("mar_100_per_class", None)),
            classes=classes_list,
            class_names=class_names,
        )

    def reset(self) -> None:
        self.metric.reset()