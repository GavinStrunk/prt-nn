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

@dataclass(frozen=True)
class SingleImageDetectionMetrics:
    """Per-image detection metrics (IoU-based matching, per class)."""
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou_tp: float
    n_matched: int

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
    
    def evaluate_image(
        self,
        prediction: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> SingleImageDetectionMetrics:
        """
        Evaluate a single image's prediction against its target and return metrics.

        Matching: greedy, per-class, by descending score. A match requires IoU >= 0.5.
        If both predictions and targets are empty, returns perfect precision/recall/F1 = 1.0.

        Args:
            prediction: {"boxes": (Np,4) xyxy, "scores": (Np,), "labels": (Np,)}
            target:     {"boxes": (Ng,4) xyxy,           "labels": (Ng,)}

        Returns:
            DetectionImageMetrics
        """
        iou_thr = 0.5  # keep simple; make a parameter later if you want

        pb = prediction["boxes"]
        ps = prediction["scores"]
        pl = prediction["labels"]
        gb = target["boxes"]
        gl = target["labels"]

        # Both empty: correct "no objects" case
        if pb.numel() == 0 and gb.numel() == 0:
            return SingleImageDetectionMetrics(
                true_positives=0, false_positives=0, false_negatives=0,
                precision=1.0, recall=1.0, f1=1.0,
                mean_iou_tp=0.0, n_matched=0
            )

        def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if a.numel() == 0 or b.numel() == 0:
                return a.new_zeros((a.shape[0], b.shape[0]))
            tl = torch.maximum(a[:, None, :2], b[None, :, :2])
            br = torch.minimum(a[:, None, 2:], b[None, :, 2:])
            wh = (br - tl).clamp(min=0)
            inter = wh[..., 0] * wh[..., 1]
            area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
            area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
            union = area_a[:, None] + area_b[None, :] - inter
            return inter / union.clamp(min=1e-9)

        # Classes present in preds or GTs
        classes = torch.unique(
            torch.cat([pl.unique() if pl.numel() else pl.new_empty(0),
                       gl.unique() if gl.numel() else gl.new_empty(0)])
        ) if (pl.numel() or gl.numel()) else pl.new_empty(0, dtype=torch.long)

        tp = fp = fn = 0
        ious_tp = []

        for c in classes.tolist():
            pm = (pl == c)
            gm = (gl == c)
            pb_c = pb[pm]
            ps_c = ps[pm]
            gb_c = gb[gm]

            if pb_c.numel() == 0 and gb_c.numel() == 0:
                continue
            if pb_c.numel() == 0:
                fn += gb_c.shape[0]
                continue
            if gb_c.numel() == 0:
                fp += pb_c.shape[0]
                continue

            # sort predictions by score descending
            order = torch.argsort(ps_c, descending=True)
            pb_c = pb_c[order]

            iou = box_iou(pb_c, gb_c)  # [Np_c, Ng_c]
            matched_g = torch.zeros(gb_c.shape[0], dtype=torch.bool, device=gb_c.device)

            for i in range(pb_c.shape[0]):
                j = torch.argmax(iou[i])
                if iou[i, j] >= iou_thr and not matched_g[j]:
                    tp += 1
                    matched_g[j] = True
                    ious_tp.append(float(iou[i, j]))
                else:
                    fp += 1

            fn += int((~matched_g).sum().item())

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall    = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        mean_iou  = (sum(ious_tp) / len(ious_tp)) if ious_tp else 0.0

        return SingleImageDetectionMetrics(
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou_tp=mean_iou,
            n_matched=len(ious_tp),
        )

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