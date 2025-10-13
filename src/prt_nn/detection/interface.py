import torch
from typing import Protocol, List, Dict
from prt_nn.detection.metrics import DetectionEvaluator, DetectionMetrics, SingleImageDetectionMetrics

Prediction = Dict[str, torch.Tensor]  # {"boxes": (B,4), "scores": (B,), "labels": (B,)}
Target  = Dict[str, torch.Tensor]     # {"boxes": (B,4), "labels": (B,)}

class BaseDetector(Protocol):
    def predict(self, images: torch.Tensor) -> List[Prediction]:
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
        ...

class DetectorInterface:
    """
    Protocol for object detection models.
    
    Methods:
    predict(images: torch.Tensor) -> List[BoundingBox]
        Perform inference on a batch of images and return bounding boxes, scores, and labels.
    
    loss(predictions: List[BoundingBox], targets: List[BoundingBox]) -> torch.Tensor
        Compute the loss given predictions and ground truth targets.
    """
    def __init__(self, model: BaseDetector):
        self.model = model
        self.evaluator = DetectionEvaluator()

    @torch.inference_mode()
    def detect(self, images: torch.Tensor) -> List[Prediction]:
        """
        Perform inference on a batch of images.

        Args:
            images (torch.Tensor): A batch of images as a tensor of shape (N, C, H, W).
        Returns:
            A list of predictions, one per image. Each prediction is a dict with keys:
                - "boxes": Tensor of shape (num_boxes, 4) in xyxy format
                - "scores": Tensor of shape (num_boxes,) with confidence scores
                - "labels": Tensor of shape (num_boxes,) with class labels
        """
        return self.model.predict(images)
    
    @torch.inference_mode()
    def evaluate(self, predictions: List[Prediction], targets: List[Target]) -> DetectionMetrics:
        """
        Evaluate predictions against targets and return detection metrics.

        Args:
            predictions (List[Dict[str, torch.Tensor]]): List of prediction dicts from the model.
            targets (List[Dict[str, torch.Tensor]]): List of ground truth dicts.
        Returns:
            DetectionMetrics object containing various evaluation metrics.
        """
        return self.evaluator.evaluate(predictions, targets)
    
    def evaluate_image(self, prediction: Prediction, target: Target) -> SingleImageDetectionMetrics:
        """
        Evaluate a single image's prediction against its target and return detection metrics.

        Args:
            prediction (Dict[str, torch.Tensor]): Prediction dict from the model for a single image.
            target (Dict[str, torch.Tensor]): Ground truth dict for a single image.
        Returns:
            SingleImageDetectionMetrics object containing evaluation metrics for the image.
        """
        return self.evaluator.evaluate_image(prediction, target)

    def reset_metrics(self) -> None:
        """
        Reset the internal state of the evaluator.
        """
        self.evaluator.reset()

    @torch.inference_mode()
    def update_metrics(self, predictions: List[Prediction], targets: List[Target]) -> None:
        """
        Update the evaluator with a new batch of predictions and targets.
        
        Args:
            predictions (List[Dict[str, torch.Tensor]]): List of prediction dicts from the model.
            targets (List[Dict[str, torch.Tensor]]): List of ground truth dicts.
        """
        self.evaluator.update(predictions, targets)

    def compute_metrics(self) -> DetectionMetrics:
        """
        Compute and return the current detection metrics.

        Returns:
            DetectionMetrics object containing various evaluation metrics.
        """
        return self.evaluator.compute()
    
if __name__ == "__main__":
    from prt_nn.detection.yolo import YoloDetector
    model = YoloDetector()

    # from prt_nn.detection.fast_rcnn import FastRCNNDetector
    # model = FastRCNNDetector()

    interface = DetectorInterface(model)
    dummy_images = (torch.randn(2, 3, 640, 640) * 255).to(torch.uint8)
    preds = interface.detect(dummy_images)
    print(preds)