"""
Detection package for prt_nn.

"""
from .interface import DetectorInterface
from .fast_rcnn import FastRCNNDetector
from .yolo import YoloDetector

__all__ = [
    "DetectorInterface",
    "FastRCNNDetector",
    "YoloDetector",
]