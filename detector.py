import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.7)


class VehicleDetector:
    """
    Vehicle detection using YOLOv8 model
    Filters detections to vehicle classes only
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        vehicle_classes: List[int] = [2, 3, 5, 7],
    ):
        """
        Initialize YOLOv8 detector

        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            vehicle_classes: COCO class IDs for vehicles (car, motorcycle, bus, truck)
        """
        self.model = YOLO(model_path)
        if torch.cuda.is_available():
            self.model.half()
        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = set(vehicle_classes)

        # COCO class names for reference
        self.class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in frame

        Args:
            frame: Input image frame

        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int, 'class_name': str}]
        """
        # Run inference
        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    # Extract detection data
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())

                    # Filter by confidence and vehicle classes
                    if (
                        confidence >= self.confidence_threshold
                        and class_id in self.vehicle_classes
                    ):
                        detection = {
                            "bbox": bbox.astype(int),
                            "confidence": float(confidence),
                            "class_id": class_id,
                            "class_name": self.class_names.get(class_id, "vehicle"),
                        }
                        detections.append(detection)

        return detections

    def format_detections_for_tracker(self, detections: List[Dict[str, Any]]) -> List:
        """
        Format detections for DeepSORT tracker

        Args:
            detections: List of detection dictionaries

        Returns:
            List in format [([x1, y1, x2, y2], confidence), ...]
        """
        if not detections:
            return []

        formatted = []
        for det in detections:
            bbox = det["bbox"]
            confidence = det["confidence"]
            # Format as tuple: (bbox_list, confidence)
            formatted.append(
                (
                    [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    float(confidence),
                )
            )

        return formatted
