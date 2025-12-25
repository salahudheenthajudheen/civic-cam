"""
CivicCam Object Detection Module

YOLOv8-based object detection with GPU/CPU support, configurable thresholds,
and real-time visualization capabilities.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


logger = logging.getLogger("civiccam.detector")


# COCO class names for reference
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


@dataclass
class Detection:
    """
    Represents a single detection result.
    
    Attributes:
        label: Class name of the detected object.
        confidence: Detection confidence score (0-1).
        bbox: Bounding box as (x1, y1, x2, y2).
        class_id: Integer class ID from the model.
    """
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    class_id: int
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Get the area of the bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary format."""
        return {
            'label': self.label,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'class_id': self.class_id,
            'center': self.center,
            'area': self.area
        }


class ObjectDetector:
    """
    YOLOv8-based object detector with configurable settings.
    
    Automatically uses GPU if available, with CPU fallback.
    Supports filtering by classes of interest and confidence thresholds.
    
    Attributes:
        model_path: Path to the YOLO model weights.
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.
        classes_of_interest: List of class names to detect.
    
    Example:
        >>> detector = ObjectDetector(model_path="yolov8n.pt")
        >>> detections = detector.detect(frame)
        >>> for det in detections:
        ...     print(f"{det.label}: {det.confidence:.2f}")
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        classes_of_interest: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLOv8 model weights.
            confidence_threshold: Minimum confidence score (0-1).
            iou_threshold: IoU threshold for NMS (0-1).
            classes_of_interest: List of class names to filter.
            device: Force device ('cuda', 'cpu', 'mps'). Auto-detect if None.
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics package not found. "
                "Install with: pip install ultralytics"
            )
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes_of_interest = classes_of_interest
        self.device = device
        
        self._model: Optional[YOLO] = None
        self._class_name_to_id: Dict[str, int] = {}
        self._filter_classes: Optional[List[int]] = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the YOLO model and setup class filtering."""
        try:
            logger.info(f"Loading YOLOv8 model: {self.model_path}")
            self._model = YOLO(self.model_path)
            
            # Determine device
            if self.device is None:
                import torch
                if torch.cuda.is_available():
                    self.device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cpu'
            
            logger.info(f"Using device: {self.device}")
            
            # Build class name to ID mapping
            if hasattr(self._model, 'names'):
                self._class_name_to_id = {
                    name.lower(): idx for idx, name in self._model.names.items()
                }
            else:
                # Fallback to COCO classes
                self._class_name_to_id = {
                    name.lower(): idx for idx, name in enumerate(COCO_CLASSES)
                }
            
            # Setup class filtering
            if self.classes_of_interest:
                self._filter_classes = []
                for class_name in self.classes_of_interest:
                    class_name_lower = class_name.lower()
                    if class_name_lower in self._class_name_to_id:
                        self._filter_classes.append(
                            self._class_name_to_id[class_name_lower]
                        )
                    else:
                        logger.warning(f"Unknown class: {class_name}")
                
                logger.info(
                    f"Filtering to classes: {self.classes_of_interest} "
                    f"(IDs: {self._filter_classes})"
                )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run object detection on a frame.
        
        Args:
            frame: Input image in BGR format (OpenCV format).
            
        Returns:
            List of Detection objects.
        """
        if self._model is None:
            logger.error("Model not loaded")
            return []
        
        try:
            # Run inference
            results = self._model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                classes=self._filter_classes,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get bounding box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Get confidence and class
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name
                    if hasattr(self._model, 'names'):
                        label = self._model.names.get(class_id, f"class_{class_id}")
                    else:
                        label = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
                    
                    detections.append(Detection(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model details.
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'classes_of_interest': self.classes_of_interest,
            'filter_class_ids': self._filter_classes
        }


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_thickness: int = 2,
    font_scale: float = 0.6,
    show_confidence: bool = True,
    show_labels: bool = True
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on a frame.
    
    Args:
        frame: Input image (will be modified in place).
        detections: List of Detection objects to draw.
        box_color: Bounding box color in BGR format.
        text_color: Label text color in BGR format.
        box_thickness: Line thickness for boxes.
        font_scale: Font scale for labels.
        show_confidence: Whether to show confidence scores.
        show_labels: Whether to show class labels.
        
    Returns:
        Frame with detections drawn.
    """
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        
        if show_labels:
            # Prepare label text
            if show_confidence:
                label_text = f"{det.label} {det.confidence:.2f}"
            else:
                label_text = det.label
            
            # Calculate label background size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                box_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                1,
                cv2.LINE_AA
            )
    
    return frame


def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.7
) -> np.ndarray:
    """
    Draw FPS counter on a frame.
    
    Args:
        frame: Input image.
        fps: Current FPS value.
        position: Text position (x, y).
        color: Text color in BGR format.
        font_scale: Font scale.
        
    Returns:
        Frame with FPS drawn.
    """
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA
    )
    return frame


def draw_detection_count(
    frame: np.ndarray,
    detections: List[Detection],
    position: Tuple[int, int] = (10, 60),
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw detection count summary on a frame.
    
    Args:
        frame: Input image.
        detections: List of detections.
        position: Text position (x, y).
        color: Text color in BGR format.
        font_scale: Font scale.
        
    Returns:
        Frame with detection count drawn.
    """
    # Count by class
    class_counts: Dict[str, int] = {}
    for det in detections:
        class_counts[det.label] = class_counts.get(det.label, 0) + 1
    
    # Create summary text
    if class_counts:
        summary = ", ".join(f"{k}: {v}" for k, v in class_counts.items())
    else:
        summary = "No detections"
    
    cv2.putText(
        frame,
        summary,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
        cv2.LINE_AA
    )
    return frame
