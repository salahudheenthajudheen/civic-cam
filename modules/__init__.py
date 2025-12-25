# CivicCam Modules Package
"""
CivicCam - AI-powered roadside litter detection system.

This package contains the core modules for:
- Video capture and processing
- Object detection using YOLOv8
- Action recognition for littering detection
- License plate recognition (ALPR)
- Utility functions and configuration management
"""

from .utils import load_config, setup_logger, FPSCalculator
from .video_capture import VideoCapture
from .detector import ObjectDetector, Detection
from .action_recognizer import ActionRecognizer, LitteringEvent, draw_littering_alert
from .alpr import LicensePlateReader, PlateResult, draw_plate_info

__all__ = [
    'load_config',
    'setup_logger', 
    'FPSCalculator',
    'VideoCapture',
    'ObjectDetector',
    'Detection',
    'ActionRecognizer',
    'LitteringEvent',
    'draw_littering_alert',
    'LicensePlateReader',
    'PlateResult',
    'draw_plate_info'
]

__version__ = '1.2.0'
