"""
CivicCam Utilities Module

Contains utility functions for configuration loading, logging setup,
and performance monitoring.
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from collections import deque

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing configuration settings.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is malformed.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logger(
    name: str = "civiccam",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger instance.
    
    Args:
        name: Logger name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
        console_output: Whether to output logs to console.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def check_gpu() -> Dict[str, Any]:
    """
    Check GPU availability and return device information.
    
    Returns:
        Dictionary with GPU availability status and device info.
    """
    import torch
    
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device': 'cpu',
        'device_count': 0,
        'device_name': None
    }
    
    if torch.cuda.is_available():
        gpu_info['device'] = 'cuda'
        gpu_info['device_count'] = torch.cuda.device_count()
        gpu_info['device_name'] = torch.cuda.get_device_name(0)
    
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpu_info['device'] = 'mps'
        gpu_info['device_name'] = 'Apple Silicon GPU'
    
    return gpu_info


class FPSCalculator:
    """
    Calculate and track frames per second (FPS) over a sliding window.
    
    Attributes:
        window_size: Number of frames to average over.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS calculator.
        
        Args:
            window_size: Number of frames to use for averaging.
        """
        self.window_size = window_size
        self.timestamps: deque = deque(maxlen=window_size)
        self._last_time = time.perf_counter()
    
    def update(self) -> float:
        """
        Update with a new frame and return current FPS.
        
        Returns:
            Current FPS value.
        """
        current_time = time.perf_counter()
        self.timestamps.append(current_time)
        
        if len(self.timestamps) < 2:
            return 0.0
        
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed > 0:
            return (len(self.timestamps) - 1) / elapsed
        return 0.0
    
    def reset(self) -> None:
        """Reset the FPS calculator."""
        self.timestamps.clear()
        self._last_time = time.perf_counter()


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists.
        
    Returns:
        Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def format_detection_info(detections: list, include_confidence: bool = True) -> str:
    """
    Format detection results as a readable string.
    
    Args:
        detections: List of Detection objects.
        include_confidence: Whether to include confidence scores.
        
    Returns:
        Formatted string of detections.
    """
    if not detections:
        return "No detections"
    
    items = []
    for det in detections:
        if include_confidence:
            items.append(f"{det.label} ({det.confidence:.2f})")
        else:
            items.append(det.label)
    
    return ", ".join(items)
