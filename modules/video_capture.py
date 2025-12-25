"""
CivicCam Video Capture Module

Handles video capture from multiple sources: webcam, video files, and RTSP streams.
Includes automatic reconnection, frame buffering, and graceful error handling.
"""

import time
import logging
from typing import Optional, Tuple, Union, Generator
from pathlib import Path

import cv2
import numpy as np


logger = logging.getLogger("civiccam.video_capture")


class VideoCapture:
    """
    Video capture handler with multi-source support and automatic reconnection.
    
    Supports webcam indices, video file paths, and RTSP/HTTP stream URLs.
    Implements context manager protocol for safe resource management.
    
    Attributes:
        source: Video source (webcam index, file path, or URL).
        fps: Target frames per second.
        resolution: Target resolution as (width, height).
        reconnect_delay: Seconds to wait before reconnecting on failure.
    
    Example:
        >>> with VideoCapture(source=0, fps=30) as cap:
        ...     for frame in cap.frames():
        ...         process_frame(frame)
    """
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        fps: int = 30,
        resolution: Optional[Tuple[int, int]] = None,
        reconnect_delay: int = 5
    ):
        """
        Initialize video capture.
        
        Args:
            source: Webcam index (int), video file path, or stream URL.
            fps: Target FPS for the capture.
            resolution: Optional (width, height) tuple.
            reconnect_delay: Seconds between reconnection attempts.
        """
        self.source = source
        self.fps = fps
        self.resolution = resolution or (1280, 720)
        self.reconnect_delay = reconnect_delay
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_file = False
        self._is_stream = False
        self._frame_count = 0
        self._running = True
        
        self._detect_source_type()
    
    def _detect_source_type(self) -> None:
        """Detect the type of video source."""
        if isinstance(self.source, int):
            # Webcam index
            self._is_file = False
            self._is_stream = False
        elif isinstance(self.source, str):
            if self.source.startswith(('rtsp://', 'http://', 'https://')):
                # Network stream
                self._is_file = False
                self._is_stream = True
            else:
                # Video file
                self._is_file = True
                self._is_stream = False
    
    def open(self) -> bool:
        """
        Open the video capture device.
        
        Returns:
            True if successfully opened, False otherwise.
        """
        try:
            # Release existing capture if any
            if self._cap is not None:
                self._cap.release()
            
            # Open video source
            if isinstance(self.source, int):
                self._cap = cv2.VideoCapture(self.source)
            else:
                self._cap = cv2.VideoCapture(str(self.source))
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set resolution for webcam
            if not self._is_file:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Opened video source: {self.source}")
            logger.info(f"Resolution: {actual_width}x{actual_height}, FPS: {actual_fps:.1f}")
            
            self._frame_count = 0
            return True
            
        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the video source.
        
        Returns:
            Tuple of (success, frame). Frame is None if read failed.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None
        
        ret, frame = self._cap.read()
        
        if ret:
            self._frame_count += 1
        
        return ret, frame
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously.
        
        Handles reconnection for webcam/stream sources.
        For video files, loops back to start when finished.
        
        Yields:
            numpy.ndarray: Video frames in BGR format.
        """
        while self._running:
            ret, frame = self.read()
            
            if not ret:
                if self._is_file:
                    # Loop video file
                    logger.debug("Video file ended, looping...")
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Try to reconnect for webcam/stream
                    logger.warning("Lost connection, attempting to reconnect...")
                    time.sleep(self.reconnect_delay)
                    if self.open():
                        continue
                    else:
                        logger.error("Reconnection failed")
                        break
            else:
                yield frame
    
    def stop(self) -> None:
        """Stop the frame generator loop."""
        self._running = False
    
    def release(self) -> None:
        """Release video capture resources."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.debug("Video capture released")
    
    def get_properties(self) -> dict:
        """
        Get current video capture properties.
        
        Returns:
            Dictionary with width, height, fps, and frame_count.
        """
        if self._cap is None or not self._cap.isOpened():
            return {}
        
        return {
            'width': int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self._cap.get(cv2.CAP_PROP_FPS),
            'frame_count': self._frame_count,
            'total_frames': int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self._is_file else None
        }
    
    def __enter__(self) -> 'VideoCapture':
        """Context manager entry."""
        if not self.open():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()
    
    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        self.release()


def test_video_source(source: Union[int, str]) -> bool:
    """
    Test if a video source can be opened.
    
    Args:
        source: Video source to test.
        
    Returns:
        True if source is valid and can be opened.
    """
    try:
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except Exception:
        return False


def list_available_cameras(max_index: int = 10) -> list:
    """
    List available camera indices.
    
    Args:
        max_index: Maximum camera index to check.
        
    Returns:
        List of available camera indices.
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available
