#!/usr/bin/env python3
"""
CivicCam - AI-Powered Roadside Litter Detection System

Main entry point for real-time video processing and object detection.
Detects littering incidents using YOLOv8 computer vision.

Usage:
    python main.py                          # Use webcam (default)
    python main.py --source 0               # Use webcam index 0
    python main.py --source video.mp4       # Use video file
    python main.py --source rtsp://...      # Use RTSP stream
    python main.py --no-display             # Run without display window
    python main.py --config custom.yaml     # Use custom config file
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from modules.utils import (
    load_config,
    setup_logger,
    check_gpu,
    FPSCalculator,
    ensure_directory
)
from modules.video_capture import VideoCapture
from modules.detector import (
    ObjectDetector,
    draw_detections,
    draw_fps,
    draw_detection_count
)
from modules.action_recognizer import (
    ActionRecognizer,
    draw_littering_alert
)


# Global flag for graceful shutdown
_running = True


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _running
    _running = False
    print("\n⚡ Shutdown signal received. Cleaning up...")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CivicCam - AI-Powered Roadside Litter Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Run with webcam
  python main.py --source video.mp4  # Process video file
  python main.py --source 1          # Use camera index 1
  python main.py --no-display        # Headless mode
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Video source: webcam index (0,1,..), file path, or RTSP URL'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--display', '-d',
        action='store_true',
        default=True,
        help='Show display window with detections (default: True)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display window (headless mode)'
    )
    
    parser.add_argument(
        '--save-detections',
        action='store_true',
        help='Save detection results to file'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Override YOLO model path from config'
    )
    
    parser.add_argument(
        '--confidence', '-conf',
        type=float,
        default=None,
        help='Override confidence threshold (0-1)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CivicCam detection system."""
    global _running
    
    # Parse arguments
    args = parse_args()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print("   Create config.yaml or specify a different path with --config")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)
    
    # Setup logging
    log_level = "DEBUG" if args.debug else config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('log_file')
    console_output = config.get('logging', {}).get('console_output', True)
    
    logger = setup_logger(
        name="civiccam",
        level=log_level,
        log_file=log_file,
        console_output=console_output
    )
    
    logger.info("=" * 60)
    logger.info("🎥 CivicCam - AI-Powered Litter Detection System")
    logger.info("=" * 60)
    
    # Check GPU availability
    gpu_info = check_gpu()
    logger.info(f"Device: {gpu_info['device'].upper()}")
    if gpu_info['device_name']:
        logger.info(f"GPU: {gpu_info['device_name']}")
    
    # Ensure directories exist
    evidence_path = config.get('storage', {}).get('evidence_path', './data/evidence')
    ensure_directory(evidence_path)
    if log_file:
        ensure_directory(Path(log_file).parent)
    
    # Determine video source
    source = args.source
    if source is None:
        source = config.get('camera', {}).get('source', 0)
    elif source.isdigit():
        source = int(source)
    
    # Get camera settings
    camera_config = config.get('camera', {})
    fps = camera_config.get('fps', 30)
    resolution = tuple(camera_config.get('resolution', [1280, 720]))
    reconnect_delay = camera_config.get('reconnect_delay', 5)
    
    # Get detection settings
    detection_config = config.get('detection', {})
    model_path = args.model or detection_config.get('model', 'yolov8n.pt')
    confidence = args.confidence or detection_config.get('confidence_threshold', 0.5)
    iou_threshold = detection_config.get('iou_threshold', 0.45)
    classes_of_interest = detection_config.get('classes_of_interest')
    
    # Get display settings
    display_config = config.get('display', {})
    show_display = args.display and not args.no_display
    window_name = display_config.get('window_name', 'CivicCam - Real-time Detection')
    box_color = tuple(display_config.get('box_color', [0, 255, 0]))
    text_color = tuple(display_config.get('text_color', [255, 255, 255]))
    box_thickness = display_config.get('box_thickness', 2)
    font_scale = display_config.get('font_scale', 0.6)
    show_fps = display_config.get('show_fps', True)
    show_labels = display_config.get('show_labels', True)
    show_confidence = display_config.get('show_confidence', True)
    
    # Get action recognition settings
    action_config = config.get('action_recognition', {})
    action_recognition_enabled = action_config.get('enable', False)
    
    # Initialize object detector
    logger.info(f"Loading model: {model_path}")
    try:
        detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            iou_threshold=iou_threshold,
            classes_of_interest=classes_of_interest
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        sys.exit(1)
    
    # Initialize action recognizer
    action_recognizer = None
    if action_recognition_enabled:
        logger.info("Initializing action recognition...")
        action_recognizer = ActionRecognizer(
            proximity_threshold=action_config.get('proximity_threshold', 150),
            fall_threshold=action_config.get('fall_threshold', 50),
            separation_threshold=action_config.get('separation_threshold', 200),
            validation_frames=action_config.get('validation_frames', 5),
            ground_level_ratio=action_config.get('ground_level_ratio', 0.8),
            cooldown_seconds=action_config.get('cooldown_seconds', 3.0),
            frame_height=resolution[1]
        )
        logger.info("✅ Action recognition enabled")
    
    # Initialize FPS calculator
    fps_calc = FPSCalculator(window_size=30)
    
    # Initialize video capture
    logger.info(f"Opening video source: {source}")
    
    try:
        with VideoCapture(
            source=source,
            fps=fps,
            resolution=resolution,
            reconnect_delay=reconnect_delay
        ) as cap:
            
            logger.info("✅ Video capture started successfully")
            logger.info("Press 'q' to quit, 's' to save current frame")
            
            frame_count = 0
            littering_events_count = 0
            recent_events = []  # Store recent events for display
            detection_count = 0
            
            for frame in cap.frames():
                if not _running:
                    break
                
                frame_count += 1
                
                # Run object detection
                detections = detector.detect(frame)
                detection_count += len(detections)
                
                # Draw detections on frame
                if detections:
                    frame = draw_detections(
                        frame,
                        detections,
                        box_color=box_color,
                        text_color=text_color,
                        box_thickness=box_thickness,
                        font_scale=font_scale,
                        show_confidence=show_confidence,
                        show_labels=show_labels
                    )
                
                # Run action recognition
                if action_recognizer:
                    new_events = action_recognizer.update(detections)
                    if new_events:
                        littering_events_count += len(new_events)
                        recent_events = new_events
                        for event in new_events:
                            # Save evidence frame
                            save_path = Path(evidence_path) / f"littering_{event.event_id}.jpg"
                            cv2.imwrite(str(save_path), frame)
                            logger.info(f"Evidence saved: {save_path}")
                    
                    # Draw alerts for recent events (last 2 seconds)
                    for event in action_recognizer.get_recent_events(seconds=2.0):
                        frame = draw_littering_alert(frame, event)
                
                # Calculate and draw FPS
                current_fps = fps_calc.update()
                if show_fps:
                    frame = draw_fps(frame, current_fps)
                
                # Draw detection count
                frame = draw_detection_count(frame, detections)
                
                # Draw littering event count if action recognition enabled
                if action_recognizer:
                    cv2.putText(
                        frame,
                        f"Littering Events: {littering_events_count}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255) if littering_events_count > 0 else (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )
                
                # Log detections periodically
                if frame_count % 100 == 0:
                    logger.debug(
                        f"Frame {frame_count}: {len(detections)} detections, "
                        f"FPS: {current_fps:.1f}"
                    )
                
                # Display frame
                if show_display:
                    cv2.imshow(window_name, frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        logger.info("Quit requested by user")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        save_path = Path(evidence_path) / f"frame_{frame_count}.jpg"
                        cv2.imwrite(str(save_path), frame)
                        logger.info(f"Saved frame to: {save_path}")
            
            # Summary statistics
            logger.info("-" * 60)
            logger.info("📊 Session Summary")
            logger.info(f"   Total frames processed: {frame_count}")
            logger.info(f"   Total detections: {detection_count}")
            if frame_count > 0:
                logger.info(f"   Average detections/frame: {detection_count/frame_count:.2f}")
            if action_recognizer:
                logger.info(f"   🚨 Littering events detected: {littering_events_count}")
                stats = action_recognizer.get_stats()
                logger.info(f"   Objects tracked: {stats['objects_tracked']}")
                logger.info(f"   Persons tracked: {stats['persons_tracked']}")
            logger.info("-" * 60)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        logger.info("🛑 CivicCam shutdown complete")


if __name__ == "__main__":
    main()
