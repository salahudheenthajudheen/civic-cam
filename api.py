#!/usr/bin/env python3
"""
CivicCam API Server with Live Video Streaming

FastAPI server providing:
- Live MJPEG video stream with detection overlays
- REST API for events and statistics
- WebSocket for real-time event notifications

Usage:
    python api.py                    # Start server on port 8000
    python api.py --port 8080        # Custom port
"""

import argparse
import asyncio
import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Generator
from contextlib import asynccontextmanager

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use system env vars

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from modules.utils import load_config, setup_logger, FPSCalculator
from modules.video_capture import VideoCapture
from modules.detector import ObjectDetector, draw_detections, draw_fps, draw_detection_count
from modules.action_recognizer import ActionRecognizer, draw_littering_alert
from modules.alpr import LicensePlateReader, draw_plate_info
from modules.telegram_notifier import TelegramNotifier, create_notifier_from_config


# Setup logging
logger = setup_logger(name="civiccam.api", level="INFO", console_output=True)


# ============================================================================
# Data Models
# ============================================================================

class LitteringEventResponse(BaseModel):
    id: str
    timestamp: str
    camera: str
    vehicleDetected: bool
    vehicleStatus: str
    imageUrl: str
    faceDetected: bool
    faceBox: Optional[dict] = None
    objectType: str
    confidence: float


class StatsResponse(BaseModel):
    totalIncidents: int
    incidentsWithVehicle: int
    lastIncident: str
    activeCameras: int
    detectionFps: float


# ============================================================================
# Video Stream Manager (runs detection in background thread)
# ============================================================================

class VideoStreamManager:
    """Manages video capture, detection, and streaming."""
    
    def __init__(self):
        self.config = None
        self.cap = None
        self.detector = None
        self.action_recognizer = None
        self.plate_reader = None
        self.telegram_notifier = None
        self.fps_calc = FPSCalculator(window_size=30)
        
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        
        self.events: List[dict] = []
        self.frame_count = 0
        self.detection_count = 0
        self.littering_count = 0
        self.plates_read = 0
        self.current_fps = 0.0
        self.last_plate = None
        
        self.evidence_path = Path("./data/evidence")
        self.ws_manager = None  # Set by app
    
    def initialize(self, config: dict):
        """Initialize detector and action recognizer."""
        self.config = config
        self.evidence_path = Path(config.get("storage", {}).get("evidence_path", "./data/evidence"))
        self.evidence_path.mkdir(parents=True, exist_ok=True)
        
        # Load detection settings
        detection_config = config.get("detection", {})
        model_path = detection_config.get("model", "yolov8n.pt")
        confidence = detection_config.get("confidence_threshold", 0.5)
        iou_threshold = detection_config.get("iou_threshold", 0.45)
        classes = detection_config.get("classes_of_interest")
        
        logger.info(f"Loading YOLOv8 model: {model_path}")
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            iou_threshold=iou_threshold,
            classes_of_interest=classes
        )
        
        # Load action recognition settings
        action_config = config.get("action_recognition", {})
        if action_config.get("enable", False):
            resolution = config.get("camera", {}).get("resolution", [1280, 720])
            self.action_recognizer = ActionRecognizer(
                proximity_threshold=action_config.get("proximity_threshold", 150),
                fall_threshold=action_config.get("fall_threshold", 50),
                separation_threshold=action_config.get("separation_threshold", 200),
                validation_frames=action_config.get("validation_frames", 5),
                ground_level_ratio=action_config.get("ground_level_ratio", 0.8),
                cooldown_seconds=action_config.get("cooldown_seconds", 3.0),
                frame_height=resolution[1]
            )
            logger.info("Action recognition enabled")
        
        # Load ALPR settings
        alpr_config = config.get("alpr", {})
        if alpr_config.get("enable", False):
            try:
                self.plate_reader = LicensePlateReader(
                    confidence_threshold=alpr_config.get("confidence_threshold", 0.5),
                    plate_region_ratio=alpr_config.get("plate_region_ratio", 0.35),
                    languages=alpr_config.get("languages", ["en"])
                )
                logger.info("ALPR enabled")
            except Exception as e:
                logger.warning(f"Could not initialize ALPR: {e}")
        
        # Initialize Telegram notifier
        self.telegram_notifier = create_notifier_from_config(config)
        if self.telegram_notifier and self.telegram_notifier.enabled:
            logger.info("Telegram notifications enabled")
    
    def start(self):
        """Start video capture and processing thread."""
        if self.running:
            return
        
        camera_config = self.config.get("camera", {})
        source = camera_config.get("source", 0)
        fps = camera_config.get("fps", 30)
        resolution = tuple(camera_config.get("resolution", [1280, 720]))
        
        logger.info(f"Opening video source: {source}")
        self.cap = VideoCapture(source=source, fps=fps, resolution=resolution)
        
        if not self.cap.open():
            logger.error("Failed to open video source")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info("Video streaming started")
    
    def stop(self):
        """Stop video capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Video streaming stopped")
    
    def _process_loop(self):
        """Main processing loop (runs in background thread)."""
        display_config = self.config.get("display", {})
        box_color = tuple(display_config.get("box_color", [0, 255, 0]))
        text_color = tuple(display_config.get("text_color", [255, 255, 255]))
        last_detections = []
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            self.frame_count += 1
            
            # Run detection every 2nd frame for smoother video
            if self.frame_count % 2 == 0:
                detections = self.detector.detect(frame)
                last_detections = detections
                self.detection_count += len(detections)
            else:
                detections = last_detections
            
            # Draw detections
            if detections:
                frame = draw_detections(
                    frame, detections,
                    box_color=box_color,
                    text_color=text_color,
                    show_confidence=True,
                    show_labels=True
                )
            
            # Run ALPR on vehicles
            if self.plate_reader:
                vehicle_classes = {'car', 'truck', 'bus', 'motorcycle'}
                for det in detections:
                    if det.label.lower() in vehicle_classes:
                        plate_result = self.plate_reader.read_plate(frame, det.bbox)
                        if plate_result and plate_result.confidence > 0.3:
                            self.plates_read += 1
                            self.last_plate = plate_result.text
                            frame = draw_plate_info(frame, plate_result, det.bbox)
                            if plate_result.is_valid:
                                logger.info(f"🚗 Plate detected: {plate_result.text}")
            
            # Run action recognition
            if self.action_recognizer:
                new_events = self.action_recognizer.update(detections)
                if new_events:
                    self.littering_count += len(new_events)
                    for event in new_events:
                        # Save evidence
                        save_path = self.evidence_path / f"littering_{event.event_id}.jpg"
                        cv2.imwrite(str(save_path), frame)
                        logger.warning(f"🚨 LITTERING: {event.object_label} - saved to {save_path}")
                        
                        # Build event data with plate info
                        event_dict = self._event_to_dict(event)
                        if self.last_plate:
                            event_dict["vehicleDetected"] = True
                            event_dict["plateNumber"] = self.last_plate
                            event_dict["vehicleStatus"] = f"Vehicle detected: {self.last_plate}"
                        
                        # Add to events list
                        self._add_event(event)
                        
                        # Send Telegram notification
                        if self.telegram_notifier:
                            self.telegram_notifier.send_incident_alert_sync(
                                event_dict,
                                save_path
                            )
                        
                        # Broadcast via WebSocket (async)
                        if self.ws_manager:
                            asyncio.run(self.ws_manager.broadcast({
                                "type": "new_event",
                                "data": event_dict
                            }))
                
                # Draw alerts
                for event in self.action_recognizer.get_recent_events(seconds=2.0):
                    frame = draw_littering_alert(frame, event)
            
            # Draw FPS and counts
            self.current_fps = self.fps_calc.update()
            frame = draw_fps(frame, self.current_fps)
            frame = draw_detection_count(frame, detections)
            
            # Draw littering count
            cv2.putText(
                frame,
                f"Littering Events: {self.littering_count}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if self.littering_count > 0 else (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            
            # Draw last plate read
            if self.last_plate:
                cv2.putText(
                    frame,
                    f"Last Plate: {self.last_plate}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )
            
            # Update current frame (thread-safe)
            with self.frame_lock:
                self.current_frame = frame.copy()
    
    def _add_event(self, event):
        """Add event to the list."""
        event_dict = self._event_to_dict(event)
        self.events.insert(0, event_dict)
        if len(self.events) > 50:
            self.events = self.events[:50]
    
    def _event_to_dict(self, event) -> dict:
        """Convert LitteringEvent to dict."""
        return {
            "id": event.event_id,
            "timestamp": datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "camera": "Camera 1",
            "vehicleDetected": False,
            "vehicleStatus": "No vehicle detected",
            "imageUrl": f"/api/evidence/littering_{event.event_id}.jpg",
            "faceDetected": True,
            "faceBox": {"x": 200, "y": 100, "width": 150, "height": 180},
            "objectType": event.object_label,
            "confidence": event.confidence
        }
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame (thread-safe)."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def generate_mjpeg(self) -> Generator[bytes, None, None]:
        """Generate MJPEG stream at 60 FPS."""
        while self.running:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Encode as JPEG (lower quality for speed)
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ret:
                continue
            
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
            )
            
            time.sleep(0.016)  # ~60 FPS
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "totalIncidents": self.littering_count,
            "incidentsWithVehicle": 0,
            "lastIncident": self.events[0]["timestamp"] if self.events else "",
            "activeCameras": 1 if self.running else 0,
            "detectionFps": round(self.current_fps, 1)
        }


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Active: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        data = json.dumps(message)
        for conn in list(self.active_connections):
            try:
                await conn.send_text(data)
            except:
                self.active_connections.discard(conn)


# ============================================================================
# Global State
# ============================================================================

stream_manager = VideoStreamManager()
ws_manager = ConnectionManager()


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CivicCam API server...")
    
    try:
        config = load_config("config.yaml")
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        config = {"camera": {"source": 0}, "detection": {"model": "yolov8n.pt"}}
    
    stream_manager.ws_manager = ws_manager
    stream_manager.initialize(config)
    stream_manager.start()
    
    logger.info("🎥 Video streaming ready at /api/video")
    
    yield
    
    stream_manager.stop()
    logger.info("API server shutdown")


app = FastAPI(
    title="CivicCam API",
    description="Real-time littering detection with live video streaming",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "CivicCam API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "video": "/api/video",
            "events": "/api/events",
            "stats": "/api/stats",
            "websocket": "/ws"
        }
    }


@app.get("/api/video")
async def video_feed():
    """Live MJPEG video stream with detection overlays."""
    return StreamingResponse(
        stream_manager.generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/events")
async def get_events(limit: int = 20):
    """Get recent littering events."""
    # Scan evidence folder for any saved events
    events = stream_manager.events.copy()
    
    # Also scan for files
    for img_file in sorted(stream_manager.evidence_path.glob("littering_*.jpg"), reverse=True):
        event_id = img_file.stem.replace("littering_", "")
        if not any(e["id"] == event_id for e in events):
            mtime = img_file.stat().st_mtime
            events.append({
                "id": event_id,
                "timestamp": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "camera": "Camera 1",
                "vehicleDetected": False,
                "vehicleStatus": "No vehicle detected",
                "imageUrl": f"/api/evidence/{img_file.name}",
                "faceDetected": True,
                "faceBox": {"x": 200, "y": 100, "width": 150, "height": 180},
                "objectType": "unknown",
                "confidence": 0.85
            })
    
    # Sort by timestamp and return
    events.sort(key=lambda x: x["timestamp"], reverse=True)
    return events[:limit]


@app.get("/api/stats")
async def get_stats():
    """Get detection statistics."""
    return stream_manager.get_stats()


@app.get("/api/evidence/{filename}")
async def get_evidence(filename: str):
    """Serve evidence images."""
    file_path = stream_manager.evidence_path / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/jpeg")


@app.post("/api/send-telegram/{event_id}")
async def send_telegram_notification(event_id: str):
    """Manually send a Telegram notification for an incident."""
    # Check if Telegram is enabled
    if not stream_manager.telegram_notifier or not stream_manager.telegram_notifier.enabled:
        raise HTTPException(status_code=503, detail="Telegram notifications not configured")
    
    # Find the event
    events = await get_events(50)
    event_data = None
    for event in events:
        if event["id"] == event_id:
            event_data = event
            break
    
    if not event_data:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Get image path
    image_url = event_data.get("imageUrl", "")
    filename = image_url.split("/")[-1] if image_url else None
    image_path = stream_manager.evidence_path / filename if filename else None
    
    # Send notification
    try:
        success = await stream_manager.telegram_notifier.send_incident_alert(
            event_data,
            image_path
        )
        if success:
            return {"status": "sent", "message": "Telegram notification sent successfully"}
        else:
            return {"status": "skipped", "message": "Notification was rate-limited or failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "initial",
            "data": stream_manager.events[:10]
        }))
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============================================================================
# Main
# ============================================================================

def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="CivicCam API Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    uvicorn.run("api:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
