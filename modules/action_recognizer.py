"""
CivicCam Action Recognition Module

Detects littering actions through motion analysis, person-object tracking,
and temporal validation using rule-based heuristics.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import deque
import uuid

import numpy as np

from .detector import Detection


logger = logging.getLogger("civiccam.action_recognizer")


@dataclass
class TrackedObject:
    """
    Represents an object being tracked across frames.
    
    Attributes:
        track_id: Unique identifier for this track
        label: Object class name
        positions: History of bounding box positions
        timestamps: History of detection timestamps
        confidence: Latest detection confidence
        is_person: Whether this is a person
        associated_person_id: If litter, the person track_id it was near
    """
    track_id: str
    label: str
    positions: deque = field(default_factory=lambda: deque(maxlen=30))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=30))
    confidence: float = 0.0
    is_person: bool = False
    associated_person_id: Optional[str] = None
    last_seen: float = 0.0
    
    @property
    def current_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get the most recent bounding box."""
        return self.positions[-1] if self.positions else None
    
    @property
    def center(self) -> Optional[Tuple[int, int]]:
        """Get center of current bounding box."""
        if not self.positions:
            return None
        x1, y1, x2, y2 = self.positions[-1]
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def bottom_center(self) -> Optional[Tuple[int, int]]:
        """Get bottom center (feet position for persons)."""
        if not self.positions:
            return None
        x1, y1, x2, y2 = self.positions[-1]
        return ((x1 + x2) // 2, y2)
    
    def get_displacement(self, frames: int = 5) -> Tuple[float, float]:
        """
        Calculate displacement over last N frames.
        
        Returns:
            (dx, dy) displacement in pixels
        """
        if len(self.positions) < 2:
            return (0.0, 0.0)
        
        n = min(frames, len(self.positions))
        old_pos = self.positions[-n]
        new_pos = self.positions[-1]
        
        old_center = ((old_pos[0] + old_pos[2]) / 2, (old_pos[1] + old_pos[3]) / 2)
        new_center = ((new_pos[0] + new_pos[2]) / 2, (new_pos[1] + new_pos[3]) / 2)
        
        return (new_center[0] - old_center[0], new_center[1] - old_center[1])
    
    def get_vertical_velocity(self, frames: int = 3) -> float:
        """
        Calculate vertical velocity (positive = downward).
        
        Returns:
            Vertical velocity in pixels per frame
        """
        if len(self.positions) < 2:
            return 0.0
        
        n = min(frames, len(self.positions))
        _, dy = self.get_displacement(n)
        return dy / n


@dataclass  
class LitteringEvent:
    """
    Represents a detected littering event.
    
    Attributes:
        event_id: Unique identifier
        timestamp: When the event was detected
        person_track_id: Track ID of the person involved
        object_track_id: Track ID of the littered object
        object_label: Type of object littered
        confidence: Detection confidence
        person_bbox: Person's bounding box at time of event
        object_bbox: Object's bounding box at time of event
        frame_number: Frame number when detected
    """
    event_id: str
    timestamp: float
    person_track_id: str
    object_track_id: str
    object_label: str
    confidence: float
    person_bbox: Tuple[int, int, int, int]
    object_bbox: Tuple[int, int, int, int]
    frame_number: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'person_track_id': self.person_track_id,
            'object_track_id': self.object_track_id,
            'object_label': self.object_label,
            'confidence': self.confidence,
            'person_bbox': self.person_bbox,
            'object_bbox': self.object_bbox,
            'frame_number': self.frame_number
        }


class ObjectTracker:
    """
    Simple IoU-based object tracker for maintaining identity across frames.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: float = 1.0):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU to match detections
            max_age: Seconds before track is considered lost
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[str, TrackedObject] = {}
    
    def update(self, detections: List[Detection], timestamp: float) -> Dict[str, TrackedObject]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of Detection objects from current frame
            timestamp: Current timestamp
            
        Returns:
            Dictionary of active tracks
        """
        # Remove stale tracks
        stale_ids = [
            tid for tid, track in self.tracks.items()
            if timestamp - track.last_seen > self.max_age
        ]
        for tid in stale_ids:
            del self.tracks[tid]
        
        # Match detections to existing tracks
        unmatched_detections = list(detections)
        matched_track_ids: Set[str] = set()
        
        for detection in detections:
            best_match_id = None
            best_iou = self.iou_threshold
            
            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                if track.label != detection.label:
                    continue
                
                iou = self._calculate_iou(detection.bbox, track.current_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
            
            if best_match_id:
                # Update existing track
                track = self.tracks[best_match_id]
                track.positions.append(detection.bbox)
                track.timestamps.append(timestamp)
                track.confidence = detection.confidence
                track.last_seen = timestamp
                matched_track_ids.add(best_match_id)
                unmatched_detections.remove(detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = str(uuid.uuid4())[:8]
            is_person = detection.label.lower() == 'person'
            
            track = TrackedObject(
                track_id=track_id,
                label=detection.label,
                confidence=detection.confidence,
                is_person=is_person,
                last_seen=timestamp
            )
            track.positions.append(detection.bbox)
            track.timestamps.append(timestamp)
            
            self.tracks[track_id] = track
        
        return self.tracks
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                       box2: Optional[Tuple[int, int, int, int]]) -> float:
        """Calculate Intersection over Union between two boxes."""
        if box2 is None:
            return 0.0
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_persons(self) -> List[TrackedObject]:
        """Get all tracked persons."""
        return [t for t in self.tracks.values() if t.is_person]
    
    def get_objects(self) -> List[TrackedObject]:
        """Get all tracked non-person objects."""
        return [t for t in self.tracks.values() if not t.is_person]


class ActionRecognizer:
    """
    Detects littering actions using spatial analysis and temporal patterns.
    
    Detection logic:
    1. Track persons and objects across frames
    2. Detect when object appears near person (association)
    3. Detect object falling/dropping (vertical displacement)
    4. Detect person moving away from stationary object
    5. Confirm event with multi-frame validation
    """
    
    # Litter object classes to track
    LITTER_CLASSES = {
        'bottle', 'cup', 'cell phone', 'handbag', 'backpack', 
        'umbrella', 'book', 'remote'  # Common droppable items
    }
    
    def __init__(
        self,
        proximity_threshold: int = 150,
        fall_threshold: int = 50,
        separation_threshold: int = 200,
        validation_frames: int = 5,
        ground_level_ratio: float = 0.8,
        cooldown_seconds: float = 3.0,
        frame_height: int = 720
    ):
        """
        Initialize action recognizer.
        
        Args:
            proximity_threshold: Max distance (px) for person-object association
            fall_threshold: Min Y displacement (px) to detect drop
            separation_threshold: Min distance (px) for person leaving object
            validation_frames: Frames to confirm stationary object
            ground_level_ratio: Y ratio below which is "ground level"
            cooldown_seconds: Min time between events for same object
            frame_height: Frame height for ground level calculation
        """
        self.proximity_threshold = proximity_threshold
        self.fall_threshold = fall_threshold
        self.separation_threshold = separation_threshold
        self.validation_frames = validation_frames
        self.ground_level_ratio = ground_level_ratio
        self.cooldown_seconds = cooldown_seconds
        self.frame_height = frame_height
        
        self.tracker = ObjectTracker(iou_threshold=0.3, max_age=2.0)
        self.pending_events: Dict[str, dict] = {}  # object_id -> event info
        self.confirmed_events: List[LitteringEvent] = []
        self.last_event_time: Dict[str, float] = {}  # object_id -> timestamp
        self.frame_count = 0
    
    def update(self, detections: List[Detection], timestamp: Optional[float] = None) -> List[LitteringEvent]:
        """
        Process new frame detections and check for littering events.
        
        Args:
            detections: List of detections from current frame
            timestamp: Current timestamp (uses time.time() if None)
            
        Returns:
            List of newly confirmed littering events
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        
        # Update object tracker
        self.tracker.update(detections, timestamp)
        
        # Get current persons and objects
        persons = self.tracker.get_persons()
        objects = self.tracker.get_objects()
        
        new_events = []
        
        # Check each tracked object
        for obj in objects:
            if obj.label.lower() not in self.LITTER_CLASSES:
                continue
            
            # Check cooldown
            if obj.track_id in self.last_event_time:
                if timestamp - self.last_event_time[obj.track_id] < self.cooldown_seconds:
                    continue
            
            # Find nearest person
            nearest_person, distance = self._find_nearest_person(obj, persons)
            
            # Check for littering pattern
            event = self._check_littering_pattern(
                obj, nearest_person, distance, timestamp
            )
            
            if event:
                new_events.append(event)
                self.confirmed_events.append(event)
                self.last_event_time[obj.track_id] = timestamp
                logger.warning(
                    f"🚨 LITTERING DETECTED: {event.object_label} "
                    f"(confidence: {event.confidence:.2f})"
                )
        
        return new_events
    
    def _find_nearest_person(
        self, 
        obj: TrackedObject, 
        persons: List[TrackedObject]
    ) -> Tuple[Optional[TrackedObject], float]:
        """Find the person nearest to an object."""
        if not persons or not obj.center:
            return None, float('inf')
        
        obj_center = obj.center
        nearest = None
        min_dist = float('inf')
        
        for person in persons:
            person_center = person.bottom_center
            if person_center is None:
                continue
            
            dist = np.sqrt(
                (obj_center[0] - person_center[0])**2 + 
                (obj_center[1] - person_center[1])**2
            )
            
            if dist < min_dist:
                min_dist = dist
                nearest = person
        
        return nearest, min_dist
    
    def _check_littering_pattern(
        self,
        obj: TrackedObject,
        nearest_person: Optional[TrackedObject],
        distance: float,
        timestamp: float
    ) -> Optional[LitteringEvent]:
        """
        Check if object trajectory matches littering pattern.
        
        Pattern:
        1. Object was near person (associated)
        2. Object fell (vertical displacement)
        3. Object now stationary
        4. Person moved away
        """
        obj_id = obj.track_id
        
        # Need enough position history
        if len(obj.positions) < self.validation_frames:
            return None
        
        # Check if object has fallen (significant downward movement)
        vertical_velocity = obj.get_vertical_velocity(frames=3)
        _, dy = obj.get_displacement(frames=5)
        
        has_fallen = dy > self.fall_threshold or vertical_velocity > 10
        
        # Check if object is near ground level
        if obj.current_bbox:
            obj_bottom = obj.current_bbox[3]
            at_ground = obj_bottom > (self.frame_height * self.ground_level_ratio)
        else:
            at_ground = False
        
        # Check if object is now stationary
        recent_displacement = obj.get_displacement(frames=3)
        is_stationary = abs(recent_displacement[0]) < 20 and abs(recent_displacement[1]) < 20
        
        # Check if person is now far from object
        person_left = distance > self.separation_threshold
        
        # Handle pending event (waiting for validation)
        if obj_id in self.pending_events:
            pending = self.pending_events[obj_id]
            pending['frames_stationary'] += 1 if is_stationary else 0
            
            # Confirm if stationary for enough frames and person left
            if pending['frames_stationary'] >= self.validation_frames and person_left:
                del self.pending_events[obj_id]
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    has_fallen=True,
                    is_stationary=is_stationary,
                    person_left=person_left,
                    at_ground=at_ground
                )
                
                return LitteringEvent(
                    event_id=str(uuid.uuid4())[:12],
                    timestamp=timestamp,
                    person_track_id=pending.get('person_id', 'unknown'),
                    object_track_id=obj_id,
                    object_label=obj.label,
                    confidence=confidence,
                    person_bbox=pending.get('person_bbox', (0, 0, 0, 0)),
                    object_bbox=obj.current_bbox or (0, 0, 0, 0),
                    frame_number=self.frame_count
                )
        
        # Check for new potential littering event
        elif has_fallen and (at_ground or is_stationary):
            # Object just fell - start tracking
            self.pending_events[obj_id] = {
                'start_time': timestamp,
                'person_id': nearest_person.track_id if nearest_person else None,
                'person_bbox': nearest_person.current_bbox if nearest_person else None,
                'frames_stationary': 1 if is_stationary else 0
            }
            logger.debug(f"Potential littering: {obj.label} fell, tracking...")
        
        return None
    
    def _calculate_confidence(
        self,
        has_fallen: bool,
        is_stationary: bool,
        person_left: bool,
        at_ground: bool
    ) -> float:
        """Calculate confidence score for littering event."""
        score = 0.0
        
        if has_fallen:
            score += 0.3
        if is_stationary:
            score += 0.25
        if person_left:
            score += 0.25
        if at_ground:
            score += 0.2
        
        return min(score, 1.0)
    
    def get_recent_events(self, seconds: float = 60.0) -> List[LitteringEvent]:
        """Get events from the last N seconds."""
        cutoff = time.time() - seconds
        return [e for e in self.confirmed_events if e.timestamp > cutoff]
    
    def get_all_events(self) -> List[LitteringEvent]:
        """Get all confirmed events."""
        return self.confirmed_events.copy()
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            'total_events': len(self.confirmed_events),
            'active_tracks': len(self.tracker.tracks),
            'persons_tracked': len(self.tracker.get_persons()),
            'objects_tracked': len(self.tracker.get_objects()),
            'pending_events': len(self.pending_events),
            'frames_processed': self.frame_count
        }
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.tracker.tracks.clear()
        self.pending_events.clear()
        self.confirmed_events.clear()
        self.last_event_time.clear()
        self.frame_count = 0


def draw_littering_alert(
    frame: np.ndarray,
    event: LitteringEvent,
    color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    """
    Draw littering alert on frame.
    
    Args:
        frame: Input image
        event: Littering event to highlight
        color: Alert color (BGR)
        
    Returns:
        Frame with alert drawn
    """
    import cv2
    
    # Draw object bounding box with thick red border
    x1, y1, x2, y2 = event.object_bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw alert label
    label = f"LITTERING: {event.object_label}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    return frame
