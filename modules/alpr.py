"""
CivicCam ALPR Module

Automatic License Plate Recognition using EasyOCR.
Detects and reads vehicle license plates from detected vehicles.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None


logger = logging.getLogger("civiccam.alpr")


# Indian license plate patterns
INDIAN_PLATE_PATTERNS = [
    # Standard: KL-01-AB-1234 or KL01AB1234
    r'^[A-Z]{2}[-\s]?\d{2}[-\s]?[A-Z]{1,3}[-\s]?\d{4}$',
    # Bharat Series: 22BH1234AB
    r'^\d{2}[-\s]?BH[-\s]?\d{4}[-\s]?[A-Z]{2}$',
    # Commercial: KA-51-C-1234
    r'^[A-Z]{2}[-\s]?\d{2}[-\s]?[A-Z][-\s]?\d{4}$',
    # Simplified pattern (more lenient)
    r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$',
]


@dataclass
class PlateResult:
    """
    Result from license plate reading.
    
    Attributes:
        text: Detected plate text (cleaned)
        raw_text: Original OCR output
        confidence: Detection confidence (0-1)
        bbox: Plate bounding box in original image
        is_valid: Whether plate matches expected format
    """
    text: str
    raw_text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    is_valid: bool = False
    
    def to_dict(self) -> dict:
        return {
            'text': self.text,
            'raw_text': self.raw_text,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'is_valid': self.is_valid
        }


class LicensePlateReader:
    """
    License plate detection and OCR reader using EasyOCR.
    
    Example:
        >>> reader = LicensePlateReader()
        >>> result = reader.read_plate(vehicle_image)
        >>> if result:
        ...     print(f"Plate: {result.text}")
    """
    
    def __init__(
        self,
        languages: List[str] = ['en'],
        confidence_threshold: float = 0.5,
        plate_region_ratio: float = 0.35,
        gpu: bool = True
    ):
        """
        Initialize the plate reader.
        
        Args:
            languages: OCR languages (default: English)
            confidence_threshold: Minimum confidence for valid read
            plate_region_ratio: Portion of vehicle bottom to search for plate
            gpu: Use GPU acceleration if available
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR not installed. Install with: pip install easyocr"
            )
        
        self.confidence_threshold = confidence_threshold
        self.plate_region_ratio = plate_region_ratio
        
        logger.info("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        logger.info("EasyOCR reader initialized")
        
        # Compile regex patterns
        self.patterns = [re.compile(p) for p in INDIAN_PLATE_PATTERNS]
    
    def read_plate(
        self, 
        image: np.ndarray,
        vehicle_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[PlateResult]:
        """
        Read license plate from an image.
        
        Args:
            image: Input image (BGR format)
            vehicle_bbox: Optional vehicle bounding box (x1, y1, x2, y2)
            
        Returns:
            PlateResult if plate found, None otherwise
        """
        # Extract plate region
        if vehicle_bbox:
            plate_region = self._extract_plate_region(image, vehicle_bbox)
        else:
            plate_region = image
        
        if plate_region is None or plate_region.size == 0:
            return None
        
        # Preprocess
        processed = self._preprocess(plate_region)
        
        # Run OCR
        try:
            results = self.reader.readtext(processed)
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return None
        
        if not results:
            return None
        
        # Find best plate-like result
        best_result = self._find_best_plate(results)
        
        if best_result:
            return best_result
        
        return None
    
    def read_plates_batch(
        self,
        images: List[np.ndarray],
        vehicle_bboxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> List[Optional[PlateResult]]:
        """Read plates from multiple images."""
        results = []
        for i, img in enumerate(images):
            bbox = vehicle_bboxes[i] if vehicle_bboxes else None
            results.append(self.read_plate(img, bbox))
        return results
    
    def _extract_plate_region(
        self,
        image: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Extract the region where license plate is likely located.
        Typically the bottom portion of the vehicle.
        """
        x1, y1, x2, y2 = vehicle_bbox
        height = y2 - y1
        width = x2 - x1
        
        # Plate is usually in bottom 35% of vehicle
        plate_y1 = y1 + int(height * (1 - self.plate_region_ratio))
        plate_y2 = y2
        
        # Ensure bounds are valid
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, plate_y1)
        x2 = min(w, x2)
        y2 = min(h, plate_y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2]
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize if too small
        h, w = gray.shape
        if w < 200:
            scale = 200 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply adaptive thresholding for better contrast
        # gray = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        # )
        
        # Denoise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        return gray
    
    def _find_best_plate(self, ocr_results: List) -> Optional[PlateResult]:
        """
        Find the best plate-like text from OCR results.
        """
        candidates = []
        
        for (bbox, text, confidence) in ocr_results:
            # Clean the text
            cleaned = self._clean_plate_text(text)
            
            if len(cleaned) < 4:
                continue
            
            # Check if valid format
            is_valid = self._validate_plate(cleaned)
            
            # Score based on validity and confidence
            score = confidence
            if is_valid:
                score += 0.5
            
            candidates.append({
                'text': cleaned,
                'raw_text': text,
                'confidence': confidence,
                'score': score,
                'is_valid': is_valid,
                'bbox': bbox
            })
        
        if not candidates:
            return None
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best = candidates[0]
        
        # Only return if meets threshold
        if best['confidence'] < self.confidence_threshold:
            return None
        
        return PlateResult(
            text=best['text'],
            raw_text=best['raw_text'],
            confidence=best['confidence'],
            is_valid=best['is_valid']
        )
    
    def _clean_plate_text(self, text: str) -> str:
        """
        Clean and normalize plate text.
        """
        # Remove spaces and special characters except hyphen
        cleaned = re.sub(r'[^A-Za-z0-9-]', '', text)
        
        # Convert to uppercase
        cleaned = cleaned.upper()
        
        # Common OCR corrections
        corrections = {
            'O': '0',  # O -> 0 in number positions
            'I': '1',  # I -> 1
            'Z': '2',
            'S': '5',
            'B': '8',
        }
        
        # Apply corrections only to middle portion (numbers)
        if len(cleaned) >= 6:
            # Keep first 2 chars as letters
            prefix = cleaned[:2]
            rest = cleaned[2:]
            
            # Fix numbers in middle
            for old, new in corrections.items():
                if rest[:2].isalpha():
                    continue
                rest = rest[:2].replace(old, new) + rest[2:]
        
            cleaned = prefix + rest
        
        return cleaned
    
    def _validate_plate(self, text: str) -> bool:
        """
        Validate plate text against known Indian formats.
        """
        for pattern in self.patterns:
            # Remove hyphens for matching
            normalized = text.replace('-', '').replace(' ', '')
            if pattern.match(normalized) or pattern.match(text):
                return True
        return False


def draw_plate_info(
    frame: np.ndarray,
    plate_result: PlateResult,
    vehicle_bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 255)
) -> np.ndarray:
    """
    Draw plate information on frame.
    """
    x1, y1, x2, y2 = vehicle_bbox
    
    # Draw plate text below vehicle
    label = f"Plate: {plate_result.text}"
    if plate_result.is_valid:
        label += " ✓"
    
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    cv2.rectangle(frame, (x1, y2), (x1 + tw + 10, y2 + th + 10), color, -1)
    cv2.putText(
        frame, label, (x1 + 5, y2 + th + 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )
    
    return frame
