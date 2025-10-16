"""
License Plate Registration System (LPRS) - Project Implementation
Course: Artificial Vision
Author: Miguel Garcia Diaz de Rivera - 178607
Date: 30/09/2025

This is a proof-of-concept implementation demonstrating the core pipeline:
- Image preprocessing (CLAHE, denoising)
- License plate detection (edge detection + contour analysis)
- OCR using Tesseract
- Privacy-preserving registry matching with SHA-256 hashing
- Accessibility features (automatic gate control simulation)

Note: Initial experiments with YOLO models (YOLOv5-nano, YOLOv8s) were attempted
but faced challenges with dataset availability and training time constraints.
This implementation uses classical CV techniques as a robust fallback.
"""

import cv2
import numpy as np
import pytesseract
from typing import Tuple, List, Dict, Optional
import hashlib
import json
from datetime import datetime


class LicensePlateRecognitionSystem:
    """
    Main LPRS pipeline for automatic license plate detection and recognition
    """

    def __init__(self):
        """Initialize the system with default parameters"""
        # Detection parameters
        self.min_plate_width = 80
        self.min_plate_height = 20
        self.max_plate_width = 400
        self.max_plate_height = 150

        # OCR parameters
        self.ocr_confidence_threshold = 60

        # Secure registry (hashed identifiers only)
        self.plate_registry = {}
        self.access_log = []

        print("LPRS System initialized")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocessing pipeline: Grayscale, CLAHE, and bilateral filtering
        Handles lighting variations and noise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter for noise reduction (preserves edges)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return denoised

    def detect_plates(self, preprocessed_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        License plate detection using edge detection and contour analysis

        Note: We initially attempted YOLO-based detection (YOLOv5-nano, YOLOv8s)
        but encountered challenges:
        - Limited availability of Mexican license plate datasets
        - Training time constraints for the project timeline
        - Hardware limitations for model training

        This classical approach serves as a robust fallback with reasonable accuracy
        in controlled conditions.
        """
        # Edge detection using Canny
        edges = cv2.Canny(preprocessed_image, 100, 200)

        # Morphological operations to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        plate_candidates = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if (self.min_plate_width <= w <= self.max_plate_width and
                self.min_plate_height <= h <= self.max_plate_height):

                # Check aspect ratio (plates are wider than tall)
                aspect_ratio = w / float(h)
                if 2.0 <= aspect_ratio <= 6.0:
                    plate_candidates.append((x, y, w, h))

        return plate_candidates

    def extract_and_rectify_plate(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract plate region and apply adaptive thresholding for better OCR
        """
        x, y, w, h = bbox
        plate_roi = image[y:y+h, x:x+w]

        # Adaptive thresholding for high contrast
        thresh = cv2.adaptiveThreshold(
            plate_roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        return thresh

    def perform_ocr(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Perform OCR using Tesseract with license plate configuration

        Note: We experimented with:
        - Fine-tuned Tesseract (current implementation)
        - CNN+CTC models (training challenges due to dataset limitations)
        - Transformer-based OCR (computational constraints)

        Tesseract with custom configuration provides acceptable results
        for our proof-of-concept.
        """
        # Configure Tesseract for license plates
        custom_config = r'--psm 7 -c tesseract_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

        # Perform OCR
        data = pytesseract.image_to_data(
            plate_image,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        # Extract text and confidence
        text_parts = []
        confidences = []

        for i, conf in enumerate(data['conf']):
            if int(conf) > 0:
                text_parts.append(data['text'][i])
                confidences.append(int(conf))

        recognized_text = ''.join(text_parts).strip()
        avg_confidence = np.mean(confidences) if confidences else 0

        return recognized_text, avg_confidence

    def hash_plate(self, plate_text: str) -> str:
        """
        Privacy-preserving hashing using SHA-256
        Ensures personal data protection
        """
        return hashlib.sha256(plate_text.encode()).hexdigest()

    def register_plate(self, plate_text: str, user_info: Dict):
        """
        Register a plate in the secure whitelist
        Supports accessibility features and access control levels
        """
        plate_hash = self.hash_plate(plate_text)
        self.plate_registry[plate_hash] = {
            'registered_date': datetime.now().isoformat(),
            'access_level': user_info.get('access_level', 'standard'),
            'is_accessible': user_info.get('is_accessible', False),
            'name': user_info.get('name', 'Unknown')
        }
        print(f"Registered: {plate_text} (Level: {user_info.get('access_level', 'standard')})")

    def check_access(self, plate_text: str) -> Optional[Dict]:
        """
        Match recognized plate against registry
        Returns user info if authorized, None otherwise
        """
        plate_hash = self.hash_plate(plate_text)
        return self.plate_registry.get(plate_hash)

    def log_access_attempt(self, plate_text: str, authorized: bool, confidence: float):
        """
        Log all access attempts with timestamp for security audit
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'plate_hash': self.hash_plate(plate_text),
            'authorized': authorized,
            'confidence': confidence
        }
        self.access_log.append(log_entry)

    def process_image(self, image: np.ndarray) -> Dict:
        """
        Complete pipeline: Detect -> Recognize -> Match -> Log
        """
        results = {
            'plates_detected': 0,
            'recognized_plates': [],
            'access_granted': []
        }

        # Step 1: Preprocessing
        preprocessed = self.preprocess_image(image)

        # Step 2: Detection
        plate_regions = self.detect_plates(preprocessed)
        results['plates_detected'] = len(plate_regions)

        # Step 3: Recognition and Matching
        for bbox in plate_regions:
            x, y, w, h = bbox

            # Extract and rectify plate
            plate_img = self.extract_and_rectify_plate(preprocessed, bbox)

            # Perform OCR
            plate_text, confidence = self.perform_ocr(plate_img)

            # Filter by confidence and minimum length
            if confidence >= self.ocr_confidence_threshold and len(plate_text) >= 5:
                results['recognized_plates'].append({
                    'text': plate_text,
                    'confidence': confidence,
                    'bbox': bbox
                })

                # Check registry
                user_info = self.check_access(plate_text)

                if user_info:
                    results['access_granted'].append({
                        'plate': plate_text,
                        'user': user_info['name'],
                        'access_level': user_info['access_level'],
                        'is_accessible': user_info['is_accessible']
                    })

                    # Simulate accessibility feature
                    if user_info['is_accessible']:
                        print(f"ACCESSIBILITY: Auto-opening gate for {user_info['name']}")

                # Log attempt
                self.log_access_attempt(plate_text, user_info is not None, confidence)

                # Visualize on image
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{plate_text} ({confidence:.1f}%)"
                cv2.putText(image, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return results, image

    def export_logs(self, filename: str = 'access_logs.json'):
        """Export access logs for analysis"""
        with open(filename, 'w') as f:
            json.dump(self.access_log, f, indent=2)
        print(f"Logs exported to {filename}")


def create_synthetic_test_image(plate_text: str, filename: str):
    """
    Generate a synthetic test image with a license plate
    Useful for testing when real images are not available
    """
    # Create base image (simulating a car/background)
    img = np.random.randint(80, 120, (400, 600, 3), dtype=np.uint8)

    # Add noise for realism
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Draw license plate
    plate_x, plate_y = 150, 160
    plate_w, plate_h = 300, 80

    # White background
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h),
                 (255, 255, 255), -1)

    # Black border
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h),
                 (0, 0, 0), 3)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(plate_text, font, 2, 3)[0]
    text_x = plate_x + (plate_w - text_size[0]) // 2
    text_y = plate_y + (plate_h + text_size[1]) // 2
    cv2.putText(img, plate_text, (text_x, text_y), font, 2, (0, 0, 0), 3)

    cv2.imwrite(filename, img)
    print(f"Generated: {filename}")
    return filename


def run_demo():
    """
    Demonstration of the LPRS system functionality
    """
    print("LICENSE PLATE REGISTRATION SYSTEM - PROOF OF CONCEPT")
    print("\nProject: Artificial Vision - LPRS Implementation")
    print("Author: Miguel Garcia Diaz de Rivera - 178607\n")

    # Initialize system
    lprs = LicensePlateRecognitionSystem()
    print()

    # Register test plates (simulating UDLAP campus scenario)
    print("Registering test plates in system...")

    lprs.register_plate("ABC1234", {
        'name': 'Student with Disability Permit',
        'access_level': 'premium',
        'is_accessible': True
    })

    lprs.register_plate("XYZ5678", {
        'name': 'Faculty Member',
        'access_level': 'standard',
        'is_accessible': False
    })

    lprs.register_plate("MEX9876", {
        'name': 'Campus Security',
        'access_level': 'admin',
        'is_accessible': False
    })

    print()

    # Generate test images
    print("Generating synthetic test images...")

    test_cases = [
        ("ABC1234", "test_registered_accessible.jpg"),
        ("XYZ5678", "test_registered_standard.jpg"),
        ("UNK1111", "test_unregistered.jpg"),
    ]

    for plate_text, filename in test_cases:
        create_synthetic_test_image(plate_text, filename)

    print()

    # Process test images
    print("Processing test images...")

    for plate_text, filename in test_cases:
        print(f"\nTest: {filename} (Expected: {plate_text})")

        image = cv2.imread(filename)
        results, annotated = lprs.process_image(image)

        print(f"  Detected: {results['plates_detected']} plate(s)")

        for plate in results['recognized_plates']:
            match = "" if plate['text'] == plate_text else ""
            print(f"   Recognized: {plate['text']} (confidence: {plate['confidence']:.1f}%)")

        if results['access_granted']:
            for access in results['access_granted']:
                print(f"   ACCESS GRANTED: {access['user']} [{access['access_level']}]")
        else:
            print(f"   ACCESS DENIED: Plate not in registry")

        # Save result
        output_filename = filename.replace('.jpg', '_result.jpg')
        cv2.imwrite(output_filename, annotated)
        print(f"   Saved result: {output_filename}")

    print()

    # Export logs
    lprs.export_logs('demo_access_logs.json')

    print("\nDEMO COMPLETED")
    print("\nGenerated files:")
    print("  - test_*.jpg (original synthetic images)")
    print("  - test_*_result.jpg (annotated detection results)")
    print("  - demo_access_logs.json (access attempt logs)")
    print("\nChallenges encountered:")
    print("   YOLO training: Limited dataset availability")
    print("   Deep learning OCR: Computational constraints")
    print("   Classical CV approach: Functional proof-of-concept")


if __name__ == "__main__":
    run_demo()
