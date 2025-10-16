## License Plate Recognition System (LPRS)
Project Implementation

> Course: Artificial Vision
> Author: Miguel Garcia Diaz de Rivera - 178607
> Date: 30/09/2025

### Project Code

The complete implementation is available in the attached Python file: vision1.py

### Development Environment

The project was developed using a virtual environment created with python3 -m venv lprs_env. The system requires several Python packages including opencv-python for computer vision operations, numpy for numerical computations, and pytesseract for OCR functionality. These dependencies were installed via pip to ensure proper environment isolation and package management.

To use the system, first activate the environment with source lprs_env/bin/activate. Then run your Python scripts normally using python your_script.py. When you are finished working with the project, deactivate the virtual environment using the deactivate command.

### Implementation Status

Yes, I implemented the proposed License Plate Registration System with the core pipeline including image preprocessing using CLAHE and bilateral filtering for noise reduction, license plate detection using edge detection and contour analysis, OCR integration with Tesseract for text recognition, privacy-preserving registry using SHA-256 hashing, access control system with user levels and accessibility features, and comprehensive logging for security audit trails.
Experimental Results and Robustness

The system successfully demonstrates license plate detection in *controlled* conditions, text recognition with confidence scoring, secure registry matching via hashing, access control with different user levels, and accessibility features including automatic gate control simulation.

Several challenges were encountered during implementation. The system shows limited robustness with real-world variations in lighting and angles. OCR accuracy varies significantly with image quality. An initial YOLO approach was attempted but faced dataset limitations, so the classical computer vision approach served as a reliable fallback for proof-of-concept.

While the system works well in controlled environments with synthetic test images, real-world robustness requires further improvements. These include larger and more diverse training datasets, deep learning-based detection models, more sophisticated image preprocessing, and better handling of various lighting conditions and plate orientations.
How to Run

Execute the system using python vision1.py. The demo will initialize the LPRS system, register test license plates, generate synthetic test images, process images and display results, and export access logs.
### Output Files Generated

The system produces test images as synthetic test images, annotated detection results as result images, and access attempt logs in JSON format for analysis and review.

This implementation serves as a functional proof-of-concept demonstrating the core computer vision pipeline for license plate recognition systems.

