# Real-Time Gesture Recognition

A modular real-time computer vision system for hand landmark detection, gesture classification, and gesture-based automation using OpenCV and MediaPipe.

## Overview

This project captures live webcam input, detects hand landmarks in real time using MediaPipe, classifies gestures from landmark positions, and maps recognized gestures to Python-based automation actions. The application is organized as a scalable pipeline with separated modules for camera input, detection, classification, rendering, and action handling.

## Features

- Real-time webcam-based hand tracking
- Hand landmark detection with MediaPipe
- Gesture classification from landmark coordinates
- On-screen gesture labels
- FPS display for performance monitoring
- Modular architecture for maintainability and scalability
- Automation hooks for gesture-triggered actions

## Project Architecture

The system follows this pipeline:

Capture Frame -> Preprocess -> Detect Hand Landmarks -> Extract Features -> Classify Gesture -> Trigger Action -> Render Output

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI
- PyTest

## Project Structure

REAL-TIME-GESTURE-RECOGNITION/
│
├── .github/
│   └── workflows/
│       └── python-ci.yml
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── camera.py
│   ├── detector.py
│   ├── gestures.py
│   ├── automation.py
│   ├── utils.py
│   └── config.py
│
├── tests/
│   ├── __init__.py
│   └── test_gestures.py
│
├── .gitignore
├── LICENSE
├── main.py
├── README.md
└── requirements.txt

## Supported Gestures

Current example gestures include:

- Open Palm
- Fist
- Peace Sign
- Thumbs Up
- Pointing

These gesture labels can be expanded as the project grows.

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/real-time-gesture-recognition.git  
cd real-time-gesture-recognition

2. Create and activate a virtual environment:

Windows:
python -m venv venv  
venv\Scripts\activate  

macOS/Linux:
python3 -m venv venv  
source venv/bin/activate  

3. Install dependencies:

pip install -r requirements.txt

## How to Run

python main.py

Press `q` to quit the application window.

## Example Use Cases

- Gesture-controlled desktop actions
- Computer vision experimentation
- Human-computer interaction projects
- Real-time landmark-based classification systems

## Future Improvements

- Train a custom machine learning classifier on gesture data
- Add more gesture classes
- Support volume, brightness, and presentation controls
- Save gesture event logs for analysis
- Deploy as a desktop app or API-backed service

## Resume Bullet

Engineered a real-time computer vision pipeline using OpenCV and MediaPipe to detect hand landmarks and classify gestures, integrating Python-based automation triggers and modular components for scalability and maintainability.

## License

This project is licensed under the terms of the LICENSE file included in this repository.