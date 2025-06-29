🤟 SilentEcho - AI-Powered Sign Language Translator

SilentEcho is an AI-driven application designed to enhance communication for the deaf and mute community by providing real-time text-to-sign language translation, with experimental gesture detection support.
✨ Features

    Real-Time Text-to-Sign Translation: Uses IBM Granite models to refine text and map it to pre-recorded sign language videos.
    AI Text Refinement: Powered by IBM Granite 3.3 8B Instruct and Granite 4.0 Tiny Preview for low-latency processing.
    Gesture Detection (Experimental): Integrates MediaPipe for basic webcam-based hand gesture recognition.
    Offline Support: Operates with local video files, suitable for low-connectivity regions.
    User Feedback: Provides status updates on processed and missing signs.

🚀 Quick Start
Prerequisites

    Python 3.8+ installed
    Webcam (optional for gesture detection)
    Local video files in the videos/ folder

Installation

    Clone or download the project files
    Install dependencies:
    bash

pip install -r requirements.txt

    requirements.txt should include: gradio, transformers, torch, mediapipe, opencv-python

Download Granite Models:

    Granite 3.3 8B Instruct: from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained("ibm/granite-3.3-8b-instruct")
    Granite 4.0 Tiny Preview: from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained("ibm-granite/granite-4.0-tiny-preview")

Run the application:
bash

    python silent_echo.py

🎮 Controls

    No manual controls: Interact via the Gradio web interface at http://localhost:7860.

🔧 Troubleshooting
Common Issues
🤖 Model Loading Issues

    Symptoms: "Model not found" or loading errors
    Solutions:
        Verify internet connection for initial model download.
        Check model name in code (e.g., "ibm/granite-3.3-8b-instruct").
        Switch to Granite 4.0 Tiny Preview if memory-constrained.

📹 Webcam Issues

    Symptoms: "Could not open webcam"
    Solutions:
        Check camera permissions in system settings.
        Ensure no other app uses the camera.
        Test with python -c "import cv2; print(cv2.VideoCapture(0).isOpened())".

⚡ Performance Issues

    Symptoms: Slow response or lag
    Solutions:
        Close unnecessary applications.
        Use Granite 4.0 Tiny Preview for lighter processing.
        Reduce input text length.

Diagnostic Tools

    Status Feedback: View processed/missing signs in the interface.

Error Logging

    Logs are printed to the console; no error_log.json implemented.

📊 Performance Optimization

    Model Selection: Switch to Granite 4.0 Tiny Preview for better performance on limited hardware.
    Input Limit: Restrict input to 4 words to maintain responsiveness.
