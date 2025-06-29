import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
except ImportError:
    print("python-dotenv is not installed. Please install it with 'pip install python-dotenv'.")
    sys.exit(1)

if not Path('.env').exists():
    print(".env file not found! Please create a .env file with the following content:")
    print("WATSONX_API_TOKEN=your_token_here\nWATSONX_PROJECT_ID=your_project_id_here")
    sys.exit(1)

load_dotenv()
API_TOKEN = os.getenv("WATSONX_API_TOKEN")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

if not API_TOKEN or not PROJECT_ID:
    print("API token or Project ID not found in .env file. Please check your .env file.")
    sys.exit(1)

   

# 1. Test Granite 3.3 8B Instruct Model Loading
print("\n--- Granite 3.3 8B Instruct Model Test ---")
try:
    import torch
except ImportError:
    print("PyTorch is not installed. Please install it with 'pip install torch'.")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "ibm/granite-3.3-8b-instruct"
    # Set up environment variables for Hugging Face/IBM access
    os.environ["HF_ENDPOINT"] = "https://ibm-watsonx-ai.huggingface.co"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=API_TOKEN)
    prompt = "Hello, how can I help you today?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    print("Model output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
except Exception as e:
    print("Granite model test failed:", e)
    print("If you see an 'Unauthorized' or 'Invalid credentials' error, please double-check your API token and project permissions on IBM watsonx.ai.")

# 2. Test Microphone
print("\n--- Microphone Test ---")
try:
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something (testing microphone)...")
        audio = r.listen(source, timeout=5)
        try:
            print("You said:", r.recognize_google(audio))
        except Exception as e:
            print("Speech recognition error:", e)
except Exception as e:
    print("Microphone test failed:", e)
    print("If you see a 'PyAudio' error, install it with 'pip install pipwin' then 'pipwin install pyaudio' on Windows.")

# 3. Test Webcam
print("\n--- Webcam Test ---")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
    else:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("webcam_test.jpg", frame)
            print("Webcam test image saved as webcam_test.jpg")
        else:
            print("Failed to capture image from webcam.")
        cap.release()
except Exception as e:
    print("Webcam test failed:", e) 