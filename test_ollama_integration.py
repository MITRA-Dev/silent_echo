import os
import sys
import requests
import json

# Test Ollama Integration with Granite 3.3:8b
print("\n--- Ollama Granite 3.3:8b Model Test ---")

def test_ollama_granite():
    """Test the local Ollama Granite 3.3:8b model"""
    try:
        # Test if Ollama is running
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("Ollama is not running. Please start Ollama and ensure Granite 3.3:8b is available.")
            return False
        
        # Test model generation
        payload = {
            "model": "granite3.3:8b",
            "prompt": "Hello, how can I help you today?",
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("Model response:", result.get('response', 'No response'))
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
        return False
    except Exception as e:
        print(f"Error testing Ollama: {e}")
        return False

# Test Microphone
print("\n--- Microphone Test ---")
try:
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something (testing microphone)...")
        audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_google(audio)
            print("You said:", text)
            
            # Test sending speech to Ollama
            if text:
                print("\n--- Testing Speech-to-Text with Ollama ---")
                payload = {
                    "model": "granite3.3:8b",
                    "prompt": f"User said: {text}. Please respond naturally.",
                    "stream": False
                }
                
                response = requests.post("http://localhost:11434/api/generate", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    print("Ollama response to speech:", result.get('response', 'No response'))
                else:
                    print("Failed to get response from Ollama")
                    
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
except Exception as e:
    print("Microphone test failed:", e)

# Test Webcam
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

# Run the main Ollama test
if test_ollama_granite():
    print("\n✅ Ollama Granite 3.3:8b is working!")
else:
    print("\n❌ Ollama test failed. Please check your Ollama installation.") 