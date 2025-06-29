import streamlit as st
import speech_recognition as sr
import cv2
import numpy as np
import threading
import time
import queue
import os
from PIL import Image
import requests
import json
from gtts import gTTS
import tempfile
import pygame
import io

# IBM Granite API configuration
API_TOKEN = os.getenv("WATSONX_API_TOKEN")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
HF_ENDPOINT = "https://ibm-watsonx-ai.huggingface.co"

class SilentEcho:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.cap = None
        self.sign_language_detected = False
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def start_webcam(self):
        """Initialize webcam for sign language detection"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("Cannot open webcam")
            return False
        return True
    
    def stop_webcam(self):
        """Stop webcam"""
        if self.cap:
            self.cap.release()
    
    def detect_sign_language(self, frame):
        """Basic sign language detection using hand landmarks"""
        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for hand-like contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Adjust threshold as needed
                self.sign_language_detected = True
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                break
        
        return frame
    
    def listen_for_speech(self):
        """Listen for speech input"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                return audio
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            st.error(f"Speech recognition error: {e}")
            return None
    
    def speech_to_text(self, audio):
        """Convert speech to text"""
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            st.error(f"Speech recognition service error: {e}")
            return None
    
    def get_granite_response(self, text):
        """Get response from IBM Granite model"""
        try:
            # Using IBM watsonx.ai API
            headers = {
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model_id": "ibm/granite-3.3-8b-instruct",
                "input": text,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{HF_ENDPOINT}/v1/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["text"].strip()
            else:
                st.error(f"API Error: {response.status_code}")
                return "I'm sorry, I couldn't process that request."
                
        except Exception as e:
            st.error(f"Error getting Granite response: {e}")
            return "I'm sorry, there was an error processing your request."
    
    def text_to_speech(self, text):
        """Convert text to speech and play it"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts.save(fp.name)
                temp_file = fp.name
            
            # Play audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Clean up
            os.unlink(temp_file)
            
        except Exception as e:
            st.error(f"Text-to-speech error: {e}")
    
    def run_communication_loop(self):
        """Main communication loop"""
        if not self.start_webcam():
            return
        
        try:
            while self.is_listening:
                # Capture webcam frame
                ret, frame = self.cap.read()
                if ret:
                    # Detect sign language
                    processed_frame = self.detect_sign_language(frame)
                    
                    # Display frame
                    st.image(processed_frame, channels="BGR", use_container_width=True)
                    
                    # Listen for speech
                    audio = self.listen_for_speech()
                    if audio:
                        text = self.speech_to_text(audio)
                        if text:
                            st.success(f"🎤 You said: {text}")
                            
                            # Get AI response
                            response = self.get_granite_response(text)
                            st.info(f"🤖 AI Response: {response}")
                            
                            # Convert to speech
                            self.text_to_speech(response)
                
                time.sleep(0.1)
                
        finally:
            self.stop_webcam()

def main():
    st.set_page_config(
        page_title="Silent Echo - AI Communication Assistant",
        page_icon="🤟",
        layout="wide"
    )
    
    st.title("🤟 Silent Echo - AI Communication Assistant")
    st.markdown("### Empowering communication for the deaf community with AI")
    
    # Initialize the app
    if 'silent_echo' not in st.session_state:
        st.session_state.silent_echo = SilentEcho()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Controls")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Communication Mode",
        ["Speech-to-Text", "Sign Language Detection", "Text Input"]
    )
    
    # Main interface
    if mode == "Speech-to-Text":
        st.header("🎤 Real-time Speech Recognition")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎙️ Start Listening", type="primary"):
                st.session_state.listening = True
                st.info("Listening... Speak now!")
                
                # Simple speech recognition
                audio = st.session_state.silent_echo.listen_for_speech()
                if audio:
                    text = st.session_state.silent_echo.speech_to_text(audio)
                    if text:
                        st.success(f"🎤 You said: {text}")
                        
                        # Get AI response
                        response = st.session_state.silent_echo.get_granite_response(text)
                        st.info(f"🤖 AI Response: {response}")
                        
                        # Play response
                        if st.button("🔊 Play Response"):
                            st.session_state.silent_echo.text_to_speech(response)
        
        with col2:
            st.subheader("💡 Tips")
            st.markdown("""
            - Speak clearly and at a normal pace
            - Ensure good microphone quality
            - Reduce background noise
            - Wait for the "Listening..." message
            """)
    
    elif mode == "Sign Language Detection":
        st.header("🤟 Sign Language Detection")
        
        if st.button("📹 Start Camera"):
            st.session_state.silent_echo.start_webcam()
            
            # Display webcam feed
            placeholder = st.empty()
            
            while True:
                ret, frame = st.session_state.silent_echo.cap.read()
                if ret:
                    processed_frame = st.session_state.silent_echo.detect_sign_language(frame)
                    placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                    
                    if st.session_state.silent_echo.sign_language_detected:
                        st.success("🤟 Sign language detected!")
                
                if st.button("⏹️ Stop Camera"):
                    st.session_state.silent_echo.stop_webcam()
                    break
    
    elif mode == "Text Input":
        st.header("⌨️ Text Communication")
        
        user_input = st.text_area("Type your message:", height=100)
        
        if st.button("🤖 Get AI Response"):
            if user_input:
                response = st.session_state.silent_echo.get_granite_response(user_input)
                st.info(f"🤖 AI Response: {response}")
                
                if st.button("🔊 Play Response"):
                    st.session_state.silent_echo.text_to_speech(response)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About Silent Echo
    Silent Echo is an AI-powered communication assistant designed to help deaf individuals 
    communicate more effectively. It uses IBM Granite models for intelligent responses and 
    provides multiple communication modes including speech recognition and sign language detection.
    
    **Features:**
    - 🎤 Real-time speech-to-text conversion
    - 🤟 Basic sign language detection
    - 🤖 AI-powered contextual responses
    - 🔊 Text-to-speech output
    - 📱 Accessible web interface
    """)

if __name__ == "__main__":
    main()