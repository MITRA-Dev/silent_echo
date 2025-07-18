# Simple Silent Echo - No Browser Permissions Required

import streamlit as st
import speech_recognition as sr
import cv2
import numpy as np
import time
import os
import requests
import json
import pandas as pd

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "granite3.3:8b"

class SilentEchoSimple:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.cap = None
        self.sign_language_detected = False
        
        # Load the sign language dataset
        self.sign_dataset = self.load_sign_dataset()
        
        # Real-time gesture tracking
        self.last_gesture = None
        self.last_response_time = time.time()
        
        # Adjust for ambient noise (only if microphone is available)
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            st.warning(f"Could not adjust for ambient noise: {e}")
    
    def load_sign_dataset(self):
        """Load the GPT-3.5 cleaned sign language dataset"""
        try:
            df = pd.read_csv('gpt-3.5-cleaned.csv')
            # Filter out empty annotated_texts
            df = df[df['annotated_texts'].notna() & (df['annotated_texts'] != '')]
            return df
        except Exception as e:
            st.error(f"Error loading sign dataset: {e}")
            return pd.DataFrame()
    
    def get_sign_gestures(self):
        """Get unique sign gestures from the dataset"""
        if not self.sign_dataset.empty:
            return self.sign_dataset['annotated_texts'].unique().tolist()
        return []
    
    def start_webcam(self):
        """Initialize webcam for sign language detection with better error handling"""
        try:
            # Release any existing camera first
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            
            # Try Media Foundation first (most stable on Windows)
            self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
            if not self.cap.isOpened():
                # Try DirectShow as fallback
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    # Try default backend
                    self.cap = cv2.VideoCapture(0)
                    if not self.cap.isOpened():
                        st.error("Cannot open webcam with any backend")
                        return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test frame reading
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                st.error("Cannot read frames from webcam")
                self.cap.release()
                self.cap = None
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing webcam: {e}")
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
            return False
    
    def stop_webcam(self):
        """Stop webcam"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except:
            pass
    
    def detect_sign_language(self, frame):
        """Simple sign language detection"""
        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Simple hand detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_gesture = None
        
        # Check for hand-like contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Adjust threshold as needed
                self.sign_language_detected = True
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Simple gesture detection
                detected_gesture = self.analyze_gesture_shape(contour)
                break
        
        return frame, detected_gesture
    
    def analyze_gesture_shape(self, contour):
        """Simple gesture analysis"""
        try:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = cv2.contourArea(contour)
            
            # Simple gesture classification
            if area > 15000:
                return "Large Hand Gesture"
            elif area > 8000:
                return "Medium Hand Gesture"
            elif aspect_ratio > 2.0:
                return "Pointing"
            elif aspect_ratio < 0.5:
                return "Raised Hand"
            else:
                return "Hand Gesture"
                
        except Exception as e:
            return "Hand Detected"
    
    def get_ollama_response(self, text):
        """Get response from Ollama"""
        try:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            data = {
                "model": MODEL_NAME,
                "prompt": text,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 150
                }
            }
            
            response = requests.post(url, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "I'm having trouble connecting to my AI brain right now. Please try again."
        except requests.exceptions.ConnectionError:
            return "I can't connect to my AI brain. Please make sure Ollama is running."
        except Exception as e:
            return f"I'm sorry, there was an error: {e}"
    
    def check_ollama_status(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

def main():
    st.set_page_config(
        page_title="Silent Echo - Simple Version",
        page_icon="🤟",
        layout="wide"
    )
    
    st.title("🤟 Silent Echo - Simple Version")
    st.markdown("### AI-Powered Communication Assistant (No Audio)")
    
    # Initialize the app
    if 'silent_echo' not in st.session_state:
        try:
            st.session_state.silent_echo = SilentEchoSimple()
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Ollama status
    ollama_status = st.session_state.silent_echo.check_ollama_status()
    if ollama_status:
        st.sidebar.success("🤖 Ollama: Connected")
    else:
        st.sidebar.error("🤖 Ollama: Not Connected")
    
    # Dataset info
    if not st.session_state.silent_echo.sign_dataset.empty:
        total_gestures = len(st.session_state.silent_echo.sign_dataset)
        st.sidebar.info(f"📊 Dataset: {total_gestures} examples")
    
    # Main interface
    st.header("🤟 Sign Language Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Hand Gesture Recognition")
        
        # Camera controls
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("📹 Start Camera", type="primary"):
                st.session_state.camera_active = True
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Camera"):
                st.session_state.camera_active = False
                st.rerun()
        
        # Live camera feed
        if st.session_state.camera_active:
            try:
                # Initialize webcam if not already done
                if not st.session_state.silent_echo.cap or not st.session_state.silent_echo.cap.isOpened():
                    if not st.session_state.silent_echo.start_webcam():
                        st.error("❌ Failed to initialize webcam")
                        st.session_state.camera_active = False
                        st.rerun()
                
                # Capture frame
                if st.session_state.silent_echo.cap and st.session_state.silent_echo.cap.isOpened():
                    ret, frame = st.session_state.silent_echo.cap.read()
                    
                    if ret and frame is not None and frame.size > 0:
                        # Process frame
                        processed_frame, detected_gesture = st.session_state.silent_echo.detect_sign_language(frame)
                        
                        # Convert BGR to RGB for display
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display the frame
                        st.image(rgb_frame, channels="RGB", use_container_width=True, caption="Live Video Feed")
                        
                        # Show detected gesture
                        if st.session_state.silent_echo.sign_language_detected and detected_gesture:
                            st.success(f"🤟 **Detected:** {detected_gesture}")
                            
                            # Get AI response
                            if st.button("🤖 Get AI Response"):
                                prompt = f"You are a helpful AI assistant. The user made this gesture: '{detected_gesture}'. Please provide a helpful response."
                                response = st.session_state.silent_echo.get_ollama_response(prompt)
                                st.info(f"🤖 **AI Response:** {response}")
                        else:
                            st.info("👋 Show your hands to the camera")
                    else:
                        st.error("Failed to capture frame")
                else:
                    st.error("Webcam not available")
                    st.session_state.camera_active = False
                    st.rerun()
                        
            except Exception as e:
                st.error(f"❌ Camera error: {e}")
                st.session_state.camera_active = False
                st.rerun()
        
        else:
            st.info("📹 Click 'Start Camera' to begin")
    
    with col2:
        st.subheader("Supported Gestures")
        
        if not st.session_state.silent_echo.sign_dataset.empty:
            available_gestures = st.session_state.silent_echo.get_sign_gestures()
            if available_gestures:
                for gesture in available_gestures[:5]:
                    st.write(f"• {gesture}")
                if len(available_gestures) > 5:
                    st.write(f"... and {len(available_gestures) - 5} more")
        else:
            st.write("Dataset not loaded")
    
    # Footer
    st.markdown("---")
    st.markdown("**Simple version without audio playback to avoid pygame issues**")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
    finally:
        # Cleanup
        if 'silent_echo' in st.session_state:
            st.session_state.silent_echo.cleanup()
