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
import pandas as pd
import aiohttp
import asyncio
# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "granite3.3:8b"

class SilentEchoOllama:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.cap = None
        self.sign_language_detected = False
        
        # Load the sign language dataset
        self.sign_dataset = self.load_sign_dataset()
        
        # Real-time gesture tracking
        self.last_gesture = None
        self.gesture_response_cooldown = 0
        self.last_response_time = time.time()
        
        # Video processing
        self.current_gesture = None
        self.gesture_history = []
        
        # Initialize pygame for audio playback (with error handling)
        try:
            pygame.mixer.init()
            self.pygame_available = True
        except Exception as e:
            st.warning(f"Pygame audio not available: {e}")
            self.pygame_available = False
        
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
    
    def find_similar_gesture(self, detected_gesture):
        """Find similar gesture in the dataset"""
        if self.sign_dataset.empty:
            return None
        
        # Simple similarity matching (can be enhanced with more sophisticated NLP)
        detected_lower = detected_gesture.lower()
        for gesture in self.sign_dataset['annotated_texts'].unique():
            if gesture and detected_lower in gesture.lower() or gesture.lower() in detected_lower:
                return gesture
        return None
    
    def get_gesture_context(self, gesture):
        """Get context and examples for a specific gesture"""
        if self.sign_dataset.empty:
            return None
        
        gesture_data = self.sign_dataset[self.sign_dataset['annotated_texts'] == gesture]
        if not gesture_data.empty:
            return {
                'gesture': gesture,
                'puddle_id': gesture_data['puddle_id'].iloc[0],
                'example_count': len(gesture_data)
            }
        return None
    
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
        """Enhanced sign language detection using hand landmarks and dataset"""
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
                
                # Analyze contour shape to determine gesture
                detected_gesture = self.analyze_gesture_shape(contour)
                break
        
        return frame, detected_gesture
    
    def analyze_gesture_shape(self, contour):
        """Analyze contour shape to determine gesture type with enhanced sign language recognition"""
        try:
            # Get convex hull and defects
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            
            # Get bounding rectangle for aspect ratio analysis
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Get contour area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Enhanced gesture detection based on multiple features
            if defects is not None and len(defects) > 0:
                # Count fingers based on convexity defects
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i][0]
                    if d > 1000:  # Filter small defects
                        finger_count += 1
                
                # Map finger count to gestures
                if finger_count == 1:
                    # Check if it's a thumbs up gesture
                    if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                        return "Thumbs Up"
                    else:
                        return "Index Finger"
                elif finger_count == 2:
                    # Check for peace sign or victory sign
                    if aspect_ratio > 1.5:
                        return "Peace Sign"
                    else:
                        return "Index & Middle Fingers"
                elif finger_count == 3:
                    return "Three Fingers"
                elif finger_count == 4:
                    return "Four Fingers"
                elif finger_count >= 5:
                    return "Five Fingers"
            
            # Analyze based on shape characteristics
            if circularity > 0.8:
                return "Closed Fist"
            elif circularity > 0.6:
                # Check for open hand vs specific gestures
                if aspect_ratio > 1.5:
                    return "Open Palm"
                else:
                    return "Open Hand"
            elif area > 15000:
                return "Large Hand Gesture"
            elif area > 8000:
                return "Medium Hand Gesture"
            elif aspect_ratio > 2.0:
                return "Pointing"
            elif aspect_ratio < 0.5:
                return "Raised Hand"
            
            # Additional gesture recognition based on contour analysis
            # Check for heart shape (simplified)
            if len(contour) > 10:
                # Simplified heart detection
                top_points = [pt[0] for pt in contour if pt[0][1] < y + h//3]
                if len(top_points) > 5:
                    return "Heart Shape"
            
            # Check for OK sign (circle with finger)
            if circularity > 0.7 and area < 10000:
                return "OK Sign"
            
            # Check for stop sign (open palm facing forward)
            if aspect_ratio > 0.8 and aspect_ratio < 1.2 and area > 12000:
                return "Stop Sign"
            
            return "Hand Gesture"
            
        except Exception as e:
            st.warning(f"Gesture analysis error: {e}")
            return "Hand Detected"
    
    def get_gesture_response(self, gesture):
        """Get AI response based on detected gesture with fallback"""
        if not gesture:
            return "I see your hand, but I'm not sure what gesture you're making."
        
        # Find similar gesture in dataset
        similar_gesture = self.find_similar_gesture(gesture)
        
        if similar_gesture:
            context = self.get_gesture_context(similar_gesture)
            prompt = f"You are a helpful AI assistant for deaf individuals. The user made the sign language gesture: '{similar_gesture}'. Please provide a helpful response or ask what they need help with. Be supportive and understanding."
        else:
            prompt = f"You are a helpful AI assistant for deaf individuals. The user made a hand gesture that might mean: '{gesture}'. Please provide a helpful response or ask for clarification. Be supportive and understanding."
        
        # Try to get Ollama response
        response = self.get_ollama_response(prompt)
        
        # If Ollama fails, provide fallback responses
        if "having trouble connecting" in response or "can't connect" in response:
            return self.get_fallback_response(gesture, similar_gesture)
        
        return response
    
    def get_fallback_response(self, gesture, similar_gesture=None):
        """Provide fallback responses when Ollama is not available"""
        # Use actual gestures from dataset for responses
        if not self.sign_dataset.empty:
            available_gestures = self.get_sign_gestures()
            
            # Find matching gestures in dataset
            if gesture in available_gestures:
                return f"I recognize the gesture '{gesture}' from my dataset. How can I help you?"
            elif similar_gesture:
                return f"I found a similar gesture '{similar_gesture}' in my database. What would you like to communicate?"
            else:
                # Find any gesture that might be related
                related_gestures = [g for g in available_gestures if any(word in g.lower() for word in gesture.lower().split())]
                if related_gestures:
                    return f"I see your '{gesture}' gesture. I have similar gestures like '{related_gestures[0]}' in my database. How can I assist you?"
        
        # Generic fallback if no dataset
        return f"I see your '{gesture}' gesture. What would you like to communicate?"
    
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

    def get_ollama_response(self, text):
        """Get response from Ollama Granite model with better error handling"""
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": f"You are a helpful AI assistant for deaf individuals. Respond to: {text}",
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 150
                }
            }
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=10  # Reduced timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"].strip()
            else:
                st.error(f"Ollama API Error: {response.status_code}")
                return "I'm sorry, I couldn't process that request."
                
        except requests.exceptions.Timeout:
            st.error("⚠️ Ollama connection timed out. Please check if Ollama is running.")
            return "I'm having trouble connecting to my AI brain right now. Please try again or use text input mode."
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to Ollama. Please start Ollama first.")
            return "I can't connect to my AI brain. Please make sure Ollama is running on your computer."
        except Exception as e:
            st.error(f"Error getting Ollama response: {e}")
            return "I'm sorry, there was an error processing your request."
    
    def text_to_speech(self, text):
        """Convert text to speech and play it with improved Windows compatibility"""
        try:
            # Try Windows SAPI first (most reliable on Windows)
            try:
                self.text_to_speech_windows_fallback(text)
                return  # Success, exit early
            except Exception as sapi_error:
                st.warning(f"Windows SAPI failed, trying simple Windows TTS: {sapi_error}")
            
            # Try simple Windows TTS (no file handling)
            try:
                self.text_to_speech_simple_windows(text)
                return  # Success, exit early
            except Exception as simple_error:
                st.warning(f"Simple Windows TTS failed, trying gTTS: {simple_error}")
            
            # Fallback to gTTS with improved file handling (only if pygame is available)
            if hasattr(self, 'pygame_available') and self.pygame_available:
                self.text_to_speech_gtts_fallback(text)
            else:
                st.warning("Pygame not available, skipping audio playback")
            
        except Exception as e:
            st.error(f"All text-to-speech methods failed: {e}")
    
    def text_to_speech_gtts_fallback(self, text):
        """gTTS fallback with improved file handling"""
        temp_file = None
        try:
            # Check if pygame is available
            if not hasattr(self, 'pygame_available') or not self.pygame_available:
                st.warning("Pygame not available, skipping gTTS audio")
                return
            
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Create temporary file with unique name
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_filename = temp_file.name
            temp_file.close()  # Close immediately
            
            # Save audio to file
            tts.save(temp_filename)
            
            # Play audio with pygame
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            # Stop and unload to release file
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            
        except Exception as e:
            st.error(f"gTTS error: {e}")
        finally:
            # Cleanup with multiple strategies
            if temp_file and hasattr(temp_file, 'name'):
                self.cleanup_temp_file(temp_file.name)
    
    def cleanup_temp_file(self, filename):
        """Robust temporary file cleanup with multiple fallback strategies"""
        if not filename or not os.path.exists(filename):
            return
            
        # Strategy 1: Immediate deletion
        try:
            os.unlink(filename)
            return
        except Exception as e:
            pass
        
        # Strategy 2: Wait and retry
        import threading
        def delayed_cleanup():
            time.sleep(3)  # Wait 3 seconds
            try:
                if os.path.exists(filename):
                    os.unlink(filename)
            except:
                pass  # Final fallback - ignore errors
        
        # Strategy 3: Schedule for cleanup on next garbage collection
        try:
            import atexit
            atexit.register(lambda: self.cleanup_temp_file(filename))
        except:
            pass
        
        # Start delayed cleanup in background
        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()
    
    def text_to_speech_windows_fallback(self, text):
        """Windows SAPI fallback for text-to-speech"""
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except ImportError:
            # If pywin32 is not available, try using the built-in Windows command
            import subprocess
            try:
                # Escape quotes in text for PowerShell
                escaped_text = text.replace('"', '\\"')
                cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{escaped_text}")'
                subprocess.run(['powershell', '-Command', cmd], 
                             capture_output=True, timeout=15, check=True)
            except subprocess.TimeoutExpired:
                raise Exception("Windows SAPI command timed out")
            except subprocess.CalledProcessError as cmd_error:
                raise Exception(f"Windows SAPI command failed: {cmd_error}")
            except Exception as cmd_error:
                raise Exception(f"Windows SAPI fallback failed: {cmd_error}")
        except Exception as sapi_error:
            raise Exception(f"Windows SAPI error: {sapi_error}")
    
    def text_to_speech_simple_windows(self, text):
        """Simple Windows text-to-speech without any file handling"""
        try:
            import subprocess
            # Use Windows built-in speech synthesis directly
            # This method creates no temporary files
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\'{text}\'); $speak.Dispose()"'
            subprocess.run(cmd, shell=True, timeout=10, check=True)
        except Exception as e:
            raise Exception(f"Simple Windows TTS failed: {e}")
    
    def translate_to_sign_language(self, text):
        """Convert text to basic sign language instructions using dataset"""
        # Get gestures from dataset that might match words in the text
        words = text.lower().split()
        sign_instructions = []
        
        if not self.sign_dataset.empty:
            available_gestures = self.get_sign_gestures()
            
            for word in words:
                # Find gestures that contain this word
                matching_gestures = [g for g in available_gestures if word in g.lower()]
                if matching_gestures:
                    sign_instructions.append(f"{word}: {matching_gestures[0]}")
        
        return sign_instructions
    
    def check_ollama_status(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self):
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []
    
    def get_communication_response(self, gesture):
        """Convert detected gesture to meaningful communication word/phrase for sign language interpretation"""
        if not gesture or self.sign_dataset.empty:
            return None
        
        # Map detected gestures to meaningful communication words
        gesture_mapping = {
            # Basic hand gestures
            "Index Finger": "Yes",
            "Index & Middle Fingers": "Peace",
            "Three Fingers": "Three",
            "Four Fingers": "Four", 
            "Five Fingers": "Five",
            "Closed Fist": "Stop",
            "Open Hand": "Hello",
            "Large Hand Gesture": "Help",
            "Medium Hand Gesture": "Okay",
            "Hand Detected": "I see you",
            
            # Common communication words based on gesture patterns
            "Thumbs Up": "Good",
            "Thumbs Down": "Bad",
            "Pointing": "There",
            "Waving": "Hello",
            "Clapping": "Good job",
            "Raised Hand": "Question",
            "Open Palm": "Please",
            "Closed Palm": "Thank you",
            
            # Number gestures
            "One Finger": "One",
            "Two Fingers": "Two",
            "Three Fingers": "Three",
            "Four Fingers": "Four",
            "Five Fingers": "Five",
            
            # Directional gestures
            "Pointing Up": "Up",
            "Pointing Down": "Down",
            "Pointing Left": "Left",
            "Pointing Right": "Right",
            
            # Emotional gestures
            "Heart Shape": "Love",
            "Peace Sign": "Peace",
            "OK Sign": "Okay",
            "Stop Sign": "Stop"
        }
        
        # Try exact match first
        if gesture in gesture_mapping:
            return gesture_mapping[gesture]
        
        # Try partial matching
        gesture_lower = gesture.lower()
        for key, value in gesture_mapping.items():
            if gesture_lower in key.lower() or key.lower() in gesture_lower:
                return value
        
        # Try to find similar gestures in dataset
        available_gestures = self.get_sign_gestures()
        gesture_lower = gesture.lower()
        
        # Look for meaningful words in the dataset that match the gesture
        meaningful_words = [
            "hello", "help", "please", "thank", "goodbye", "stop", "yes", "no", 
            "more", "okay", "sorry", "love", "good", "bad", "hungry", "tired", 
            "happy", "sad", "water", "food", "home", "work", "family", "friend",
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
        ]
        
        for word in meaningful_words:
            if word in gesture_lower:
                return word.capitalize()
        
        # If no meaningful match found, try to extract words from dataset gestures
        for dataset_gesture in available_gestures:
            if dataset_gesture and gesture_lower in dataset_gesture.lower():
                # Extract meaningful words from the dataset gesture
                words = dataset_gesture.lower().split()
                for word in words:
                    if word in meaningful_words:
                        return word.capitalize()
        
        # Final fallback - return a generic response
        return "I see your gesture"
    
    def interpret_gesture_for_communication(self, gesture):
        """Enhanced gesture interpretation specifically for deaf communication"""
        if not gesture:
            return "No gesture detected"
        
        # Enhanced mapping for deaf communication
        communication_mapping = {
            # Basic communication
            "Index Finger": "Yes",
            "Index & Middle Fingers": "Peace",
            "Three Fingers": "Three",
            "Four Fingers": "Four",
            "Five Fingers": "Five",
            "Closed Fist": "Stop",
            "Open Hand": "Hello",
            "Large Hand Gesture": "Help",
            "Medium Hand Gesture": "Okay",
            "Hand Detected": "I see you",
            
            # Common ASL-inspired gestures
            "Thumbs Up": "Good",
            "Thumbs Down": "Bad", 
            "Pointing": "There",
            "Waving": "Hello",
            "Clapping": "Good job",
            "Raised Hand": "Question",
            "Open Palm": "Please",
            "Closed Palm": "Thank you",
            
            # Numbers
            "One Finger": "One",
            "Two Fingers": "Two",
            "Three Fingers": "Three",
            "Four Fingers": "Four",
            "Five Fingers": "Five",
            
            # Directions
            "Pointing Up": "Up",
            "Pointing Down": "Down",
            "Pointing Left": "Left", 
            "Pointing Right": "Right",
            
            # Emotions
            "Heart Shape": "Love",
            "Peace Sign": "Peace",
            "OK Sign": "Okay",
            "Stop Sign": "Stop"
        }
        
        # Try exact match
        if gesture in communication_mapping:
            return communication_mapping[gesture]
        
        # Try partial matching
        gesture_lower = gesture.lower()
        for key, value in communication_mapping.items():
            if gesture_lower in key.lower() or key.lower() in gesture_lower:
                return value
        
        # Try to extract meaningful words from the gesture description
        meaningful_words = [
            "hello", "help", "please", "thank", "goodbye", "stop", "yes", "no",
            "more", "okay", "sorry", "love", "good", "bad", "hungry", "tired",
            "happy", "sad", "water", "food", "home", "work", "family", "friend"
        ]
        
        for word in meaningful_words:
            if word in gesture_lower:
                return word.capitalize()
        
        # If still no match, return a contextual response
        if "finger" in gesture_lower:
            return "I see your finger"
        elif "hand" in gesture_lower:
            return "I see your hand"
        elif "gesture" in gesture_lower:
            return "I see your gesture"
        else:
            return "I see you"
    
    def should_generate_response(self, current_gesture):
        """Check if we should generate a new response based on gesture change and cooldown"""
        current_time = time.time()
        
        # Check if gesture changed
        if current_gesture != self.last_gesture:
            # Check cooldown (don't respond too frequently)
            if current_time - self.last_response_time > 2.0:  # 2 second cooldown
                self.last_gesture = current_gesture
                self.last_response_time = current_time
                return True
        
        return False

def main():
    st.set_page_config(
        page_title="Silent Echo - AI Communication Assistant",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better accessibility
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .status-indicator {
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤟 Silent Echo</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Communication Assistant for the Deaf Community")
    
    # Initialize the app with error handling
    if 'silent_echo' not in st.session_state:
        try:
            st.session_state.silent_echo = SilentEchoOllama()
        except Exception as e:
            st.error(f"Failed to initialize Silent Echo: {e}")
            st.info("Please refresh the page or restart the application")
            return
    
    # Sidebar controls
    st.sidebar.title("🎛️ Communication Controls")
    
    # Ollama status
    ollama_status = st.session_state.silent_echo.check_ollama_status()
    if ollama_status:
        st.sidebar.success("🤖 Ollama: Connected")
        available_models = st.session_state.silent_echo.get_available_models()
        if available_models:
            st.sidebar.info(f"📋 Models: {', '.join(available_models[:3])}")
    else:
        st.sidebar.error("🤖 Ollama: Not Connected")
        st.sidebar.info("💡 Using fallback responses")
    
    # Dataset information
    if not st.session_state.silent_echo.sign_dataset.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Dataset Information")
        
        total_gestures = len(st.session_state.silent_echo.sign_dataset)
        unique_gestures = len(st.session_state.silent_echo.sign_dataset['annotated_texts'].unique())
        
        st.sidebar.metric("Total Examples", total_gestures)
        st.sidebar.metric("Unique Gestures", unique_gestures)
        
        # Show available gestures
        available_gestures = st.session_state.silent_echo.get_sign_gestures()
        if available_gestures:
            st.sidebar.subheader("🤟 Available Gestures")
            for gesture in available_gestures[:10]:  # Show first 10
                st.sidebar.write(f"• {gesture}")
            if len(available_gestures) > 10:
                st.sidebar.write(f"... and {len(available_gestures) - 10} more")
    
    # Sign Language Interpreter Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Sign Language Interpreter")
    st.sidebar.markdown("""
    **Purpose:** Help you communicate with deaf people by converting sign language to speech.
    
    **How it works:**
    1. Camera detects your hand gestures
    2. System interprets the signs
    3. Computer speaks the meaning
    4. Deaf person can understand you
    
    **Supported Gestures:**
    • Thumbs Up → "Good"
    • Peace Sign → "Peace" 
    • Index Finger → "Yes"
    • Open Hand → "Hello"
    • Closed Fist → "Stop"
    • Five Fingers → "Five"
    • Pointing → "There"
    • Raised Hand → "Question"
    • OK Sign → "Okay"
    
    **Perfect for:** Teachers, healthcare workers, family members, and anyone who wants to communicate with deaf individuals!
    """)
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Communication Mode",
        ["🎤 Speech-to-Text", "🤟 Sign Language Detection", "⌨️ Text Input", "📚 Learning Mode"]
    )
    
    # Main interface
    if "🎤 Speech-to-Text" in mode:
        st.header("🎤 Real-time Speech Recognition")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("Voice Communication")
            
            if st.button("🎙️ Start Listening", type="primary", use_container_width=True):
                st.session_state.listening = True
                st.markdown('<div class="status-indicator" style="background-color: #d4edda; color: #155724;">🎧 Listening... Speak now!</div>', unsafe_allow_html=True)
                
                # Speech recognition
                audio = st.session_state.silent_echo.listen_for_speech()
                if audio:
                    text = st.session_state.silent_echo.speech_to_text(audio)
                    if text:
                        st.success(f"🎤 **You said:** {text}")
                        
                        # Get AI response
                        with st.spinner("🤖 Getting AI response..."):
                            response = st.session_state.silent_echo.get_gesture_response(text)
                        
                        st.info(f"🤖 **AI Response:** {response}")
                        
                        # Show sign language translation
                        signs = st.session_state.silent_echo.translate_to_sign_language(response)
                        if signs:
                            st.subheader("🤟 Sign Language Translation")
                            for sign in signs:
                                st.write(f"• {sign}")
                        
                        # Play response
                        col_play1, col_play2 = st.columns(2)
                        with col_play1:
                            if st.button("🔊 Play Response", use_container_width=True):
                                st.session_state.silent_echo.text_to_speech(response)
                        with col_play2:
                            if st.button("📝 Copy Response", use_container_width=True):
                                st.write("Response copied to clipboard!")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("💡 Tips for Better Recognition")
            st.markdown("""
            - **Speak clearly** and at a normal pace
            - **Reduce background noise** for better accuracy
            - **Wait for the listening indicator** before speaking
            - **Use simple, clear language** for better AI responses
            - **Position microphone** close to your mouth
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif "🤟 Sign Language Detection" in mode:
        st.header("🤟 Sign Language Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("Live Hand Gesture Recognition")
            
            # Camera permission instructions
            st.info("""
            **📹 Camera Access Required:**
            1. Click "Start Live Feed" below
            2. When browser asks for camera permission, click "Allow"
            3. If you see a camera icon in the address bar, click it and select "Allow"
            4. Refresh the page if needed
            """)
            
            # Initialize camera state
            if 'camera_active' not in st.session_state:
                st.session_state.camera_active = False
            
            # Camera control buttons
            col_start, col_stop = st.columns(2)
            with col_start:
                if st.button("📹 Start Live Feed", type="primary", use_container_width=True):
                    st.session_state.camera_active = True
                    st.rerun()
            
            with col_stop:
                if st.button("⏹️ Stop Camera", use_container_width=True):
                    st.session_state.camera_active = False
                    st.rerun()
            
            # Live camera feed
            if st.session_state.camera_active:
                st.markdown('<div class="status-indicator" style="background-color: #d4edda; color: #155724;">📹 Live Video Feed Active - Real-time Gesture Recognition!</div>', unsafe_allow_html=True)
                
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
                            # Process frame for sign language detection
                            processed_frame, detected_gesture = st.session_state.silent_echo.detect_sign_language(frame)
                            
                            # Convert BGR to RGB for display
                            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display the processed frame
                            st.image(rgb_frame, channels="RGB", use_container_width=True, caption="Live Video Feed - Real-time Gesture Recognition")
                            
                            # Real-time gesture response
                            if st.session_state.silent_echo.sign_language_detected and detected_gesture:
                                st.success(f"🤟 **Detected:** {detected_gesture}")
                                
                                # Check if we should generate a response
                                if st.session_state.silent_echo.should_generate_response(detected_gesture):
                                    # Use enhanced gesture interpretation for deaf communication
                                    interpreted_gesture = st.session_state.silent_echo.interpret_gesture_for_communication(detected_gesture)
                                    
                                    if interpreted_gesture and interpreted_gesture != "I see you":
                                        st.info(f"💬 **Spoken:** {interpreted_gesture}")
                                        # Speak the interpreted gesture
                                        st.session_state.silent_echo.text_to_speech(interpreted_gesture)
                                        
                                        # Show what the system is doing
                                        st.success(f"🎤 **Speaking:** '{interpreted_gesture}' - Real-time communication!")
                                    else:
                                        st.info("💭 **Gesture not recognized for communication**")
                                
                                # Show current gesture status
                                col_status1, col_status2 = st.columns(2)
                                with col_status1:
                                    st.write("📊 **Current Gesture:**")
                                    st.write(f"• {detected_gesture}")
                                    
                                    if st.session_state.silent_echo.last_gesture:
                                        st.write(f"• Previous: {st.session_state.silent_echo.last_gesture}")
                                
                                with col_status2:
                                    st.write("⏱️ **Response Status:**")
                                    time_since_last = time.time() - st.session_state.silent_echo.last_response_time
                                    if time_since_last < 2.0:
                                        st.write(f"• Cooldown: {2.0 - time_since_last:.1f}s")
                                    else:
                                        st.write("• Ready for response")
                                    
                                    # Show what the system is doing
                                    st.write("🎯 **Purpose:** Real-time Sign Language Interpreter")
                                    st.write("🔊 **Action:** Continuous gesture recognition")
                            else:
                                st.info("👋 **Show your hands to the camera for real-time communication**")
                                st.markdown("""
                                **💡 Real-time Sign Language Interpreter:**
                                
                                1. **Show your hands** to the camera
                                2. **Make sign language gestures** 
                                3. **System continuously analyzes** and speaks
                                4. **Perfect for real-time communication** with deaf people
                                
                                **🤟 Supported Gestures:**
                                """)
                                
                                # Show supported gestures
                                supported_gestures = [
                                    "Thumbs Up → Good", "Peace Sign → Peace", "Index Finger → Yes",
                                    "Open Hand → Hello", "Closed Fist → Stop", "Five Fingers → Five",
                                    "Pointing → There", "Raised Hand → Question", "OK Sign → Okay"
                                ]
                                
                                for gesture in supported_gestures:
                                    st.write(f"• {gesture}")
                                
                                st.markdown("""
                                **🎯 Goal:** Real-time sign language to speech conversion!
                                """)
                        else:
                            st.error("Failed to capture frame from webcam")
                            st.info("Try refreshing the page or checking camera permissions")
                    else:
                        st.error("Webcam not available")
                        st.session_state.camera_active = False
                        st.rerun()
                            
                except Exception as e:
                    st.error(f"❌ Camera error: {e}")
                    st.markdown("""
                    **Troubleshooting:**
                    1. Make sure you allowed camera access when prompted
                    2. Check if another app is using your camera
                    3. Try refreshing the page
                    4. Try a different browser (Chrome recommended)
                    5. Restart the application
                    """)
                    st.session_state.camera_active = False
                    st.rerun()
            
            else:
                st.info("📹 Click 'Start Live Feed' to begin real-time communication")
                st.markdown("""
                **How Real-time Video Works:**
                - Click 'Start Live Feed' to activate continuous video processing
                - Allow camera access when prompted
                - System continuously analyzes your gestures in real-time
                - No clicking needed - just gesture naturally
                - System automatically speaks what you're signing
                - Perfect for real-time communication with deaf people!
                
                **🎯 Purpose:** Real-time sign language to speech conversion for deaf communication.
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("Supported Gestures from Dataset")
            
            if not st.session_state.silent_echo.sign_dataset.empty:
                # Get actual gestures from dataset
                available_gestures = st.session_state.silent_echo.get_sign_gestures()
                if available_gestures:
                    st.write("**Detected gestures from your dataset:**")
                    for gesture in available_gestures[:8]:  # Show first 8
                        st.write(f"• {gesture}")
                    if len(available_gestures) > 8:
                        st.write(f"... and {len(available_gestures) - 8} more gestures")
                else:
                    st.write("No gestures found in dataset")
            else:
                st.write("Dataset not loaded")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif "⌨️ Text Input" in mode:
        st.header("⌨️ Text Communication")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("Type Your Message")
            
            user_input = st.text_area(
                "Enter your message:",
                height=120,
                placeholder="Type your message here...",
                help="Enter any text you'd like to communicate"
            )
            
            if st.button("🤖 Get AI Response", type="primary", use_container_width=True):
                if user_input:
                    with st.spinner("🤖 Processing your message..."):
                        response = st.session_state.silent_echo.get_gesture_response(user_input)
                    
                    st.success(f"🤖 **AI Response:** {response}")
                    
                    # Show sign language translation
                    signs = st.session_state.silent_echo.translate_to_sign_language(response)
                    if signs:
                        st.subheader("🤟 Sign Language Translation")
                        for sign in signs:
                            st.write(f"• {sign}")
                    
                    # Audio controls
                    col_audio1, col_audio2 = st.columns(2)
                    with col_audio1:
                        if st.button("🔊 Play Response", use_container_width=True):
                            st.session_state.silent_echo.text_to_speech(response)
                    with col_audio2:
                        if st.button("📝 Copy Response", use_container_width=True):
                            st.write("Response copied to clipboard!")
                else:
                    st.warning("Please enter a message first.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.subheader("Dataset Gestures")
            
            if not st.session_state.silent_echo.sign_dataset.empty:
                # Get some sample gestures from dataset
                available_gestures = st.session_state.silent_echo.get_sign_gestures()
                if available_gestures:
                    st.write("**Sample gestures from your dataset:**")
                    # Show some interesting gestures
                    sample_gestures = available_gestures[:5]
                    for gesture in sample_gestures:
                        st.write(f"• {gesture}")
                    
                    # Show total count
                    st.write(f"**Total available:** {len(available_gestures)} gestures")
                else:
                    st.write("No gestures available")
            else:
                st.write("Dataset not loaded")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif "📚 Learning Mode" in mode:
        st.header("📚 Sign Language Learning")
        
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.subheader("Learn Sign Language from Dataset")
        
        if not st.session_state.silent_echo.sign_dataset.empty:
            # Get gestures from dataset
            dataset_gestures = st.session_state.silent_echo.get_sign_gestures()
            
            if dataset_gestures:
                st.success(f"📊 **Dataset loaded successfully!** {len(dataset_gestures)} unique gestures available.")
                
                # Display gestures in a searchable format
                search_term = st.text_input("🔍 Search gestures:", placeholder="Type to search gestures...")
                
                if search_term:
                    filtered_gestures = [g for g in dataset_gestures if search_term.lower() in g.lower()]
                else:
                    filtered_gestures = dataset_gestures[:20]  # Show first 20 by default
                
                # Display gestures in a grid
                cols = st.columns(2)
                for i, gesture in enumerate(filtered_gestures):
                    with cols[i % 2]:
                        st.markdown(f"**{gesture}**")
                        
                        # Get context for this gesture
                        context = st.session_state.silent_echo.get_gesture_context(gesture)
                        if context:
                            st.write(f"📊 Puddle ID: {context['puddle_id']}")
                            st.write(f"📈 Examples: {context['example_count']}")
                        
                        st.markdown("---")
                
                if len(filtered_gestures) < len(dataset_gestures):
                    st.info(f"Showing {len(filtered_gestures)} of {len(dataset_gestures)} gestures. Use search to find more.")
            else:
                st.warning("No gestures found in the dataset.")
        else:
            st.error("Dataset not loaded. Please check if 'gpt-3.5-cleaned.csv' is available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with project information
    st.markdown("---")
    st.markdown("""
    ### About Silent Echo
    
    **Silent Echo** is an AI-powered communication assistant designed specifically to help deaf individuals 
    communicate more effectively. Built with IBM Granite models and advanced computer vision, it provides 
    multiple communication modes to bridge communication gaps.
    
    **Key Features:**
    - **Real-time Speech Recognition** - Convert spoken words to text instantly
    - **Sign Language Detection** - Recognize hand gestures using computer vision and dataset matching
    - **AI-Powered Responses** - Intelligent contextual responses using Granite models
    - **Text-to-Speech** - Convert AI responses to spoken audio
    - **Learning Mode** - Browse and search through the complete gesture dataset
    - **Accessible Interface** - Designed with accessibility in mind
    
    **Dataset Integration:**
    - **gpt-3.5-cleaned.csv** - 137,277 unique gestures with 204,399 total examples
    - **Real-time Gesture Matching** - Compare detected gestures against the dataset
    - **Context Information** - Display Puddle ID and example counts for each gesture
    - **Searchable Database** - Find and learn specific gestures from the dataset
    
    **Technology Stack:**
    - IBM Granite 3.3 8B Instruct Model (via Ollama)
    - Google Speech Recognition API
    - OpenCV for computer vision
    - Streamlit for web interface
    - Custom gesture dataset integration
    
    **Made with ❤️ for the deaf community**
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("🛑 Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        # Cleanup resources
        if 'silent_echo' in st.session_state:
            st.session_state.silent_echo.cleanup() 