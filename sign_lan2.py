import cv2
import numpy as np
import time
import queue
import os
import logging
from PIL import Image
import requests
import json
from gtts import gTTS
import tempfile
import pygame
import pandas as pd
import aiohttp
import asyncio
import speech_recognition as sr
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import sys

# Configuration
@dataclass
class Config:
    """Configuration class for Silent Echo"""
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "granite3.3:8b"
    
    # Audio settings
    audio_timeout: float = 5.0
    phrase_time_limit: float = 10.0
    ambient_noise_duration: float = 1.0
    
    # Video settings
    min_contour_area: int = 5000
    gesture_cooldown: float = 2.0
    frame_rate: int = 30
    
    # AI settings
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 150
    
    # Debug settings
    debug_mode: bool = False
    save_frames: bool = False
    log_level: str = "INFO"

# Global configuration
config = Config()

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('silent_echo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SilentEchoOllama:
    def __init__(self):
        logger.info("Initializing Silent Echo - AI Communication Assistant")
        
        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'audio_processed': 0,
            'ai_responses': 0,
            'start_time': time.time()
        }
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.setup_microphone()
        
        # Audio queue for continuous listening
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.listening_thread = None
        
        # Video components
        self.cap = None
        self.sign_language_detected = False
        self.last_gesture = None
        self.last_response_time = time.time()
        
        # Enhanced gesture detection
        self.setup_gesture_detection()
        
        # Initialize pygame for audio playback
        self.setup_audio()
        
        # Load sign language dataset
        self.sign_dataset = self.load_sign_dataset()
        logger.info(f"Loaded {len(self.sign_dataset)} sign language examples")
        
        # Response cache for performance
        self.response_cache = {}
        self.cache_size = 100

    def ensure_stats_keys(self):
        """Ensure all required stats keys exist"""
        required_keys = ['frames_processed', 'gestures_detected', 'audio_processed', 'ai_responses', 'start_time']
        for key in required_keys:
            if key not in self.stats:
                self.stats[key] = 0 if key != 'start_time' else time.time()

    def setup_gesture_detection(self):
        """Setup enhanced gesture detection with MediaPipe fallback"""
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
            logger.info("MediaPipe hand detection enabled")
        except ImportError:
            logger.info("MediaPipe not available, using OpenCV contour detection")
    
    def setup_audio(self):
        """Setup audio components with better error handling"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            logger.info("Pygame audio initialized successfully")
        except Exception as e:
            logger.error(f"Pygame initialization error: {e}")
            # Try alternative initialization
            try:
                pygame.mixer.init()
                logger.info("Pygame initialized with default settings")
            except Exception as e2:
                logger.error(f"Pygame initialization completely failed: {e2}")
    
    def setup_microphone(self):
        """Setup microphone with improved fallback options"""
        try:
            # Try default microphone first
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=config.ambient_noise_duration)
            logger.info("Microphone setup successful with default device")
        except Exception as e:
            logger.warning(f"Default microphone setup failed: {e}")
            # Try alternative device indices
            for device_index in [1, 2, 3, 4, 5]:
                try:
                    self.microphone = sr.Microphone(device_index=device_index)
                    with self.microphone as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=config.ambient_noise_duration)
                    logger.info(f"Microphone setup successful with device {device_index}")
                    break
                except Exception as e2:
                    logger.debug(f"Device {device_index} failed: {e2}")
                    continue
            if not self.microphone:
                logger.error("No working microphone found")

    def load_sign_dataset(self):
        try:
            df = pd.read_csv('gpt-3.5-cleaned.csv')
            return df[df['annotated_texts'].notna() & (df['annotated_texts'] != '')]
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return pd.DataFrame()

    def get_sign_gestures(self):
        if not self.sign_dataset.empty:
            return self.sign_dataset['annotated_texts'].unique().tolist()
        return []

    async def get_ollama_response(self, text):
        try:
            payload = {"model": config.model_name, "prompt": f"You are a helpful AI assistant for deaf individuals. Respond to: {text}", "stream": False, "options": {"temperature": config.temperature, "top_p": config.top_p, "num_predict": config.max_tokens}}
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{config.ollama_base_url}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["response"].strip()
                    print(f"Ollama API Error: {response.status}")
                    return "Error processing request."
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Ollama connection issue."

    def get_ollama_response_sync(self, text):
        """Enhanced AI response with caching, better prompts, and comprehensive error handling"""
        # Check cache first
        if text in self.response_cache:
            logger.debug(f"Cache hit for: {text}")
            return self.response_cache[text]
        
        # Check if Ollama is available first
        if not self.check_ollama_status():
            logger.warning("Ollama not available, using fallback responses")
            return self.get_fallback_response(text)
        
        try:
            # Enhanced prompt for better responses
            enhanced_prompt = f"""You are Silent Echo, an AI assistant designed to help with communication for the deaf community. 
            Your role is to be helpful, clear, and supportive. The user said: "{text}"
            
            Please provide a helpful response that:
            1. Acknowledges their message
            2. Offers assistance or information
            3. Uses clear, simple language
            4. Is encouraging and supportive
            
            Response:"""
            
            payload = {
                "model": config.model_name,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "num_predict": config.max_tokens
                }
            }
            
            logger.info(f"Sending request to Ollama: {text[:50]}...")
            response = requests.post(f"{config.ollama_base_url}/api/generate", json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result["response"].strip()
                
                # Cache the response
                if len(self.response_cache) >= self.cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.response_cache))
                    del self.response_cache[oldest_key]
                
                self.response_cache[text] = ai_response
                self.stats['ai_responses'] += 1
                logger.info(f"AI response generated successfully: {ai_response[:50]}...")
                
                return ai_response
            else:
                logger.error(f"Ollama API Error: {response.status_code} - {response.text}")
                return self.get_fallback_response(text, f"API Error {response.status_code}"),
                
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return self.get_fallback_response(text, "timeout")
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama server")
            return self.get_fallback_response(text, "connection")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return self.get_fallback_response(text, str(e))
    
    def get_fallback_response(self, text, error_type="unknown"):
        """Provide intelligent fallback responses based on input and error type"""
        text_lower = text.lower()
        
        # Common greeting patterns
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "Hello! I'm here to help you communicate. How can I assist you today?"
        
        # Question patterns
        if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return "That's a great question. I'd be happy to help you find the answer. Could you provide more details?"
        
        # Help requests
        if any(word in text_lower for word in ['help', 'assist', 'support', 'need']):
            return "I'm here to help! I can assist with communication, answer questions, or just chat with you. What do you need?"
        
        # Thank you patterns
        if any(word in text_lower for word in ['thank', 'thanks', 'appreciate']):
            return "You're very welcome! I'm glad I could help. Is there anything else you'd like assistance with?"
        
        # Goodbye patterns
        if any(word in text_lower for word in ['bye', 'goodbye', 'see you', 'later']):
            return "Goodbye! It was nice talking with you. Take care!"
        
        # Weather related
        if any(word in text_lower for word in ['weather', 'temperature', 'rain', 'sunny']):
            return "I can't check the weather right now, but I'd be happy to help you with other questions or tasks."
        
        # Time related
        if any(word in text_lower for word in ['time', 'clock', 'hour', 'minute']):
            return f"The current time is {time.strftime('%I:%M %p')}. Is there anything else you'd like to know?"
        
        # Default responses based on error type
        if error_type == "timeout":
            return "I'm taking a bit longer to respond than usual. Let me try a simpler approach: How can I help you today?"
        elif error_type == "connection":
            return "I'm having trouble connecting to my AI brain right now, but I can still help you with basic communication. What would you like to talk about?"
        elif "API Error" in error_type:
            return "I'm experiencing some technical difficulties with my advanced features, but I'm still here to help you communicate. What's on your mind?"
        else:
            return "I understand what you're saying. Let me help you with that. Could you tell me more about what you need assistance with?"
    
    def diagnose_system_issues(self):
        """Comprehensive system diagnostics"""
        issues = []
        warnings = []
        
        print("\n🔍 System Diagnostics:")
        print("=" * 40)
        
        # Check Ollama
        print("🤖 Checking Ollama...")
        if self.check_ollama_status():
            print("   ✅ Ollama server is running")
            try:
                models = self.get_available_models()
                if models:
                    print(f"   ✅ Available models: {', '.join(models[:3])}")
                    if config.model_name not in models:
                        warnings.append(f"Configured model '{config.model_name}' not found. Available: {models[:3]}")
                else:
                    warnings.append("No models found in Ollama")
            except Exception as e:
                warnings.append(f"Error checking models: {e}")
        else:
            issues.append("Ollama server not accessible")
            print("   ❌ Ollama server not accessible")
            print("   💡 Try: ollama serve")
        
        # Check microphone
        print("🎤 Checking microphone...")
        if self.microphone:
            print("   ✅ Microphone is available")
        else:
            issues.append("No microphone found")
            print("   ❌ No microphone found")
            print("   💡 Check microphone permissions and connections")
        
        # Check webcam
        print("📹 Checking webcam...")
        if self.cap and self.cap.isOpened():
            print("   ✅ Webcam is working")
        else:
            issues.append("Webcam not accessible")
            print("   ❌ Webcam not accessible")
            print("   💡 Check camera permissions and connections")
        
        # Check dataset
        print("📊 Checking dataset...")
        if not self.sign_dataset.empty:
            print(f"   ✅ Dataset loaded: {len(self.sign_dataset)} examples")
        else:
            warnings.append("No sign language dataset loaded")
            print("   ⚠️ No sign language dataset loaded")
            print("   💡 Ensure 'gpt-3.5-cleaned.csv' is in the current directory")
        
        # Check gesture detection
        print("🎯 Checking gesture detection...")
        if self.use_mediapipe:
            print("   ✅ MediaPipe hand detection enabled")
        else:
            print("   ⚠️ Using OpenCV contour detection (MediaPipe not available)")
            print("   💡 Install MediaPipe for better accuracy: pip install mediapipe")
        
        # Performance check
        print("⚡ Checking performance...")
        stats = self.get_performance_stats()
        if stats['frames_per_second'] < 5:
            warnings.append(f"Low FPS detected: {stats['frames_per_second']:.1f}")
            print(f"   ⚠️ Low FPS: {stats['frames_per_second']:.1f}")
            print("   💡 Try reducing frame rate or resolution")
        else:
            print(f"   ✅ Good performance: {stats['frames_per_second']:.1f} FPS")
        
        # Summary
        print("\n📋 Summary:")
        if issues:
            print(f"   ❌ Critical issues: {len(issues)}")
            for issue in issues:
                print(f"      - {issue}")
        if warnings:
            print(f"   ⚠️ Warnings: {len(warnings)}")
            for warning in warnings:
                print(f"      - {warning}")
        if not issues and not warnings:
            print("   ✅ All systems operational")
        
        return issues, warnings

    def text_to_speech(self, text):
        """Convert text to speech with multiple fallback options"""
        if not text:
            return
        
        print(f"🔊 Speaking: {text}")
        
        # Try Windows SAPI first (if available)
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            return
        except ImportError:
            print("Windows SAPI not available, using gTTS")
        except Exception as e:
            print(f"Windows SAPI failed: {e}")
        
        # Fallback to gTTS
        self.text_to_speech_gtts_fallback(text)
    
    def text_to_speech_gtts_fallback(self, text):
        """gTTS fallback with improved error handling"""
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_file.name)
            
            pygame.mixer.music.load(temp_file.name)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.2)
            
        except Exception as e:
            print(f"gTTS error: {e}")
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except Exception as e:
                    print(f"Could not delete temp file: {e}")

    def start_webcam(self):
        """Robust webcam initialization with better error handling"""
        # First, try to release any existing camera
        if hasattr(self, 'cap') and self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        # Try different camera backends in order of preference
        backends = [
            (cv2.CAP_MSMF, "Media Foundation"),  # Most stable on Windows
            (cv2.CAP_DSHOW, "DirectShow"),       # Fallback
            (cv2.CAP_ANY, "Auto-detect")         # Last resort
        ]
        
        camera_indices = [0]  # Start with index 0 only to avoid conflicts
        
        for backend_code, backend_name in backends:
            for index in camera_indices:
                try:
                    logger.info(f"Trying camera index {index} with {backend_name}")
                    
                    # Create VideoCapture with specific backend
                    cap = cv2.VideoCapture(index, backend_code)
                    
                    # Wait a moment for initialization
                    time.sleep(0.5)
                    
                    if cap.isOpened():
                        # Set camera properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, config.frame_rate)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Test frame reading with timeout
                        start_time = time.time()
                        frame_read = False
                        
                        while time.time() - start_time < 3.0:  # 3 second timeout
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                frame_read = True
                                break
                            time.sleep(0.1)
                        
                        if frame_read:
                            self.cap = cap
                            logger.info(f"Successfully opened camera {index} with {backend_name}")
                            return True
                        else:
                            logger.warning(f"Camera {index} opened but can't read frames")
                            cap.release()
                    else:
                        logger.debug(f"Failed to open camera {index} with {backend_name}")
                        if cap:
                            cap.release()
                            
                except Exception as e:
                    logger.debug(f"Error trying camera {index} with {backend_name}: {e}")
                    if 'cap' in locals() and cap:
                        try:
                            cap.release()
                        except:
                            pass
        
        # If all attempts failed, try one more time with default settings
        try:
            logger.info("Trying default camera initialization")
            cap = cv2.VideoCapture(0)
            time.sleep(0.5)
            
            if cap.isOpened():
                # Test frame reading
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.cap = cap
                    logger.info("Default camera initialization successful")
                    return True
                else:
                    cap.release()
        except Exception as e:
            logger.error(f"Default camera initialization failed: {e}")
            if 'cap' in locals() and cap:
                try:
                    cap.release()
                except:
                    pass
        
        logger.error("All camera initialization attempts failed")
        return False

    def stop_webcam(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def detect_sign_language(self, frame):
        """Enhanced sign language detection with MediaPipe support"""
        self.sign_language_detected = False
        detected_gesture = None
        
        if self.use_mediapipe:
            detected_gesture = self.detect_with_mediapipe(frame)
        else:
            detected_gesture = self.detect_with_opencv(frame)
        
        self.stats['frames_processed'] += 1
        if detected_gesture:
            self.stats['gestures_detected'] += 1
        
        return frame, detected_gesture
    
    def detect_with_mediapipe(self, frame):
        """Detect gestures using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    import mediapipe as mp
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    
                    # Analyze gesture
                    gesture = self.analyze_mediapipe_landmarks(hand_landmarks)
                    if gesture:
                        self.sign_language_detected = True
                        return gesture
        except Exception as e:
            logger.error(f"MediaPipe detection error: {e}")
        
        return None
    
    def detect_with_opencv(self, frame):
        """Detect gestures using OpenCV contour detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > config.min_contour_area:
                    self.sign_language_detected = True
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                    return self.analyze_gesture_shape(contour)
        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
        
        return None
    
    def analyze_mediapipe_landmarks(self, landmarks):
        """Analyze MediaPipe hand landmarks for gesture detection"""
        try:
            # Get finger tip and pip landmarks
            finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
            finger_pips = [6, 10, 14, 18]  # Corresponding pip joints
            
            # Check if fingers are extended
            extended_fingers = []
            for tip, pip in zip(finger_tips, finger_pips):
                if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
                    extended_fingers.append(True)
                else:
                    extended_fingers.append(False)
            
            # Thumb check (different orientation)
            thumb_tip = landmarks.landmark[4]
            thumb_ip = landmarks.landmark[3]
            thumb_extended = thumb_tip.x > thumb_ip.x if thumb_tip.x > 0.5 else thumb_tip.x < thumb_ip.x
            
            total_extended = sum(extended_fingers) + (1 if thumb_extended else 0)
            
            # Gesture classification
            if total_extended == 0:
                return "Closed Fist"
            elif total_extended == 1 and thumb_extended:
                return "Thumbs Up"
            elif total_extended == 1 and extended_fingers[0]:  # Index finger
                return "Index Finger"
            elif total_extended == 2 and extended_fingers[0] and extended_fingers[1]:
                return "Peace Sign"
            elif total_extended == 5:
                return "Open Palm"
            elif total_extended >= 3:
                return "Multiple Fingers"
            else:
                return "Hand Gesture"
                
        except Exception as e:
            logger.error(f"MediaPipe landmark analysis error: {e}")
            return None
    
    def analyze_gesture_shape(self, contour):
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            if defects is not None and len(defects) > 0:
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, depth = defects[i, 0]
                    if depth > 1000:  # depth threshold
                        finger_count += 1

                if finger_count == 0:
                    # Possibly closed fist
                    if circularity > 0.7:
                        return "Closed Fist"
                    else:
                        return "Hand Gesture"
                elif finger_count == 1:
                    if 0.8 < aspect_ratio < 1.2:
                        return "Thumbs Up"
                    else:
                        return "Index Finger"
                elif finger_count == 2:
                    if aspect_ratio > 1.5:
                        return "Peace Sign"
                    else:
                        return "Two Fingers"
                elif finger_count >= 4:
                    return "Open Palm"
                else:
                    return "Hand Gesture"
            else:
                # No defects found
                if circularity > 0.7 and area < 10000:
                    return "OK Sign"
                elif aspect_ratio > 1.5 and area > 12000:
                    return "Open Palm"
                else:
                    return "Closed Fist"
        except Exception as e:
            print(f"Gesture analysis error: {e}")
            return "Hand Detected"

    def interpret_gesture_for_communication(self, gesture):
        if not gesture or self.sign_dataset.empty:
            return "No gesture detected"
        available_gestures = self.get_sign_gestures()
        gesture_lower = gesture.lower()
        for g in available_gestures:
            if gesture_lower in g.lower():
                return g
        communication_mapping = {
            "Index Finger": "Yes", "Peace Sign": "Peace", "Open Hand": "Hello",
            "Closed Fist": "Stop", "Thumbs Up": "Good", "OK Sign": "Okay"
        }
        for key, value in communication_mapping.items():
            if gesture_lower in key.lower():
                return value
        return "I see your gesture"

    def should_generate_response(self, current_gesture):
        current_time = time.time()
        if current_gesture != self.last_gesture and current_time - self.last_response_time > config.gesture_cooldown:
            self.last_gesture = current_gesture
            self.last_response_time = current_time
            return True
        return False

    def check_ollama_status(self):
        try:
            response = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_models(self):
        try:
            response = requests.get(f"{config.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []

    def cleanup(self):
        self.is_listening = False
        self.stop_webcam()
        try:
            pygame.mixer.quit()
        except:
            pass

    def start_continuous_listening(self):
        """Start continuous listening in background thread"""
        if not self.microphone:
            print("No microphone available for continuous listening.")
            return False
        
        if self.is_listening:
            print("Continuous listening is already active.")
            return True
        
        self.is_listening = True
        self.listening_thread = threading.Thread(target=self._continuous_listen_worker, daemon=True)
        self.listening_thread.start()
        print("🎙️ Continuous listening started!")
        return True
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=2.0)
        print("🛑 Continuous listening stopped.")
    
    def _continuous_listen_worker(self):
        """Background worker for continuous audio capture"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                while self.is_listening:
                    try:
                        # Listen for audio with shorter timeout for responsiveness
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                        if audio:
                            self.audio_queue.put({
                                'audio': audio,
                                'timestamp': time.time()
                            })
                    except sr.WaitTimeoutError:
                        # No audio detected, continue listening
                        continue
                    except Exception as e:
                        print(f"Audio capture error: {e}")
                        time.sleep(0.1)
        except Exception as e:
            print(f"Continuous listening worker error: {e}")
        finally:
            self.is_listening = False
    
    def process_audio_queue(self):
        """Process queued audio data and convert to text"""
        if self.audio_queue.empty():
            return None
        
        try:
            audio_data = self.audio_queue.get_nowait()
            audio = audio_data['audio']
            
            # Convert audio to text
            text = self.speech_to_text(audio)
            if text:
                return {
                    'text': text,
                    'timestamp': audio_data['timestamp']
                }
        except queue.Empty:
            return None
        except Exception as e:
            print(f"Audio processing error: {e}")
        
        return None
    
    def listen_for_speech(self):
        """Listen for speech input"""
        if not self.microphone:
            print("No microphone available.")
            return None
        try:
            print("Listening... Please speak.")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Stopped listening.")
                return audio
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")
            return None
        except Exception as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def speech_to_text(self, audio):
        """Convert speech to text"""
        if not audio:
            return None
        try:
            text = self.recognizer.recognize_google(audio, language="en-US")
            return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics with safety checks"""
        self.ensure_stats_keys()
        runtime = time.time() - self.stats['start_time']
        return {
            'runtime_seconds': runtime,
            'frames_per_second': self.stats['frames_processed'] / runtime if runtime > 0 else 0,
            'gestures_per_minute': (self.stats['gestures_detected'] / runtime) * 60 if runtime > 0 else 0,
            'gestures_detected': self.stats['gestures_detected'],
            'audio_processed': self.stats['audio_processed'],
            'ai_responses': self.stats['ai_responses'],
            'cache_hit_rate': len(self.response_cache) / self.cache_size if self.cache_size > 0 else 0
        }

    def get_troubleshooting_guide(self, issue_type=None):
        """Get specific troubleshooting steps for different issues"""
        guides = {
            "ollama": {
                "title": "🤖 Ollama Connection Issues",
                "steps": [
                    "1. Check if Ollama is installed: ollama --version",
                    "2. Start Ollama server: ollama serve",
                    "3. Verify server is running: curl http://localhost:11434/api/tags",
                    "4. Check if model is downloaded: ollama list",
                    "5. Download model if needed: ollama pull granite3.3:8b",
                    "6. Check firewall/antivirus settings",
                    "7. Try different port if 11434 is blocked"
                ]
            },
            "microphone": {
                "title": "🎤 Microphone Issues",
                "steps": [
                    "1. Check microphone permissions in Windows Settings",
                    "2. Verify microphone is not muted",
                    "3. Test microphone in Windows Sound settings",
                    "4. Try different microphone device",
                    "5. Update audio drivers",
                    "6. Check if microphone is being used by another app",
                    "7. Restart the application"
                ]
            },
            "webcam": {
                "title": "📹 Webcam Issues",
                "steps": [
                    "1. Check camera permissions in Windows Settings",
                    "2. Verify camera is not being used by another app",
                    "3. Test camera in Windows Camera app",
                    "4. Update camera drivers",
                    "5. Try different camera index (0, 1, 2)",
                    "6. Check USB connection if external camera",
                    "7. Restart the application"
                ]
            },
            "performance": {
                "title": "⚡ Performance Issues",
                "steps": [
                    "1. Reduce frame rate in config (frame_rate: 15)",
                    "2. Lower resolution by modifying webcam settings",
                    "3. Close other applications using camera/microphone",
                    "4. Check CPU and memory usage",
                    "5. Update graphics drivers",
                    "6. Try running as administrator",
                    "7. Restart computer if issues persist"
                ]
            },
            "ai_responses": {
                "title": "🤖 AI Response Issues",
                "steps": [
                    "1. Check Ollama server status",
                    "2. Verify model is downloaded and working",
                    "3. Check internet connection for fallback responses",
                    "4. Try different model: ollama pull llama2:7b",
                    "5. Increase timeout in config (audio_timeout: 10.0)",
                    "6. Check system resources (CPU/RAM)",
                    "7. Restart Ollama server: ollama serve"
                ]
            }
        }
        
        if issue_type and issue_type in guides:
            guide = guides[issue_type]
            print(f"\n{guide['title']}")
            print("=" * len(guide['title']))
            for step in guide['steps']:
                print(f"   {step}")
        else:
            print("\n🔧 General Troubleshooting Guide:")
            print("=" * 35)
            for issue, guide in guides.items():
                print(f"   • {guide['title']}")
            print("\n💡 Use 'd' key to run diagnostics and identify specific issues")
    
    def quick_fix_common_issues(self):
        """Attempt to automatically fix common issues"""
        print("\n🔧 Attempting Quick Fixes...")
        fixes_applied = []
        
        # Try to restart Ollama connection
        if not self.check_ollama_status():
            print("   🔄 Attempting to reconnect to Ollama...")
            try:
                # Wait a moment and retry
                time.sleep(2)
                if self.check_ollama_status():
                    fixes_applied.append("Ollama connection restored")
                    print("   ✅ Ollama connection restored")
                else:
                    print("   ❌ Ollama still not accessible")
            except Exception as e:
                print(f"   ❌ Ollama reconnection failed: {e}")
        
        # Try different camera index
        if not self.cap or not self.cap.isOpened():
            print("   🔄 Trying different camera index...")
            for index in [1, 2, 3]:
                try:
                    test_cap = cv2.VideoCapture(index)
                    if test_cap.isOpened():
                        test_cap.release()
                        self.cap = cv2.VideoCapture(index)
                        if self.cap.isOpened():
                            fixes_applied.append(f"Camera connected on index {index}")
                            print(f"   ✅ Camera connected on index {index}")
                            break
                        else:
                            test_cap.release()
                except Exception as e:
                    print(f"   ❌ Camera index {index} failed: {e}")
        
        # Try different microphone device
        if not self.microphone:
            print("   🔄 Trying different microphone device...")
            for device_index in [1, 2, 3, 4, 5]:
                try:
                    test_mic = sr.Microphone(device_index=device_index)
                    with test_mic as source:
                        self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.microphone = test_mic
                    fixes_applied.append(f"Microphone connected on device {device_index}")
                    print(f"   ✅ Microphone connected on device {device_index}")
                    break
                except Exception as e:
                    print(f"   ❌ Microphone device {device_index} failed: {e}")
        
        if fixes_applied:
            print(f"\n✅ Quick fixes applied: {', '.join(fixes_applied)}")
        else:
            print("\n❌ No automatic fixes could be applied")
            print("💡 Run diagnostics ('d' key) for manual troubleshooting steps")
        
        return fixes_applied

    def log_detailed_error(self, error, context=""):
        """Log detailed error information for troubleshooting"""
        import traceback
        import platform
        import psutil
        
        error_info = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'system_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                'memory_available': f"{psutil.virtual_memory().available / (1024**3):.1f} GB"
            },
            'app_state': {
                'ollama_connected': self.check_ollama_status(),
                'microphone_available': self.microphone is not None,
                'webcam_available': self.cap is not None and self.cap.isOpened(),
                'mediapipe_enabled': self.use_mediapipe,
                'dataset_loaded': not self.sign_dataset.empty
            },
            'traceback': traceback.format_exc()
        }
        
        # Log to file
        with open('error_log.json', 'a') as f:
            import json
            json.dump(error_info, f, indent=2)
            f.write('\n---\n')
        
        # Print user-friendly error message
        print(f"\n❌ Error occurred: {error}")
        print(f"📝 Context: {context}")
        print(f"📋 Error details logged to 'error_log.json'")
        print(f"💡 Use 'd' key to run diagnostics or 'f' for quick fixes")
        
        return error_info
    
    def get_error_summary(self):
        """Get a summary of recent errors from the log file"""
        try:
            with open('error_log.json', 'r') as f:
                content = f.read()
                if not content.strip():
                    return "No errors logged"
                
                # Count error types
                error_counts = {}
                lines = content.split('\n---\n')
                for line in lines:
                    if line.strip():
                        try:
                            import json
                            error_data = json.loads(line)
                            error_type = error_data.get('error_type', 'Unknown')
                            error_counts[error_type] = error_counts.get(error_type, 0) + 1
                        except:
                            continue
                
                if error_counts:
                    summary = "Recent errors:\n"
                    for error_type, count in error_counts.items():
                        summary += f"   • {error_type}: {count} occurrences\n"
                    return summary
                else:
                    return "No errors logged"
        except FileNotFoundError:
            return "No error log file found"
        except Exception as e:
            return f"Error reading log file: {e}"

    def diagnose_webcam_issues(self):
        """Comprehensive webcam diagnostics"""
        print("\n📹 Webcam Diagnostics:")
        print("=" * 30)
        
        issues = []
        working_configs = []
        
        # Test different backends and camera indices
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto-detect")
        ]
        
        for backend_code, backend_name in backends:
            print(f"\n🔍 Testing {backend_name} backend...")
            for index in range(4):  # Test indices 0-3
                try:
                    cap = cv2.VideoCapture(index, backend_code)
                    if cap.isOpened():
                        # Test frame reading
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            # Get camera properties
                            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            print(f"   ✅ Camera {index}: {int(width)}x{int(height)} @ {fps:.1f}fps")
                            working_configs.append({
                                'index': index,
                                'backend': backend_name,
                                'backend_code': backend_code,
                                'resolution': f"{int(width)}x{int(height)}",
                                'fps': fps
                            })
                        else:
                            print(f"   ⚠️ Camera {index}: Opened but can't read frames")
                            issues.append(f"Camera {index} ({backend_name}): Can't read frames")
                    else:
                        print(f"   ❌ Camera {index}: Not accessible")
                except Exception as e:
                    print(f"   ❌ Camera {index}: Error - {e}")
                    issues.append(f"Camera {index} ({backend_name}): {e}")
                finally:
                    if 'cap' in locals() and cap:
                        cap.release()
        
        # Summary
        print(f"\n📋 Summary:")
        if working_configs:
            print(f"   ✅ Working configurations: {len(working_configs)}")
            for config in working_configs:
                print(f"      • Index {config['index']} ({config['backend']}): {config['resolution']} @ {config['fps']:.1f}fps")
        else:
            print("   ❌ No working camera configurations found")
        
        if issues:
            print(f"   ❌ Issues found: {len(issues)}")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"      • {issue}")
            if len(issues) > 5:
                print(f"      • ... and {len(issues) - 5} more issues")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if not working_configs:
            print("   • Check camera permissions in Windows Settings")
            print("   • Ensure camera is not being used by another application")
            print("   • Try updating camera drivers")
            print("   • Test camera in Windows Camera app")
            print("   • Restart the application")
        else:
            print("   • Use the first working configuration listed above")
            print("   • If performance is poor, try a different backend")
        
        return working_configs, issues

def startup_check():
    """Validate all components before starting the main application"""
    print("🔍 Running startup validation...")
    
    issues = []
    
    # Check imports
    try:
        import cv2
        import numpy as np
        import requests
        import speech_recognition as sr
        import pygame
        import pandas as pd
        import aiohttp
        from gtts import gTTS
        print("   ✅ All required packages imported")
    except ImportError as e:
        issues.append(f"Missing package: {e}")
        print(f"   ❌ Import error: {e}")
    
    # Check webcam
    try:
        print("   🔍 Testing webcam...")
        
        # Try a single camera configuration to avoid conflicts
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_MSMF)  # Try Media Foundation first
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print("   ✅ Webcam validated (Index 0, Media Foundation)")
                    webcam_working = True
                else:
                    print("   ⚠️ Camera opened but can't read frames")
                    webcam_working = False
            else:
                print("   ⚠️ Camera not accessible with Media Foundation")
                webcam_working = False
        except Exception as e:
            print(f"   ⚠️ Media Foundation failed: {e}")
            webcam_working = False
        finally:
            if cap:
                cap.release()
        
        # If Media Foundation failed, try DirectShow
        if not webcam_working:
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print("   ✅ Webcam validated (Index 0, DirectShow)")
                        webcam_working = True
                    else:
                        print("   ⚠️ Camera opened but can't read frames")
                else:
                    print("   ⚠️ Camera not accessible with DirectShow")
            except Exception as e:
                print(f"   ⚠️ DirectShow failed: {e}")
            finally:
                if cap:
                    cap.release()
        
        if not webcam_working:
            issues.append("Webcam can't read frames from any camera")
            print("   ❌ Webcam can't read frames from any camera")
            
    except Exception as e:
        issues.append(f"Webcam error: {e}")
        print(f"   ❌ Webcam error: {e}")
    
    # Check microphone
    try:
        mic = sr.Microphone()
        with mic as source:
            print("   ✅ Microphone validated")
    except Exception as e:
        issues.append(f"Microphone error: {e}")
        print(f"   ❌ Microphone error: {e}")
    
    # Check dataset
    try:
        df = pd.read_csv('gpt-3.5-cleaned.csv')
        if not df.empty:
            print(f"   ✅ Dataset validated: {len(df)} entries")
        else:
            issues.append("Dataset is empty")
            print("   ❌ Dataset is empty")
    except Exception as e:
        issues.append(f"Dataset error: {e}")
        print(f"   ❌ Dataset error: {e}")
    
    # Check Ollama (optional but recommended)
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("   ✅ Ollama server validated")
        else:
            print(f"   ⚠️ Ollama server error: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Ollama not accessible: {e}")
    
    if issues:
        print(f"\n❌ Found {len(issues)} critical issues:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Fix these issues before running the application.")
        print("💡 Run 'python debug_silent_echo.py' for detailed diagnostics.")
        return False
    else:
        print("   ✅ All components validated successfully!")
        return True

def main():
    print("🤟 Silent Echo - AI Communication Assistant")
    print("=" * 50)
    print("Enhanced Version with MediaPipe Support")
    print("=" * 50)
    
    # Check for webcam-only mode
    webcam_only = "--webcam-only" in sys.argv
    
    if webcam_only:
        print("🔧 Webcam-only mode enabled")
        print("This mode will only test webcam functionality")
    
    # Run startup validation
    if not startup_check():
        print("\n🛑 Startup validation failed. Please fix the issues above.")
        input("Press Enter to exit...")
        return
    
    # Initialize the app
    se = SilentEchoOllama()
    
    # Run initial diagnostics
    print("\n🔍 Running startup diagnostics...")
    issues, warnings = se.diagnose_system_issues()
    
    if issues:
        print(f"\n⚠️ Found {len(issues)} critical issues:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n💡 Recommendations:")
        if "Ollama server not accessible" in str(issues):
            print("   • Start Ollama: ollama serve")
            print("   • Download model: ollama pull granite3.3:8b")
        if "No microphone found" in str(issues):
            print("   • Check microphone permissions")
            print("   • Try different microphone device")
        if "Webcam not accessible" in str(issues):
            print("   • Check camera permissions")
            print("   • Try different camera index")
        print("\n🔄 Press 'f' to attempt automatic fixes")
        print("📋 Press 'd' to run detailed diagnostics")
    
    if warnings:
        print(f"\n⚠️ Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not issues and not warnings:
        print("✅ All systems operational!")
    
    # Check Ollama status
    ollama_status = se.check_ollama_status()
    print(f"🤖 Ollama: {'✅ Connected' if ollama_status else '❌ Not Connected'}")
    
    # Display system info
    print(f"📊 Dataset: {len(se.sign_dataset)} examples loaded")
    print(f"🎯 Gesture Detection: {'MediaPipe' if se.use_mediapipe else 'OpenCV'}")
    print(f"🎤 Microphone: {'✅ Available' if se.microphone else '❌ Not Available'}")
    
    # Start continuous listening if microphone is available and not in webcam-only mode
    if se.microphone and not webcam_only:
        print("🎙️ Starting continuous listening...")
        se.start_continuous_listening()
    
    # Start webcam
    if not se.start_webcam():
        print("❌ Could not open webcam.")
        return
    
    print("📹 Webcam started successfully")
    
    if webcam_only:
        print("\n🎮 Webcam-only mode controls:")
        print("- Press 'q' or 'ESC' to quit")
        print("- Watch for webcam feed and any errors")
    else:
        print("\n🎮 Controls:")
        print("- Press 's' to speak manually")
        print("- Press 'p' to show performance stats")
        print("- Press 'd' to run system diagnostics")
        print("- Press 'w' to run webcam diagnostics")
        print("- Press 'f' to attempt quick fixes")
        print("- Press 'e' to show error summary")
        print("- Press 'h' for help")
        print("- Press 'q' or 'ESC' to quit")
        print("\n🚀 Starting detection...")
        print("- Make hand gestures in front of the camera")
        print("- Speak naturally for continuous listening")
        print("- Watch for real-time feedback")
    
    # Performance monitoring
    last_stats_time = time.time()
    stats_interval = 30
    
    try:
        while True:
            try:
                # Check if webcam is still working
                if not se.cap or not se.cap.isOpened():
                    logger.error("Webcam connection lost, attempting to reconnect...")
                    if not se.start_webcam():
                        logger.error("Failed to reconnect webcam")
                        break
                
                # Process video
                ret, frame = se.cap.read()
                if not ret or frame is None or frame.size == 0:
                    logger.warning("Failed to grab frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                if webcam_only:
                    # Webcam-only mode: just display the frame
                    cv2.imshow("Silent Echo - Webcam Test", frame)
                else:
                    # Full mode: process frame for sign language detection
                    processed_frame, gesture = se.detect_sign_language(frame)
                    
                    # Handle sign language detection
                    if se.sign_language_detected and gesture:
                        if se.should_generate_response(gesture):
                            interpreted = se.interpret_gesture_for_communication(gesture)
                            print(f"🤟 Detected: {gesture} -> 💬 Spoken: {interpreted}")
                            se.text_to_speech(interpreted)
                    
                    # Process audio queue
                    audio_result = se.process_audio_queue()
                    if audio_result:
                        se.stats['audio_processed'] += 1
                        print(f"🎤 You said: {audio_result['text']}")
                        # Get AI response
                        response = se.get_ollama_response_sync(audio_result['text'])
                        print(f"🤖 AI Response: {response}")
                        se.text_to_speech(response)
                    
                    # Display the processed frame
                    cv2.imshow("Silent Echo - Sign Language Detection", processed_frame)
                
            except cv2.error as e:
                logger.error(f"OpenCV error: {e}")
                se.log_detailed_error(e, "OpenCV operation")
                time.sleep(0.5)  # Wait before retrying
                continue
            except Exception as e:
                se.log_detailed_error(e, "Main processing loop")
                logger.error(f"Main loop error: {e}")
                time.sleep(0.5)  # Wait before retrying
                continue
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('q'):
                break
            elif not webcam_only:  # Only show full controls in normal mode
                if key == ord('s'):
                    # Manual speech input
                    try:
                        print("🎤 Manual speech mode - speak now...")
                        audio = se.listen_for_speech()
                        if audio:
                            text = se.speech_to_text(audio)
                            if text:
                                se.stats['audio_processed'] += 1
                                print(f"🎤 You said: {text}")
                                response = se.get_ollama_response_sync(text)
                                print(f"🤖 AI Response: {response}")
                                se.text_to_speech(response)
                    except Exception as e:
                        se.log_detailed_error(e, "Manual speech input")
                elif key == ord('p'):
                    # Show performance stats
                    try:
                        stats = se.get_performance_stats()
                        print("\n📊 Performance Statistics:")
                        print(f"   Runtime: {stats['runtime_seconds']:.1f} seconds")
                        print(f"   FPS: {stats['frames_per_second']:.1f}")
                        print(f"   Gestures/min: {stats['gestures_per_minute']:.1f}")
                        print(f"   Audio processed: {stats['audio_processed']}")
                        print(f"   AI responses: {stats['ai_responses']}")
                        print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
                    except Exception as e:
                        se.log_detailed_error(e, "Performance stats display")
                elif key == ord('d'):
                    # Run system diagnostics
                    try:
                        issues, warnings = se.diagnose_system_issues()
                        if issues or warnings:
                            print("\n📋 System Diagnostics Results:")
                            if issues:
                                print(f"   Critical issues: {len(issues)}")
                                for issue in issues:
                                    print(f"      - {issue}")
                            if warnings:
                                print(f"   Warnings: {len(warnings)}")
                                for warning in warnings:
                                    print(f"      - {warning}")
                    except Exception as e:
                        se.log_detailed_error(e, "System diagnostics")
                elif key == ord('w'):
                    # Run webcam diagnostics
                    try:
                        working_configs, issues = se.diagnose_webcam_issues()
                        if working_configs:
                            print(f"\n Found {len(working_configs)} working webcam configurations")
                        if issues:
                            print(f"\n Found {len(issues)} webcam issues")
                    except Exception as e:
                        se.log_detailed_error(e, "Webcam diagnostics")
                elif key == ord('f'):
                    # Attempt quick fixes
                    try:
                        se.quick_fix_common_issues()
                    except Exception as e:
                        se.log_detailed_error(e, "Quick fixes")
                elif key == ord('e'):
                    # Show error summary
                    try:
                        print("\n📋 Error Summary:")
                        print(se.get_error_summary())
                    except Exception as e:
                        se.log_detailed_error(e, "Error summary")
                elif key == ord('h'):
                    print("\n📖 Help:")
                    print("- Press 's' to speak manually")
                    print("- Press 'p' to show performance stats")
                    print("- Press 'd' to run system diagnostics")
                    print("- Press 'w' to run webcam diagnostics")
                    print("- Press 'f' to attempt quick fixes")
                    print("- Press 'e' to show error summary")
                    print("- Press 'h' for help")
                    print("- Press 'q' or 'ESC' to quit")
            
            # Periodic stats display (only in full mode)
            if not webcam_only:
                current_time = time.time()
                if current_time - last_stats_time > stats_interval:
                    try:
                        stats = se.get_performance_stats()
                        print(f"\n📈 Runtime: {stats['runtime_seconds']:.0f}s | FPS: {stats['frames_per_second']:.1f} | Gestures: {stats['gestures_detected']}")
                    except KeyError as e:
                        logger.error(f"Missing stats key: {e}")
                        print(f"\n📈 Runtime: {current_time - se.stats.get('start_time', current_time):.0f}s | FPS: {se.stats.get('frames_processed', 0) / max(1, current_time - se.stats.get('start_time', current_time)):.1f}")
                    except Exception as e:
                        logger.error(f"Error displaying stats: {e}")
                    last_stats_time = current_time
            
            # Small delay to prevent high CPU usage
            time.sleep(1.0 / config.frame_rate)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    finally:
        print("🧹 Cleaning up resources...")
        
        # Final stats
        try:
            stats = se.get_performance_stats()
            print(f"\n📊 Final Statistics:")
            print(f"   Total runtime: {stats['runtime_seconds']:.1f} seconds")
            print(f"   Average FPS: {stats['frames_per_second']:.1f}")
            print(f"   Total gestures detected: {stats['gestures_detected']}")
            print(f"   Total audio processed: {stats['audio_processed']}")
            print(f"   Total AI responses: {stats['ai_responses']}")
        except Exception as e:
            logger.error(f"Error displaying final stats: {e}")
            print(f"\n📊 Final Statistics (partial):")
            print(f"   Runtime: {time.time() - se.stats.get('start_time', time.time()):.1f} seconds")
            print(f"   Gestures: {se.stats.get('gestures_detected', 0)}")
            print(f"   Audio: {se.stats.get('audio_processed', 0)}")
            print(f"   AI Responses: {se.stats.get('ai_responses', 0)}")
        
        se.cleanup()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete. Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"\n❌ Application crashed with error: {e}")
        print("📋 Full traceback:")
        import traceback
        traceback.print_exc()
        
        # Save error to log file
        try:
            with open('crash_log.txt', 'a') as f:
                f.write(f"\n--- Crash at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(f"Error: {e}\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
        except:
            pass
        
        print(f"\n💡 Error details saved to 'crash_log.txt'")
        print("💡 Run 'python debug_silent_echo.py' for diagnostics")
        print("💡 Check the troubleshooting guide in README.md")
    
    print("\n👋 Application ended.")
    input("Press Enter to exit...")
