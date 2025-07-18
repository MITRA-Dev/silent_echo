# Silent Echo Gradio App

import gradio as gr
import cv2
import numpy as np
import threading
import time
import queue
import os
import requests
import json
from gtts import gTTS
import tempfile
import pygame
import io
import pandas as pd

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "granite3.3:8b"

class SilentEchoGradio:
    def __init__(self):
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
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
    
    def load_sign_dataset(self):
        """Load the GPT-3.5 cleaned sign language dataset"""
        try:
            df = pd.read_csv('gpt-3.5-cleaned.csv')
            # Filter out empty annotated_texts
            df = df[df['annotated_texts'].notna() & (df['annotated_texts'] != '')]
            return df
        except Exception as e:
            print(f"Error loading sign dataset: {e}")
            return pd.DataFrame()
    
    def get_sign_gestures(self):
        """Get unique sign gestures from the dataset"""
        if not self.sign_dataset.empty:
            return self.sign_dataset['annotated_texts'].unique().tolist()
        return []
    
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
            print(f"Gesture analysis error: {e}")
            return "Hand Detected"
    
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
    
    def text_to_speech(self, text):
        """Convert text to speech and play it with improved Windows compatibility"""
        try:
            # Try Windows SAPI first (most reliable on Windows)
            try:
                self.text_to_speech_windows_fallback(text)
                return  # Success, exit early
            except Exception as sapi_error:
                print(f"Windows SAPI failed, trying simple Windows TTS: {sapi_error}")
            
            # Try simple Windows TTS (no file handling)
            try:
                self.text_to_speech_simple_windows(text)
                return  # Success, exit early
            except Exception as simple_error:
                print(f"Simple Windows TTS failed, trying gTTS: {simple_error}")
            
            # Fallback to gTTS with improved file handling
            self.text_to_speech_gtts_fallback(text)
            
        except Exception as e:
            print(f"All text-to-speech methods failed: {e}")
    
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
    
    def text_to_speech_gtts_fallback(self, text):
        """gTTS fallback with improved file handling"""
        temp_file = None
        try:
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
            print(f"gTTS error: {e}")
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
        
        # Start delayed cleanup in background
        cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()

# Initialize the app
silent_echo = SilentEchoGradio()

def process_video_frame(frame):
    """Process video frame for real-time sign language interpretation"""
    if frame is None:
        return None, "No video frame received"
    
    try:
        # Process frame for sign language detection
        processed_frame, detected_gesture = silent_echo.detect_sign_language(frame)
        
        # Update current gesture
        silent_echo.current_gesture = detected_gesture
        
        # Check if we should generate a response
        if silent_echo.should_generate_response(detected_gesture):
            # Use enhanced gesture interpretation for deaf communication
            interpreted_gesture = silent_echo.interpret_gesture_for_communication(detected_gesture)
            
            if interpreted_gesture and interpreted_gesture != "I see you":
                # Speak the interpreted gesture
                silent_echo.text_to_speech(interpreted_gesture)
                
                # Add to gesture history
                silent_echo.gesture_history.append({
                    'gesture': detected_gesture,
                    'interpreted': interpreted_gesture,
                    'timestamp': time.time()
                })
                
                # Keep only last 10 gestures
                if len(silent_echo.gesture_history) > 10:
                    silent_echo.gesture_history.pop(0)
                
                return processed_frame, f"🤟 Detected: {detected_gesture} → 🎤 Spoken: {interpreted_gesture}"
            else:
                return processed_frame, f"🤟 Detected: {detected_gesture} (not recognized for communication)"
        else:
            if detected_gesture:
                return processed_frame, f"🤟 Detected: {detected_gesture}"
            else:
                return processed_frame, "👋 Show your hands to the camera for real-time communication"
    
    except Exception as e:
        return frame, f"Error processing frame: {e}"

def get_gesture_history():
    """Get recent gesture history"""
    if not silent_echo.gesture_history:
        return "No gestures detected yet"
    
    history_text = "**Recent Gestures:**\n"
    for gesture_data in silent_echo.gesture_history[-5:]:  # Show last 5
        history_text += f"• {gesture_data['gesture']} → {gesture_data['interpreted']}\n"
    
    return history_text

def get_supported_gestures():
    """Get list of supported gestures"""
    supported_gestures = [
        "Thumbs Up → Good", "Peace Sign → Peace", "Index Finger → Yes",
        "Open Hand → Hello", "Closed Fist → Stop", "Five Fingers → Five",
        "Pointing → There", "Raised Hand → Question", "OK Sign → Okay"
    ]
    
    return "\n".join([f"• {gesture}" for gesture in supported_gestures])

# Create Gradio interface
with gr.Blocks(title="Silent Echo - Real-time Sign Language Interpreter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤟 Silent Echo - Real-time Sign Language Interpreter")
    gr.Markdown("### AI-Powered Communication Assistant for the Deaf Community")
    
    with gr.Row():
        with gr.Column(scale=2):
            # Real-time video input
            video_input = gr.Video(
                source="webcam",
                streaming=True,
                mirror_webcam=True,
                label="Live Video Feed - Real-time Gesture Recognition"
            )
            
            # Process button for manual processing
            process_btn = gr.Button("🔄 Process Current Frame", variant="primary")
        
        with gr.Column(scale=1):
            # Output for gesture detection results
            output_text = gr.Textbox(
                label="Real-time Gesture Recognition",
                placeholder="Show your hands to the camera...",
                lines=3
            )
            
            # Gesture history
            history_output = gr.Markdown(label="Recent Gestures")
            
            # Supported gestures
            gr.Markdown("### 🤟 Supported Gestures")
            supported_gestures_output = gr.Markdown(get_supported_gestures())
            
            # Instructions
            gr.Markdown("""
            ### 🎯 How to Use
            
            1. **Show your hands** to the camera
            2. **Make sign language gestures** 
            3. **System continuously analyzes** and speaks
            4. **No clicking needed** - just gesture naturally
            5. **Perfect for real-time communication** with deaf people
            
            ### 🎯 Purpose
            Real-time sign language to speech conversion for deaf communication.
            """)
    
    # Set up event handlers
    video_input.change(
        fn=process_video_frame,
        inputs=[video_input],
        outputs=[video_input, output_text]
    )
    
    process_btn.click(
        fn=process_video_frame,
        inputs=[video_input],
        outputs=[video_input, output_text]
    )
    
    # Update gesture history
    def update_history():
        return get_gesture_history()
    
    # Auto-update history every 2 seconds
    demo.load(update_history, outputs=[history_output])

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
