#!/usr/bin/env python3
"""
Debug version of Silent Echo to identify why the application quits unexpectedly.
This version has extensive error handling and will show exactly where the problem occurs.
"""

import sys
import traceback
import time

def main():
    print("🔍 Silent Echo Debug Mode")
    print("=" * 40)
    
    try:
        print("1. Testing imports...")
        import cv2
        print("   ✅ OpenCV imported")
        
        import numpy as np
        print("   ✅ NumPy imported")
        
        import requests
        print("   ✅ Requests imported")
        
        import speech_recognition as sr
        print("   ✅ SpeechRecognition imported")
        
        import pygame
        print("   ✅ Pygame imported")
        
        import pandas as pd
        print("   ✅ Pandas imported")
        
        import aiohttp
        print("   ✅ aiohttp imported")
        
        from gtts import gTTS
        print("   ✅ gTTS imported")
        
        print("2. Testing basic components...")
        
        # Test webcam
        print("   Testing webcam...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("   ✅ Webcam working")
            else:
                print("   ❌ Webcam can't read frames")
            cap.release()
        else:
            print("   ❌ Webcam not accessible")
        
        # Test microphone
        print("   Testing microphone...")
        try:
            mic = sr.Microphone()
            with mic as source:
                print("   ✅ Microphone accessible")
        except Exception as e:
            print(f"   ❌ Microphone error: {e}")
        
        # Test Ollama
        print("   Testing Ollama...")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("   ✅ Ollama server running")
            else:
                print(f"   ❌ Ollama server error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Ollama connection failed: {e}")
        
        # Test dataset
        print("   Testing dataset...")
        try:
            df = pd.read_csv('gpt-3.5-cleaned.csv')
            print(f"   ✅ Dataset loaded: {len(df)} entries")
        except Exception as e:
            print(f"   ❌ Dataset error: {e}")
        
        print("3. All basic tests passed. Starting main application...")
        
        # Import and start the main application
        from sign_lan2 import main as main_app
        print("   ✅ Main application imported successfully")
        
        print("4. Starting Silent Echo...")
        main_app()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install missing package with: pip install <package_name>")
        print("💡 Or run: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("📋 Full traceback:")
        traceback.print_exc()
        
        # Additional debugging info
        print("\n🔍 System Information:")
        import platform
        print(f"   Platform: {platform.platform()}")
        print(f"   Python: {platform.python_version()}")
        
        try:
            import psutil
            print(f"   Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB available")
            print(f"   CPU: {psutil.cpu_count()} cores")
        except:
            print("   Memory/CPU info not available")
    
    print("\n🔍 Debug session completed.")
    print("💡 If you see errors above, fix them before running the main application.")
    print("💡 If no errors, the issue might be in the main loop - check error_log.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Debug interrupted by user")
    except Exception as e:
        print(f"\n❌ Debug script error: {e}")
        traceback.print_exc()
    
    input("\nPress Enter to exit...") 