#!/usr/bin/env python3
"""
Silent Echo Test Suite
Run this script to test your Silent Echo setup and identify any issues.
"""

import sys
import time
import traceback
from pathlib import Path

def test_imports():
    """Test if all required packages are available"""
    print("🔍 Testing imports...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'requests': 'requests',
        'speech_recognition': 'SpeechRecognition',
        'pygame': 'pygame',
        'pandas': 'pandas',
        'aiohttp': 'aiohttp',
        'gtts': 'gTTS'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (install: pip install {pip_name})")
            missing_packages.append(pip_name)
    
    # Test optional packages
    try:
        import mediapipe
        print("   ✅ mediapipe (optional)")
    except ImportError:
        print("   ⚠️ mediapipe (optional - install: pip install mediapipe)")
    
    try:
        import psutil
        print("   ✅ psutil (optional)")
    except ImportError:
        print("   ⚠️ psutil (optional - install: pip install psutil)")
    
    return missing_packages

def test_ollama():
    """Test Ollama connection and models"""
    print("\n🤖 Testing Ollama...")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            print(f"   ✅ Ollama server running")
            print(f"   📋 Available models: {', '.join(models[:3])}")
            
            # Check for recommended model
            if any('granite' in model.lower() for model in models):
                print("   ✅ Recommended model found")
            else:
                print("   ⚠️ Recommended model not found")
                print("   💡 Run: ollama pull granite3.3:8b")
            
            return True, models
        else:
            print(f"   ❌ Ollama server error: {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to Ollama server")
        print("   💡 Start Ollama: ollama serve")
        return False, []
    except Exception as e:
        print(f"   ❌ Ollama test failed: {e}")
        return False, []

def test_microphone():
    """Test microphone availability"""
    print("\n🎤 Testing microphone...")
    
    try:
        import speech_recognition as sr
        mic = sr.Microphone()
        with mic as source:
            print("   ✅ Microphone available")
            return True
    except Exception as e:
        print(f"   ❌ Microphone test failed: {e}")
        print("   💡 Check microphone permissions and connections")
        return False

def test_webcam():
    """Test webcam availability"""
    print("\n📹 Testing webcam...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("   ✅ Webcam working")
                cap.release()
                return True
            else:
                print("   ❌ Cannot read frames from webcam")
                cap.release()
                return False
        else:
            print("   ❌ Cannot open webcam")
            return False
    except Exception as e:
        print(f"   ❌ Webcam test failed: {e}")
        return False

def test_dataset():
    """Test if sign language dataset is available"""
    print("\n📊 Testing dataset...")
    
    dataset_file = Path("gpt-3.5-cleaned.csv")
    if dataset_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(dataset_file)
            print(f"   ✅ Dataset loaded: {len(df)} entries")
            return True
        except Exception as e:
            print(f"   ❌ Dataset loading failed: {e}")
            return False
    else:
        print("   ❌ Dataset file not found: gpt-3.5-cleaned.csv")
        return False

def test_audio_playback():
    """Test audio playback capabilities"""
    print("\n🔊 Testing audio playback...")
    
    try:
        import pygame
        pygame.mixer.init()
        print("   ✅ Pygame audio initialized")
        
        # Test gTTS
        try:
            from gtts import gTTS
            print("   ✅ gTTS available")
            return True
        except Exception as e:
            print(f"   ❌ gTTS test failed: {e}")
            return False
    except Exception as e:
        print(f"   ❌ Audio playback test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🤟 Silent Echo - Comprehensive Test Suite")
    print("=" * 50)
    
    results = {}
    
    # Test imports
    missing_packages = test_imports()
    results['imports'] = len(missing_packages) == 0
    
    # Test Ollama
    ollama_ok, models = test_ollama()
    results['ollama'] = ollama_ok
    
    # Test microphone
    mic_ok = test_microphone()
    results['microphone'] = mic_ok
    
    # Test webcam
    webcam_ok = test_webcam()
    results['webcam'] = webcam_ok
    
    # Test dataset
    dataset_ok = test_dataset()
    results['dataset'] = dataset_ok
    
    # Test audio
    audio_ok = test_audio_playback()
    results['audio'] = audio_ok
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 20)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"   {status} {test}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Silent Echo should work correctly.")
    else:
        print("\n🔧 Issues found. Here are the recommended fixes:")
        
        if not results['imports']:
            print("\n📦 Install missing packages:")
            for package in missing_packages:
                print(f"   pip install {package}")
        
        if not results['ollama']:
            print("\n🤖 Ollama setup:")
            print("   1. Install Ollama: https://ollama.ai")
            print("   2. Start server: ollama serve")
            print("   3. Download model: ollama pull granite3.3:8b")
        
        if not results['microphone']:
            print("\n🎤 Microphone issues:")
            print("   1. Check Windows microphone permissions")
            print("   2. Test microphone in Windows Sound settings")
            print("   3. Try different microphone device")
        
        if not results['webcam']:
            print("\n📹 Webcam issues:")
            print("   1. Check Windows camera permissions")
            print("   2. Test camera in Windows Camera app")
            print("   3. Try different camera index")
        
        if not results['dataset']:
            print("\n📊 Dataset issues:")
            print("   1. Ensure 'gpt-3.5-cleaned.csv' is in the current directory")
            print("   2. Check file permissions")
        
        if not results['audio']:
            print("\n🔊 Audio issues:")
            print("   1. Check audio drivers")
            print("   2. Test speakers/headphones")
            print("   3. Install audio dependencies")
    
    print(f"\n💡 Run the main application: python sign_lan2.py")
    print("💡 Use 'd' key for diagnostics or 'f' for quick fixes")

if __name__ == "__main__":
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        print("Traceback:")
        traceback.print_exc() 