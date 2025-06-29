#!/usr/bin/env python3
"""
Silent Echo Launcher
A helper script to check prerequisites and launch the Silent Echo application.
"""

import subprocess
import sys
import os
import requests
import time

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'speech_recognition', 'cv2', 'numpy', 
        'requests', 'gtts', 'pygame', 'PIL'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except requests.exceptions.RequestException:
        pass
    
    print("❌ Ollama is not running or not accessible")
    print("Please start Ollama with: ollama serve")
    return False

def check_granite_model():
    """Check if Granite model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            granite_models = [m for m in models if 'granite' in m.get('name', '').lower()]
            if granite_models:
                print(f"✅ Granite model found: {granite_models[0]['name']}")
                return True
            else:
                print("❌ Granite model not found")
                print("Please pull the model with: ollama pull granite:3.3-8b")
                return False
    except requests.exceptions.RequestException:
        print("❌ Cannot check for Granite model - Ollama not accessible")
        return False

def install_dependencies():
    """Install missing dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def start_ollama():
    """Start Ollama service"""
    print("\n🚀 Starting Ollama service...")
    try:
        # Start Ollama in background
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for Ollama to start
        for i in range(10):
            time.sleep(1)
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✅ Ollama started successfully")
                    return True
            except requests.exceptions.RequestException:
                continue
        
        print("❌ Ollama failed to start")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first.")
        print("Visit: https://ollama.ai for installation instructions")
        return False

def pull_granite_model():
    """Pull the Granite model"""
    print("\n📥 Pulling Granite model...")
    try:
        subprocess.check_call(["ollama", "pull", "granite:3.3-8b"])
        print("✅ Granite model pulled successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to pull Granite model")
        return False

def launch_app():
    """Launch the Silent Echo application"""
    print("\n🚀 Launching Silent Echo...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "silent_echo_ollama.py"])
    except KeyboardInterrupt:
        print("\n👋 Silent Echo stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")

def main():
    """Main launcher function"""
    print("🤟 Silent Echo - AI Communication Assistant")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        response = input("\nWould you like to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Check Ollama
    print("\n🤖 Checking Ollama...")
    if not check_ollama():
        response = input("Would you like to start Ollama? (y/n): ")
        if response.lower() == 'y':
            if not start_ollama():
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Check Granite model
    print("\n🧠 Checking Granite model...")
    if not check_granite_model():
        response = input("Would you like to pull the Granite model? (y/n): ")
        if response.lower() == 'y':
            if not pull_granite_model():
                sys.exit(1)
        else:
            sys.exit(1)
    
    # All checks passed
    print("\n✅ All prerequisites met!")
    print("🎉 Ready to launch Silent Echo")
    
    # Launch the application
    launch_app()

if __name__ == "__main__":
    main() 