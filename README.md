# 🤟 Silent Echo - AI Communication Assistant

Silent Echo is an AI-powered communication assistant designed for the deaf community, combining real-time sign language detection with speech recognition and AI responses.

## ✨ Features

- **Real-time Sign Language Detection**: Uses MediaPipe (with OpenCV fallback) for hand gesture recognition
- **Speech Recognition**: Continuous listening with Google Speech Recognition
- **AI Responses**: Powered by Ollama with intelligent fallback responses
- **Text-to-Speech**: Converts AI responses to spoken audio
- **Comprehensive Diagnostics**: Built-in troubleshooting and system diagnostics
- **Performance Monitoring**: Real-time stats and performance tracking

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
3. **Microphone** and **webcam** with proper permissions

### Installation

1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Ollama** (if not already installed):
   - Download from: https://ollama.ai
   - Start the server: `ollama serve`
   - Download the model: `ollama pull granite3.3:8b`

4. **Run the test suite** to verify your setup:
   ```bash
   python test_silent_echo.py
   ```

5. **Start Silent Echo**:
   ```bash
   python sign_lan2.py
   ```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `s` | Manual speech input |
| `p` | Show performance stats |
| `d` | Run system diagnostics |
| `f` | Attempt quick fixes |
| `e` | Show error summary |
| `h` | Show help |
| `q` or `ESC` | Quit |

## 🔧 Troubleshooting

### Common Issues

#### 🤖 Ollama Connection Issues
**Symptoms**: "Ollama connection issue" or "I'm experiencing technical difficulties"

**Solutions**:
1. Check if Ollama is installed: `ollama --version`
2. Start Ollama server: `ollama serve`
3. Verify server is running: `curl http://localhost:11434/api/tags`
4. Check if model is downloaded: `ollama list`
5. Download model if needed: `ollama pull granite3.3:8b`
6. Check firewall/antivirus settings
7. Try different port if 11434 is blocked

#### 🎤 Microphone Issues
**Symptoms**: "No microphone available" or no audio detection

**Solutions**:
1. Check microphone permissions in Windows Settings
2. Verify microphone is not muted
3. Test microphone in Windows Sound settings
4. Try different microphone device
5. Update audio drivers
6. Check if microphone is being used by another app
7. Restart the application

#### 📹 Webcam Issues
**Symptoms**: "Could not open webcam" or camera not working

**Solutions**:
1. Check camera permissions in Windows Settings
2. Verify camera is not being used by another app
3. Test camera in Windows Camera app
4. Update camera drivers
5. Try different camera index (0, 1, 2)
6. Check USB connection if external camera
7. Restart the application

#### ⚡ Performance Issues
**Symptoms**: Low FPS, laggy video, or slow responses

**Solutions**:
1. Reduce frame rate in config (frame_rate: 15)
2. Lower resolution by modifying webcam settings
3. Close other applications using camera/microphone
4. Check CPU and memory usage
5. Update graphics drivers
6. Try running as administrator
7. Restart computer if issues persist

### Diagnostic Tools

#### 1. Test Suite
Run the comprehensive test suite to identify issues:
```bash
python test_silent_echo.py
```

#### 2. Built-in Diagnostics
Use the `d` key in the application to run system diagnostics.

#### 3. Quick Fixes
Use the `f` key to attempt automatic fixes for common issues.

#### 4. Error Summary
Use the `e` key to view recent errors and their frequency.

### Error Logging

The application automatically logs detailed error information to `error_log.json`. This includes:
- Error type and message
- System information
- Application state
- Full traceback
- Timestamp

### Manual Troubleshooting Steps

1. **Check System Requirements**:
   - Python 3.8+
   - Sufficient RAM (4GB+ recommended)
   - Working microphone and webcam
   - Internet connection for speech recognition

2. **Verify Dependencies**:
   ```bash
   pip list | grep -E "(opencv|numpy|requests|speech|pygame|pandas|aiohttp|gtts)"
   ```

3. **Test Individual Components**:
   - Test microphone: `python -c "import speech_recognition as sr; print('Microphone OK')"`
   - Test camera: `python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Failed')"`
   - Test Ollama: `curl http://localhost:11434/api/tags`

4. **Check Permissions**:
   - Windows: Settings → Privacy → Camera/Microphone
   - Ensure Silent Echo has access to both

## 📊 Performance Optimization

### Frame Rate Adjustment
If experiencing low FPS, reduce the frame rate in the config:
```python
frame_rate: int = 15  # Default is 30
```

### Model Selection
For better performance, try a smaller model:
```bash
ollama pull llama2:7b
```

### Memory Management
- Close other applications while running Silent Echo
- Restart the application periodically for long sessions
- Monitor system resources with Task Manager

## 🔍 Advanced Configuration

### Configuration Options
Edit the `Config` class in `sign_lan2.py`:

```python
@dataclass
class Config:
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "granite3.3:8b"
    
    # Audio settings
    audio_timeout: float = 5.0
    phrase_time_limit: float = 10.0
    
    # Video settings
    min_contour_area: int = 5000
    gesture_cooldown: float = 2.0
    frame_rate: int = 30
    
    # AI settings
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 150
```

### Logging
Adjust log level for debugging:
```python
log_level: str = "DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR
```

## 📁 File Structure

```
silent_echo/
├── sign_lan2.py              # Main application
├── test_silent_echo.py       # Test suite
├── requirements.txt          # Dependencies
├── gpt-3.5-cleaned.csv      # Sign language dataset
├── silent_echo.log          # Application logs
├── error_log.json           # Error logs
└── README.md                # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

If you're still experiencing issues:

1. **Run the test suite**: `python test_silent_echo.py`
2. **Check error logs**: Look at `error_log.json` for detailed error information
3. **Use built-in diagnostics**: Press `d` in the application
4. **Try quick fixes**: Press `f` in the application
5. **Check system requirements**: Ensure all prerequisites are met

### Common Error Messages

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| "Ollama connection issue" | Ollama not running | Start with `ollama serve` |
| "No microphone available" | Permission/device issue | Check Windows settings |
| "Could not open webcam" | Camera in use/permissions | Close other apps, check permissions |
| "Low FPS" | Performance issue | Reduce frame rate, close other apps |
| "Model not found" | Model not downloaded | Run `ollama pull granite3.3:8b` |

---

**Happy communicating! 🤟**