import streamlit as st
import requests
import io
import os
import time
from datetime import datetime
from streamlit_mic_recorder import mic_recorder

# Page configuration
st.set_page_config(page_title="Silent Echo", page_icon="🎙️", layout="wide")

# Title and logo
st.title("Silent Echo: Text-to-Speech and Speech-to-Sign")
# Uncomment and add logo.png to silent_echo_frontend/
# st.image("silent_echo_frontend/logo.png", width=200)

# Backend API endpoints (replace with actual URLs from your backend team)
BACKEND_URL = "http://localhost:8000"  # Update with your backend's URL
TTS_ENDPOINT = f"{BACKEND_URL}/tts"
STT_ENDPOINT = f"{BACKEND_URL}/stt"
SIGN_ENDPOINT = f"{BACKEND_URL}/sign"

# Mock mode for testing without backend
MOCK_MODE = True  # Set to False when backend is ready

# Directory for saving recordings
RECORDINGS_DIR = "silent_echo_frontend/recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Layout with columns
col1, col2 = st.columns([1, 1])

# Text to Speech
with col1:
    st.header("Text to Speech")
    st.markdown("Enter text below to convert it to speech.")
    text_input = st.text_area("Text input:", height=100, key="tts_input")
    if st.button("Generate Speech", key="tts_button", use_container_width=True):
        if text_input:
            with st.spinner("Processing text..."):
                try:
                    if MOCK_MODE:
                        audio_file = "silent_echo_frontend/mock_audio.mp3"
                        if os.path.exists(audio_file):
                            st.audio(audio_file, format="audio/mp3")
                            st.success("Mock audio generated!")
                        else:
                            st.error("Mock audio file not found. Add mock_audio.mp3 to silent_echo_frontend/")
                    else:
                        response = requests.post(TTS_ENDPOINT, json={"text": text_input}, timeout=10)
                        response.raise_for_status()
                        audio_file = "temp_audio.mp3"
                        with open(audio_file, "wb") as f:
                            f.write(response.content)
                        st.audio(audio_file, format="audio/mp3")
                        st.success("Audio generated!")
                        os.remove(audio_file)
                except requests.RequestException as e:
                    st.error(f"Backend error: {str(e)}. Ensure backend is running at {TTS_ENDPOINT}.")
                except Exception as e:
                    st.error(f"Error: {str(e)}. Check file paths or backend response.")
        else:
            st.warning("Please enter some text.")

# Speech to Text with Audio Recording
with col2:
    st.header("Speech to Text")
    st.markdown("Click below to record audio. Ensure microphone access is enabled in Chrome/Firefox.")
    try:
        audio_data = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            key="recorder"
        )
    except Exception as e:
        st.error(f"Failed to load mic_recorder: {str(e)}. Run 'pip install streamlit-mic-recorder'.")
        audio_data = None
transcribed_text = None
if audio_data and audio_data.get("bytes"):
    st.write("Debug: Audio data received from mic_recorder.")
    # Save recorded audio
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")
        with open(audio_file, "wb") as f:
            f.write(audio_data["bytes"])
        st.success(f"Audio saved as {audio_file}")
        st.audio(audio_data["bytes"], format="audio/wav")
    except Exception as e:
        st.error(f"Failed to save audio: {str(e)}")
    with st.spinner("Processing audio..."):
        try:
            if MOCK_MODE:
                transcribed_text = "Mock transcribed text: Hello world"
                st.write("Transcribed text:", transcribed_text)
                st.success("Mock transcription complete!")
            else:
                audio_bytes = audio_data["bytes"]
                files = {"audio": ("recording.wav", io.BytesIO(audio_bytes), "audio/wav")}
                response = requests.post(STT_ENDPOINT, files=files, timeout=10)
                response.raise_for_status()
                transcribed_text = response.json().get("text", "")
                if transcribed_text:
                    st.write("Transcribed text:", transcribed_text)
                    st.success("Transcription complete!")
                else:
                    st.warning("No transcription returned from backend.")
        except requests.RequestException as e:
            st.error(f"Backend error: {str(e)}. Ensure backend is running at {STT_ENDPOINT}.")
        except Exception as e:
            st.error(f"Error: {str(e)}. Check browser console (F12) for issues.")
else:
    if audio_data is not None:
        st.warning("No audio recorded. Ensure microphone permissions are enabled.")

# Speech to Sign Language
st.header("Sign Language Output")
st.markdown("View sign language visuals for the transcribed text below.")
if transcribed_text:
    with st.spinner("Fetching sign language visuals..."):
        try:
            if MOCK_MODE:
                mock_signs = ["https://example.com/hello.mp4"]
                for item in mock_signs:
                    st.video(item)
                st.success("Mock sign language visuals loaded!")
            else:
                response = requests.post(SIGN_ENDPOINT, json={"text": transcribed_text}, timeout=10)
                response.raise_for_status()
                sign_data = response.json().get("signs", [])
                if sign_data:
                    for item in sign_data:
                        if item.endswith((".mp4", ".webm")):
                            st.video(item)
                        elif item.endswith((".png", ".jpg", ".jpeg")):
                            st.image(item, caption="Sign")
                        else:
                            st.warning(f"Unsupported file format: {item}")
                    st.success("Sign language visuals loaded!")
                else:
                    st.warning("No sign language visuals returned.")
        except requests.RequestException as e:
            st.error(f"Backend error: {str(e)}. Ensure backend is running at {SIGN_ENDPOINT}.")
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Record and transcribe audio first to see sign language.")

# Sidebar for settings
st.sidebar.title("Silent Echo Settings")
st.sidebar.markdown("Configure settings for the app.")
language = st.sidebar.selectbox("Language:", ["English"], help="Select language for transcription")