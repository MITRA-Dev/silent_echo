import speech_recognition as sr
import sys

recognizer = sr.Recognizer()
print("Available microphones:", sr.Microphone.list_microphone_names())
sys.stdout.flush()
try:
    with sr.Microphone() as source:
        print("Microphone initialized, press Ctrl+C to stop")
        while True:
            audio = recognizer.listen(source)
            print("Audio captured")
except Exception as e:
    print(f"Error: {e}")