from vosk import Model, KaldiRecognizer
import wave

wf = wave.open("audio_file.wav", "rb")  # Replace with your WAV file
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    exit(1)

model = Model("model")  # Path to the extracted model folder
recognizer = KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if recognizer.AcceptWaveform(data):
        print(recognizer.Result())
    else:
        print(recognizer.PartialResult())

print(recognizer.FinalResult())