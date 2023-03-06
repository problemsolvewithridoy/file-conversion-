# audio file to text file
import speech_recognition as sr

filename = "audio.wav"
# initialize the recognizer
recognizer = sr.Recognizer()

# read the audio file
with sr.AudioFile(filename) as source:
    audio = recognizer.record(source)

# recognize speech using Google Speech Recognition
try:
    text = recognizer.recognize_google(audio, language='en-US')
    print(f"Text: {text}")
except sr.UnknownValueError:
    print("Sorry, I could not recognize the audio")
except sr.RequestError as e:
    print(f"recognition request failed: {e}")