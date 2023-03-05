# change file format
from pydub import AudioSegment

sound = AudioSegment.from_file("audio.mp3", format="mp3")
sound.export("audioo.wav", format="wav")