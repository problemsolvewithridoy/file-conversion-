import io
import os
import moviepy.editor as mp
import speech_recognition as sr
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums
from transformers import AutoModelForCausalLM, AutoTokenizer

# Instantiates a client
client = speech_v1.SpeechClient()


# Set up Google Cloud Speech-to-Text API
def transcribe_gcs(gcs_uri):
    audio = speech_v1.types.RecognitionAudio(uri=gcs_uri)
    config = speech_v1.types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='bn-BD')
    response = client.recognize(config, audio)
    for result in response.results:
        return result.alternatives[0].transcript


# Set up Hugging Face Transformers language model for Bangla
model_name = "sagorsarker/bangla-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Transcribe Bangla video to text
def transcribe_video_to_text(video_path):
    # Extract audio from video
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

    # Transcribe audio to text using Google Cloud Speech-to-Text API
    text = transcribe_gcs(f"gs://<BUCKET_NAME>/<FILE_NAME>.wav")

    # Correct any recognition errors using Hugging Face Transformers language model for Bangla
    input_ids = tokenizer.encode(text, return_tensors="pt")
    beam_output = model.generate(
        input_ids, max_length=1024, num_beams=5, no_repeat_ngram_size=2, early_stopping=True
    )
    corrected_text = tokenizer.decode(beam_output[0], skip_special_tokens=True)

    return corrected_text


# Example usage
video_path = "test1.mp4"
transcribed_text = transcribe_video_to_text(video_path)
print(transcribed_text)