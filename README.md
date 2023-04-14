# Video to Bangla Text Converter
This is a Python script that can be used to convert Bangla speech from a video file to Bangla text. It uses the Google Cloud Speech-to-Text API to transcribe the audio from the video file into text.

## Prerequisites
```bash
Python 3.x
Google Cloud SDK
Google Cloud Project with Speech-to-Text API enabled
```

## Installation
1. Clone this repository: git clone https://github.com/yourusername/video-to-bangla-text-converter.git
2. Install the required Python packages: pip install -r requirements.txt
3. Set up Google Cloud SDK and authentication by following the instructions in the Google Cloud Speech-to-Text API documentation.
4. Update the config.py file with your Google Cloud Project ID and the name of your input video file.

## Deployment

To deploy this project run

```bash
# Please Subscribe my youtube channel "@problemsolvewithridoy"
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
```

## License
This project is licensed under the MIT License.

## Acknowledgments
1. Thanks to the Google Cloud team for providing the Speech-to-Text API.
2. Thanks to Bangla Speech Corpus for providing the sample Bangla speech audio used in the demo.

## You can follow me

Facebook:- https://www.facebook.com/problemsolvewithridoy/

Linkedin:- https://www.linkedin.com/in/ridoyhossain/

YouTube:- https://www.youtube.com/@problemsolvewithridoy

Gmail:- entridoy2@gmail.com

If you have any confusion, please feel free to contact me. Thank you
