import streamlit as st
import os
import time
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import wave
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import torch
import accelerate
print(torch.__version__)


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


# Load tokenizer and model, create trainer
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model)


# Function to capture audio
def capture_audio(duration, sample_rate=44100):
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return recording, sample_rate


# Function to save audio as a PCM WAV file
def save_audio_as_wav(file_path, AudioData, SampleRate):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # Sample width in bytes (16-bit PCM)
        wf.setframerate(SampleRate)
        wf.writeframes(AudioData.tobytes())


def convert_to_wav(input_path, output_path):
    try:
        st.write(f"Converting file: {input_path}")
        # st.write(f"Input file exists: {os.path.exists(input_path)}")
        # st.write(f"Input file size: {os.path.getsize(input_path)} bytes")
        data, samplerate = sf.read(input_path)
        sf.write(output_path, data, samplerate)
        # st.write(f"Conversion complete. Output file: {output_path}")
        # st.write(f"Output file exists: {os.path.exists(output_path)}")
        # st.write(f"Output file size: {os.path.getsize(output_path)} bytes")
        return True
    except Exception as e:
        st.write(f"Error converting audio to WAV: {str(e)}")
        # st.write(f"Error type: {type(e)._name_}")
        return False


def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error with the speech recognition service."


def analyze_emotions(text):
    tokenized_texts = tokenizer([text], truncation=True, padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    predictions = trainer.predict(pred_dataset)

    temp = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True))

    emotions = {
        'anger': temp[0][0],
        'disgust': temp[0][1],
        'fear': temp[0][2],
        'joy': temp[0][3],
        'neutral': temp[0][4],
        'sadness': temp[0][5],
        'surprise': temp[0][6],
    }

    return emotions


def print_emotions(emotions):
    # st.write("--------------------------")
    # for emotion, score in emotions.items():
    #     st.write(f"{emotion.capitalize()}: {score:.2%}")

    st.write("--------------------------")
    st.write("Emotion Analysis Results:")
    dominant_emotion = max(emotions, key=emotions.get)
    st.write(f"Bossy emotion : {dominant_emotion.capitalize()} ({emotions[dominant_emotion]:.2%})")


st.title("Emotion Classification App")


def add_transcription_to_history(source, text, emotions):
    timestamp = time.strftime('%B %d, %Y at %I:%M')
    emotion_analysis = "\n".join([f"{emotion.capitalize()}: {score:.2%}" for emotion, score in emotions.items()])
    dominant_emotion = max(emotions, key=emotions.get)
    entry = f'Audio Received at "{timestamp}"\n\nAudio Loaded From "{source}"\n\nTranscribed Text:\n\n{text[:100]}...\n\nEmotion Analysis:\n\n{emotion_analysis}\n\nBossy emotion : {dominant_emotion.capitalize()} ({emotions[dominant_emotion]:.2%})'
    st.session_state['history'].insert(0, entry)


if 'history' not in st.session_state:
    st.session_state['history'] = []

st.sidebar.header("Transcription History")

if st.session_state['history']:
    for i, entry in enumerate(reversed(st.session_state['history']), 1):
        with st.sidebar.expander(f"Transcription {i}", expanded=(i == 1)):
            st.write(entry)
    if st.sidebar.button("Clear History"):
        st.session_state['history'] = []
        st.rerun()
else:
    st.sidebar.write("No Previous Transcription Available.")

input_method = st.selectbox("Choose audio input method:", ["Record Audio", "Upload File"])

if input_method == "Record Audio":
    duration = st.slider("Select recording duration (seconds):", min_value=1, max_value=30, value=5, step=1)
    # duration = st.number_input("Select recording duration (seconds):", min_value=1, max_value=30, value=5, step=1)
    # duration = st.selectbox("Select recording duration (seconds):", options=[1, 5, 10, 15, 20, 30], index=1)
    # duration = st.radio("Select recording duration (seconds):", options=[1, 5, 10, 15, 20, 30])
    # duration = st.text_input("Select recording duration (seconds):", value="5")
    if st.button("Start Recording"):
        st.write(f"Recording for {duration} seconds... Please speak into your microphone.")
        audio_data, sample_rate = capture_audio(duration)
        audio_file_path = "recorded_audio.wav"

        save_audio_as_wav(audio_file_path, audio_data, sample_rate)
        st.write("Recording stopped. Processing audio file...")
        text = transcribe_audio(audio_file_path)
        st.write("Transcribed Text:")
        st.write(text)

        emotions = analyze_emotions(text)
        # st.write("Emotion Analysis:")
        print_emotions(emotions)

        add_transcription_to_history("Recorded Audio", text, emotions)
        os.remove(audio_file_path)

elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac", "aac"])
    if uploaded_file is not None:
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            file_extension = uploaded_file.name.split('.')[-1].lower()
            temp_file_path = f"uploaded_audio_{timestamp}.{file_extension}"

            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.write(f"Uploaded file: {temp_file_path}")
            # st.write(f"File size: {os.path.getsize(temp_file_path)} bytes")

            # If the file is already a WAV, we don't need to convert it
            if file_extension == 'wav':
                wav_file_path = temp_file_path
                # st.write("File is already in WAV format. Skipping conversion.")
            else:
                wav_file_path = f"uploaded_audio_{timestamp}.wav"
                if convert_to_wav(temp_file_path, wav_file_path):
                    st.write(f"Converted to WAV: {wav_file_path}")
                    # st.write(f"WAV file size: {os.path.getsize(wav_file_path)} bytes")
                else:
                    st.write("Failed to convert the uploaded audio file to WAV format.")
                    raise Exception("Conversion to WAV failed")

            text = transcribe_audio(wav_file_path)
            st.write("Transcribed Text:")
            st.write(text)
            emotions = analyze_emotions(text)
            # st.write("Emotion Analysis:")
            print_emotions(emotions)

            add_transcription_to_history("Uploaded File", text, emotions)

            # Clean up temporary files
            if os.path.exists(temp_file_path) and temp_file_path != wav_file_path:
                os.remove(temp_file_path)
                # st.write(f"Removed temporary file: {temp_file_path}")
            if os.path.exists(wav_file_path):
                os.remove(wav_file_path)
                # st.write(f"Removed WAV file: {wav_file_path}")

        except Exception as e:
            st.write(f"An error occurred: {str(e)}")
            st.write(f"Error type: {type(e)._name_}")
