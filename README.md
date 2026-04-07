# 🎙️ Emotion Classification App

A real-time speech emotion analysis application built with **Streamlit** that captures or accepts audio input, transcribes it to text, and classifies the underlying emotion using a pre-trained transformer model.

---

## ✨ Features

- 🎤 **Live Audio Recording** — Record directly from your microphone with a configurable duration
- 📁 **File Upload Support** — Upload audio files in WAV, MP3, OGG, FLAC, or AAC format
- 🔤 **Speech-to-Text Transcription** — Powered by Google Speech Recognition
- 🧠 **Emotion Detection** — Uses `j-hartmann/emotion-english-distilroberta-base` to classify 7 emotions
- 📜 **Transcription History** — Sidebar log with timestamps, transcripts, and emotion breakdowns
- 🔄 **Auto Format Conversion** — Non-WAV files are automatically converted before processing

---

## 🧪 Detected Emotions

| Emotion   | Description                        |
|-----------|------------------------------------|
| Anger     | Hostile or frustrated tone         |
| Disgust   | Aversion or strong disapproval     |
| Fear      | Anxious or frightened speech       |
| Joy       | Positive, happy expressions        |
| Neutral   | Calm, matter-of-fact delivery      |
| Sadness   | Sorrowful or low-energy tone       |
| Surprise  | Unexpected or astonished reactions |

---

## 🛠️ Tech Stack

| Component            | Library / Tool                                      |
|----------------------|-----------------------------------------------------|
| UI Framework         | [Streamlit](https://streamlit.io/)                  |
| Speech Recognition   | `SpeechRecognition` + Google Web Speech API         |
| Audio Capture        | `sounddevice`                                       |
| Audio Processing     | `soundfile`, `wave`                                 |
| Emotion Model        | `transformers` — DistilRoBERTa fine-tuned on emotions |
| Deep Learning        | `PyTorch` + `accelerate`                            |
| Numerical Computing  | `NumPy`                                             |

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/emotion-classification-app.git
cd emotion-classification-app
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On some systems, `sounddevice` requires PortAudio. Install it with:
> - **Ubuntu/Debian:** `sudo apt-get install portaudio19-dev`
> - **macOS:** `brew install portaudio`
> - **Windows:** Usually works out of the box.

---

## 📋 Requirements

Create a `requirements.txt` with the following:

```
streamlit
speechrecognition
sounddevice
soundfile
transformers
torch
accelerate
numpy
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Usage

### Record Audio
1. Select **"Record Audio"** from the dropdown
2. Use the slider to choose recording duration (1–30 seconds)
3. Click **"Start Recording"** and speak into your microphone
4. The app will transcribe the speech and display the emotion analysis

### Upload Audio File
1. Select **"Upload File"** from the dropdown
2. Upload a supported audio file (`.wav`, `.mp3`, `.ogg`, `.flac`, `.aac`)
3. The app automatically converts non-WAV files and processes them

### View History
- All past transcriptions and emotion results are stored in the **sidebar**
- Each entry includes the timestamp, source, transcript preview, and full emotion breakdown
- Click **"Clear History"** to reset

---

## 📁 Project Structure

```
emotion-classification-app/
│
├── app.py                  # Main Streamlit application
└── README.md               # Project documentation
```

---

## 🤖 Model Details

This app uses [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) from Hugging Face — a DistilRoBERTa model fine-tuned on multiple emotion datasets to classify English text into 7 emotion categories.

Emotion scores are computed using **softmax** over raw logits, giving a probability distribution across all 7 labels.

---

## ⚠️ Known Limitations

- Speech recognition requires an **active internet connection** (Google Web Speech API)
- Microphone recording may not work in all cloud/remote environments
- Accuracy depends on audio clarity and background noise levels
- Long audio files may take additional time to process

---

## 🙏 Acknowledgements

- [Jochen Hartmann](https://huggingface.co/j-hartmann) for the emotion classification model
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the model hub and inference utilities
- [Streamlit](https://streamlit.io/) for the rapid UI framework
