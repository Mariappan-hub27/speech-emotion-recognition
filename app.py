
import streamlit as st
import librosa
import torch
import tempfile

from transformers import (
    AutoProcessor,
    AutoModelForAudioClassification
)

processor = AutoProcessor.from_pretrained(
    "model"
)

model = AutoModelForAudioClassification.from_pretrained(
    "model"
)

id2label = {
    0: "angry",
    1: "disgust",
    2: "fearful",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprised"
}

def predict_emotion(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        path = tmp.name

    audio, sr = librosa.load(
        path,
        sr=16000
    )

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probs = torch.softmax(
        outputs.logits,
        dim=1
    )

    pred = torch.argmax(
        probs,
        dim=1
    ).item()

    emotion = id2label[pred]

    confidence = probs[
        0,
        pred
    ].item()

    return emotion, confidence

st.title(
    "Speech Emotion Recognition"
)

uploaded_file = st.file_uploader(
    "Upload WAV File",
    type=["wav"]
)

if uploaded_file:

    st.audio(
        uploaded_file
    )

    if st.button(
        "Predict Emotion"
    ):

        emotion, confidence = predict_emotion(
            uploaded_file
        )

        st.success(
            f"Emotion: {emotion}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )
