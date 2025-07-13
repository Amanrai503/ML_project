import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

# Load model
model = load_model("sentiment_cnn_model.h5")

# Load tokenizer
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(json.load(f))

maxlen = 100  # Must match what was used during training

# App UI
st.title("📊 Sentiment Classifier")
st.write("Enter a review below, and the model will predict its sentiment.")

user_input = st.text_area("✍️ Type your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocess
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=maxlen, padding='post')

        # Predict
        prediction = model.predict(padded)[0][0]
        label = "Positive 😊" if prediction >= 0.5 else "Negative 😞"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: `{prediction:.2f}`")
