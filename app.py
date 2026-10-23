from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "ham10000_mobilenetv2_finetuned.keras"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMG_SIZE = (160, 160)

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🔬")

st.title("HAM10000 Skin-Lesion Classifier")
st.write("Upload a skin-lesion image to see the model's predictions.")

st.warning(
    "Educational research prototype only. This model is not a medical "
    "diagnostic tool. Do not use its predictions for medical decisions."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    if st.button("Predict"):
        if not MODEL_PATH.exists():
            st.error(f"Model not found: {MODEL_PATH}")
            st.stop()

        model = load_model()

        image_array = np.array(
            image.resize(IMG_SIZE),
            dtype=np.float32
        )
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            image_array
        )
        image_array = np.expand_dims(image_array, axis=0)

        probabilities = model.predict(image_array, verbose=0)[0]

        if len(probabilities) != len(CLASS_NAMES):
            st.error("Model output does not match the seven expected classes.")
            st.stop()

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index])

        st.subheader("Prediction")
        st.write(f"**Predicted class:** {predicted_class}")
        st.write(f"**Model confidence:** {confidence:.2%}")

        st.subheader("Class probabilities")
        chart_data = {
            name: float(probability)
            for name, probability in zip(CLASS_NAMES, probabilities)
        }
        st.bar_chart(chart_data)

        st.caption(
            "Confidence is the model's output score, not the probability "
            "that a diagnosis is medically correct."
        )
