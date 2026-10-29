
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="centered"
)

st.title("🛣️ RoadGuard AI")
st.subheader("AI-Powered Road Damage Classification")
st.write(
    "Upload a road image to classify it as damaged "
    "or not damaged using our trained MobileNetV2 model."
)

MODEL_PATH = Path(__file__).parent / "roadguard_final.keras"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file roadguard_final.keras is missing."
        )
    return tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded road image",
                 use_container_width=True)

        if st.button("Analyze Road", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_model()
                resized = image.resize((224, 224))
                array = np.asarray(resized, dtype=np.float32)
                batch = preprocess_input(
                    np.expand_dims(array, axis=0)
                )
                score = float(
                    model.predict(batch, verbose=0)[0][0]
                )

            st.metric("Model damage score", f"{score:.2%}")

            if score >= 0.5:
                st.error("🚧 ROAD DAMAGE DETECTED")
            else:
                st.success("✅ NO ROAD DAMAGE DETECTED")

            st.caption(
                "The score is not a calibrated probability. "
                "Predictions can be incorrect and should be "
                "reviewed by a person."
            )

    except Exception as e:
        st.error("Unable to analyze this image.")
        st.exception(e)

st.divider()
st.caption(
    "Academic prototype | Binary image classification | "
    "MobileNetV2"
)
