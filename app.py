
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="RoadGuard AI | Smart Road Analysis",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path(__file__).parent / "roadguard_final.keras"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1728 55%, #101f35 100%);
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}
.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.hero {
    padding: 34px 30px;
    border: 1px solid #243852;
    border-radius: 24px;
    background: linear-gradient(120deg, #13253c, #102039);
    margin-bottom: 25px;
}
.eyebrow {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero h1 {
    font-size: clamp(34px, 5vw, 54px);
    line-height: 1.1;
    margin: 12px 0;
    color: #f8fafc;
    font-weight: 800;
}
.hero p {
    color: #b8c7d9;
    font-size: 16px;
    line-height: 1.7;
}
.panel {
    background: #101e31;
    border: 1px solid #263a53;
    border-radius: 20px;
    padding: 24px;
    min-height: 220px;
}
.panel h3 {
    color: #f8fafc;
    margin-top: 0;
}
.muted {
    color: #9fb0c5;
    font-size: 14px;
    line-height: 1.7;
}
div.stButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: 1px solid #38bdf8;
    background: linear-gradient(90deg, #0284c7, #2563eb);
    color: white;
    font-weight: 700;
    transition: 0.2s;
}
div.stButton > button:hover {
    border-color: #7dd3fc;
    color: white;
    transform: translateY(-1px);
}
[data-testid="stFileUploader"] {
    background: #0b1728;
    border: 1px dashed #3b82a0;
    border-radius: 14px;
    padding: 12px;
}
[data-testid="stFileUploader"] section {
    background: transparent;
}
div[data-testid="stMetric"] {
    background: #13253a;
    border: 1px solid #2a4059;
    padding: 17px;
    border-radius: 15px;
}
div[data-testid="stMetricLabel"] {
    color: #b8c7d9;
}
.footer {
    text-align: center;
    color: #8295ac;
    font-size: 12px;
    border-top: 1px solid #24364d;
    padding-top: 22px;
    margin-top: 38px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "roadguard_final.keras was not found in the repository."
        )
    return tf.keras.models.load_model(MODEL_PATH)


st.markdown("""
<div class="hero">
    <div class="eyebrow">Computer Vision • Deep Learning</div>
    <h1>RoadGuard <span style="color:#38bdf8">AI.</span></h1>
    <p>
        Intelligent road image classification powered by MobileNetV2.
        Upload a road image and let the model analyze it.
    </p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="eyebrow">01 / Upload</div>
        <h3>Analyze a road</h3>
        <p class="muted">
            Select a clear road photograph in JPG, JPEG or PNG format.
            The model will classify the image into one of two categories.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose road image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    analyze = st.button(
        "✦  Analyze Road Image",
        type="primary",
        disabled=uploaded_file is None
    )

with right:
    st.markdown("""
    <div class="eyebrow">02 / Image Preview</div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Image selected for analysis",
                use_container_width=True
            )
        except Exception:
            st.error("This image could not be opened. Please try another.")
            image = None
    else:
        image = None
        st.markdown("""
        <div class="panel" style="text-align:center;padding:55px 20px">
            <div style="font-size:48px">🛣️</div>
            <h3>Ready for inspection</h3>
            <p class="muted">
                Your road image preview will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)

if analyze and image is not None:
    try:
        with st.spinner("AI is analyzing the road image..."):
            model = load_model()
            resized = image.resize((224, 224))
            array = np.asarray(resized, dtype=np.float32)
            batch = preprocess_input(np.expand_dims(array, axis=0))
            score = float(model.predict(batch, verbose=0)[0][0])

        st.markdown("---")
        st.markdown(
            '<div class="eyebrow">03 / Analysis Result</div>',
            unsafe_allow_html=True
        )

        result_col, score_col = st.columns([1.4, 1])

        with result_col:
            if score >= 0.5:
                st.error("🚧 ROAD DAMAGE DETECTED")
                st.write(
                    "The model classified this image as containing "
                    "road damage."
                )
            else:
                st.success("✓ NO ROAD DAMAGE DETECTED")
                st.write(
                    "The model classified this image as not containing "
                    "road damage."
                )

        with score_col:
            st.metric("Model damage score", f"{score:.2%}")
            st.caption(
                "Model score, not a calibrated probability."
            )

        st.progress(min(max(score, 0.0), 1.0))

    except Exception as e:
        st.error("Analysis failed. Check the model and dependencies.")
        st.exception(e)

st.markdown("""
<div class="footer">
    <b>ROADGUARD AI</b> &nbsp; • &nbsp;
    Academic Deep Learning Project<br><br>
    MobileNetV2 · Binary Image Classification<br>
    This prototype may make incorrect predictions.
    Results require human review.
</div>
""", unsafe_allow_html=True)
