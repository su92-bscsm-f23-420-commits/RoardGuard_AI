
import random
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="RoadGuard AI | Road Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "roadguard_final.keras"

# --------------------------------------------------
# RANDOM SCENIC BACKGROUNDS
# These are decorative backgrounds, NOT test images.
# --------------------------------------------------
ROAD_VIEWS = [
    {
        "name": "American Highway",
        "region": "USA-inspired scenery",
        "url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1800&q=85",
    },
    {
        "name": "Mountain Journey",
        "region": "Northern Pakistan-inspired scenery",
        "url": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1800&q=85",
    },
    {
        "name": "Highland Escape",
        "region": "Mountain landscape",
        "url": "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=1800&q=85",
    },
    {
        "name": "Scenic Road Trip",
        "region": "Open-road scenery",
        "url": "https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=1800&q=85",
    },
]

if "road_view" not in st.session_state:
    st.session_state.road_view = random.randrange(len(ROAD_VIEWS))

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #06101e;
    --panel: #0b1b30;
    --border: #1b3b59;
    --blue: #16b8ff;
    --cyan: #42e8dc;
    --muted: #a6bbd0;
}

.stApp {
    background:
      radial-gradient(ellipse at 80% 0%, #102d4a 0%, transparent 42%),
      linear-gradient(145deg, #050d18, #081629 55%, #06101e);
    color: #f1f7ff;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] {
    background: rgba(5, 13, 24, 0.75);
}
.block-container {
    max-width: 1500px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif;
}
h1 { letter-spacing: -1.7px; }
h2 { letter-spacing: -0.5px; }
section[data-testid="stSidebar"] {
    background: #061323;
    border-right: 1px solid #19334d;
}
section[data-testid="stSidebar"] * {
    color: #e4efff;
}
.brand {
    font-size: 26px;
    font-weight: 900;
    letter-spacing: -1px;
}
.brand span { color: var(--blue); }
.tagline {
    font-size: 12px;
    color: #8ca9c4;
    margin-bottom: 22px;
}
.eyebrow {
    color: var(--cyan);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero {
    border: 1px solid #285071;
    border-radius: 24px;
    padding: 34px;
    min-height: 260px;
    background-size: cover;
    background-position: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 60px rgba(0,0,0,.25);
}
.hero h1 {
    color: white;
    font-size: clamp(38px, 5vw, 64px);
    font-weight: 900;
    line-height: 1.08;
    margin: 14px 0 10px;
}
.hero p {
    color: #e2edf9;
    font-size: 16px;
    max-width: 650px;
    line-height: 1.8;
}
.glass-panel {
    background: linear-gradient(145deg, rgba(15,37,60,.96), rgba(7,22,39,.96));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 12px 32px rgba(0,0,0,.14);
}
.section-label {
    color: var(--blue);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}
.small-muted {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
}
div.stButton > button {
    min-height: 44px;
    border-radius: 11px;
    border: 1px solid #1679be;
    background: linear-gradient(100deg, #0877eb, #08a7d8);
    color: white;
    font-weight: 750;
    transition: all .2s ease;
}
div.stButton > button:hover {
    border-color: #55e8ff;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 5px 22px rgba(0,160,255,.2);
}
[data-testid="stFileUploader"] {
    border: 1px dashed #2777a6;
    border-radius: 14px;
    padding: 12px;
    background: rgba(3,15,29,.55);
}
[data-testid="stMetric"] {
    background: #0c2137;
    border: 1px solid #1d4564;
    padding: 15px;
    border-radius: 15px;
}
[data-testid="stMetricLabel"] { color: #a9bfd6; }
div[data-testid="stImage"] img {
    border-radius: 14px;
}
hr { border-color: #203951; }
.footer {
    color: #829db8;
    font-size: 12px;
    text-align: center;
    padding: 24px 0 10px;
    border-top: 1px solid #1b334b;
}
div[data-testid="stAlert"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "roadguard_final.keras is missing from the repository root."
        )
    return tf.keras.models.load_model(MODEL_PATH)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand">🛣️ RoadGuard <span>AI</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="tagline">SEE THE ROAD. DETECT THE RISK.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏠  Dashboard", "ℹ️  About Project",
         "⚙️  How It Works", "🧠  Model Information"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.markdown("**PROJECT STATUS**")
    st.success("Application interface ready")
    st.caption("Deep Learning · Image Classification")
    st.markdown("---")
    st.caption("Academic prototype — not a road safety authority.")

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "🏠  Dashboard":
    view = ROAD_VIEWS[st.session_state.road_view]

    # Random decorative background
    st.markdown(f"""
    <div class="hero" style="
        background-image:
        linear-gradient(90deg, rgba(3,12,24,.94),
        rgba(3,12,24,.60) 60%, rgba(3,12,24,.25)),
        url('{view["url"]}');
    ">
        <div class="eyebrow">✦ AI-POWERED ROAD INTELLIGENCE</div>
        <h1>RoadGuard <span style="color:#16b8ff">AI.</span></h1>
        <p>
            Smarter road inspection starts here. Upload a road image
            and use deep learning to classify possible road damage.
        </p>
        <div style="color:#b5d7f2;font-size:12px">
            CURRENT SCENIC THEME: {view["name"]} · {view["region"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("🔄 Change Road View", use_container_width=True):
            old_view = st.session_state.road_view
            choices = [i for i in range(len(ROAD_VIEWS)) if i != old_view]
            st.session_state.road_view = random.choice(choices)
            st.rerun()
    with b2:
        st.caption("Background changes independently of predictions.")

    st.write("")
    st.markdown(
        '<div class="section-label">01 / YOUR ROAD IMAGE</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown("### Upload & Inspect")
        st.markdown(
            '<div class="small-muted">Choose a road image from your device. '
            'Use a clear photo of the road surface for best results.</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Upload JPG, JPEG or PNG",
            type=["jpg", "jpeg", "png"],
            key="road_upload",
        )

        image_to_analyze = None
        if uploaded:
            try:
                image_to_analyze = Image.open(uploaded).convert("RGB")
            except Exception:
                st.error("This file could not be opened as an image.")

        st.markdown(
            '<div class="small-muted">Tip: try a photo with visible road '
            'surface, potholes or cracks. The model classifies the whole '
            'image; it does not draw damage-location boxes.</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("### Image Preview")
        if image_to_analyze is not None:
            st.image(
                image_to_analyze,
                caption="Selected image — ready for classification",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div class="glass-panel" style="
                text-align:center;padding:54px 18px;
                border-style:dashed;">
                <div style="font-size:48px">🛣️</div>
                <h3>Ready for inspection</h3>
                <p class="small-muted">
                    Your selected road image will appear here.
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    analyze = st.button(
        "✦  ANALYZE ROAD IMAGE",
        type="primary",
        use_container_width=True,
        disabled=image_to_analyze is None,
    )

    if analyze and image_to_analyze is not None:
        try:
            with st.spinner("Loading model and analyzing your image..."):
                model = load_model()
                resized = image_to_analyze.resize((224, 224))
                arr = np.asarray(resized, dtype=np.float32)
                batch = preprocess_input(np.expand_dims(arr, axis=0))
                start_time = time.time()
                score = float(model.predict(batch, verbose=0)[0][0])
                elapsed = time.time() - start_time

            st.markdown("---")
            st.markdown(
                '<div class="section-label">02 / CLASSIFICATION REPORT</div>',
                unsafe_allow_html=True,
            )

            r1, r2 = st.columns([1.3, 1], gap="large")
            with r1:
                st.markdown("### Analysis Result")
                if score >= 0.5:
                    st.error("🚧 ROAD DAMAGE DETECTED")
                    st.write(
                        "The model classified this image as likely "
                        "containing road damage."
                    )
                else:
                    st.success("✓ NO ROAD DAMAGE DETECTED")
                    st.write(
                        "The model classified this image as not showing "
                        "road damage."
                    )

            with r2:
                st.markdown("### Prediction Details")
                st.metric("Model damage score", f"{score:.1%}")
                st.metric("Inference time", f"{elapsed:.2f} sec")
                st.progress(min(max(score, 0.0), 1.0))
                st.caption(
                    "Score is not a calibrated probability. Results can "
                    "be wrong and require human review."
                )

            st.info(
                "This is a binary image classifier. It does not identify "
                "the exact location, size or severity of damage."
            )

        except Exception as exc:
            st.error("Prediction could not be completed.")
            st.exception(exc)

    st.markdown("---")
    st.markdown("### Built for smarter road awareness")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("#### ⚡ Fast inference")
        st.caption("Uses the trained MobileNetV2 model.")
    with f2:
        st.markdown("#### 🧠 Deep learning")
        st.caption("Binary classification of uploaded road images.")
    with f3:
        st.markdown("#### 🌍 Safer infrastructure")
        st.caption("An academic prototype for road-image analysis.")

# --------------------------------------------------
# ABOUT
# -------------------------------------------------
elif page == "ℹ️  About Project":
    st.markdown("# About RoadGuard AI")
    st.markdown(
        "RoadGuard AI is an academic deep-learning project that "
        "classifies uploaded road images into two categories: "
        "**damage detected** and **no damage detected**."
    )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎯 Project Mission")
        st.write(
            "Explore how computer vision can support early awareness "
            "of possible road-surface problems."
        )
        st.markdown("### 🛣️ Intended Use")
        st.write(
            "Educational demonstrations and preliminary image screening. "
            "It is not a substitute for professional road inspection."
        )
    with c2:
        st.markdown("### 🔍 What it can do")
        st.write(
            "- Accept a road photo\n"
            "- Resize and preprocess the image\n"
            "- Predict one of two classes\n"
            "- Display the model score"
        )
        st.markdown("### ⚠️ What it cannot do")
        st.write(
            "- Guarantee a correct result\n"
            "- Locate or measure a pothole\n"
            "- Assess structural road safety\n"
            "- Replace an engineering inspection"
        )

# --------------------------------------------------
# HOW IT WORKS
# --------------------------------------------------
elif page == "⚙️  How It Works":
    st.markdown("# How It Works")
    st.write("The application follows this image-classification pipeline.")

    steps = [
        ("01", "Image Upload", "User selects a road image."),
        ("02", "Preprocessing", "Image is converted to RGB, resized to 224 × 224, and normalized using MobileNetV2 preprocessing."),
        ("03", "Model Inference", "The trained Keras model generates a score."),
        ("04", "Classification", "A score of 0.5 or higher is labelled damage detected; otherwise, no damage detected."),
        ("05", "Result Display", "The application displays the class and score."),
    ]

    for number, title, detail in steps:
        st.markdown(f"""
        <div class="glass-panel">
            <div class="section-label">STEP {number}</div>
            <h3>{title}</h3>
            <div class="small-muted">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------
elif page == "🧠  Model Information":
    st.markdown("# Model Information")
    st.write("Technical details of the deployed classifier.")

    a, b = st.columns(2)
    with a:
        st.markdown("### Architecture")
        st.write("**Backbone:** MobileNetV2 pretrained on ImageNet")
        st.write("**Output:** One sigmoid score")
        st.write("**Input size:** 224 × 224 × 3")
        st.write("**Classes:** Damage / No damage")
    with b:
        st.markdown("### Runtime")
        st.write("**Framework:** TensorFlow / Keras")
        st.write("**Preprocessing:** MobileNetV2 preprocess_input")
        st.write("**Weights:** `roadguard_final.keras`")
        st.write("**Decision threshold:** 0.5")

    st.warning(
        "A high model score is not necessarily a calibrated probability. "
        "The model can make confident mistakes, particularly on images "
        "that differ from its training data."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    <b>ROADGUARD AI</b> · Deep Learning Project · MobileNetV2<br><br>
    Academic prototype — predictions must be reviewed by a person.
</div>
""", unsafe_allow_html=True)
