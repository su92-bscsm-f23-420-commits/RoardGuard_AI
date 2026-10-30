import random
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------------------------------
# ROADGUARD AI — Bolt-inspired Streamlit frontend
# Actual inference is performed by roadguard_final.keras.
# ------------------------------------------------------------
st.set_page_config(
    page_title="RoadGuard AI | Intelligent Road Inspection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(__file__).parent / "roadguard_final.keras"
THRESHOLD = 0.50

ROAD_VIEWS = [
    ("Lahore", "Punjab · illustrative road scene", "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=2000&q=85"),
    ("Karachi", "Sindh · illustrative road scene", "https://images.unsplash.com/photo-1519608487953-e999c86e7455?auto=format&fit=crop&w=2000&q=85"),
    ("Islamabad", "Capital Territory · illustrative road scene", "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=2000&q=85"),
    ("Rawalpindi", "Punjab · illustrative road scene", "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=2000&q=85"),
    ("Peshawar", "Khyber Pakhtunkhwa · illustrative road scene", "https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=2000&q=85"),
    ("Quetta", "Balochistan · illustrative road scene", "https://images.unsplash.com/photo-1519681393784-d120267933ba?auto=format&fit=crop&w=2000&q=85"),
    ("Multan", "Punjab · illustrative road scene", "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=2000&q=85"),
    ("Faisalabad", "Punjab · illustrative road scene", "https://images.unsplash.com/photo-1473448912268-2022ce9509d8?auto=format&fit=crop&w=2000&q=85"),
]
if "road_view" not in st.session_state:
    st.session_state.road_view = random.randrange(len(ROAD_VIEWS))
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("roadguard_final.keras must be in the same GitHub folder as app.py.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_image(image: Image.Image):
    model = load_model()
    resized = image.convert("RGB").resize((224, 224))
    arr = np.asarray(resized, dtype=np.float32)
    batch = preprocess_input(np.expand_dims(arr, axis=0))
    start = time.perf_counter()
    output = model.predict(batch, verbose=0)
    elapsed = (time.perf_counter() - start) * 1000
    score = float(np.asarray(output).reshape(-1)[0])
    return score, elapsed

# Enhanced Modern CSS Styling (Cleaned)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ink: #0f2330;
  --ink2: #1b384b;
  --muted: #4e6c80;
  --blue-accent: #0098db;
  --blue-light: #e0f4fc;
  --border-subtle: rgba(111, 177, 207, 0.22);
  --glass-bg: rgba(255, 255, 255, 0.82);
  --glass-strong: rgba(255, 255, 255, 0.95);
  --shadow-smooth: 0 16px 40px rgba(40, 98, 128, 0.08);
}

.stApp {
  background: 
    radial-gradient(circle at 10% 0%, rgba(150, 220, 248, 0.65), transparent 40%),
    radial-gradient(circle at 90% 15%, rgba(195, 235, 252, 0.8), transparent 35%),
    linear-gradient(135deg, #f4f9fd 0%, #e6f3fb 50%, #f0f7fc 100%);
  color: var(--ink);
  font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { 
  background: transparent; 
}

.block-container { 
  max-width: 1400px; 
  padding-top: 1.5rem; 
  padding-bottom: 3.5rem; 
}

h1, h2, h3, h4 { 
  font-family: 'Space Grotesk', sans-serif !important; 
  color: var(--ink) !important; 
  letter-spacing: -0.7px; 
}

p, li, label, span, div { 
  font-family: 'DM Sans', sans-serif; 
}

div[data-testid="stMarkdownContainer"] p { 
  color: var(--muted); 
}

/* Top Navigation Bar */
.topbar {
  display: flex; 
  justify-content: space-between; 
  align-items: center; 
  padding: 14px 22px;
  border: 1px solid var(--border-subtle); 
  border-radius: 18px; 
  background: var(--glass-bg);
  backdrop-filter: blur(18px); 
  -webkit-backdrop-filter: blur(18px);
  margin-bottom: 24px; 
  box-shadow: var(--shadow-smooth);
}

.brand { 
  font-family: 'Space Grotesk', sans-serif; 
  font-weight: 800; 
  font-size: 24px; 
  color: var(--ink); 
  letter-spacing: -0.8px; 
  display: flex;
  align-items: center;
  gap: 8px;
}
.brand span { 
  color: var(--blue-accent); 
}
.brandtag { 
  color: #53778d; 
  font-size: 9px; 
  letter-spacing: 2.5px; 
  font-weight: 700; 
  margin-top: 3px; 
  text-transform: uppercase;
}

.status { 
  color: #0b688c; 
  font-size: 11px; 
  font-weight: 800; 
  border: 1px solid #a8dbec; 
  background: #def3fb; 
  border-radius: 30px; 
  padding: 6px 14px;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* Hero Banner */
.hero {
  border: 1px solid rgba(255, 255, 255, 0.7); 
  border-radius: 26px; 
  padding: 42px 46px; 
  min-height: 310px;
  background-size: cover; 
  background-position: center; 
  position: relative; 
  overflow: hidden;
  box-shadow: 0 24px 60px rgba(35, 92, 121, 0.18);
  margin-bottom: 24px;
}

.eyebrow { 
  color: #9fe2ff; 
  font-size: 11px; 
  font-weight: 800; 
  letter-spacing: 2.5px; 
  text-transform: uppercase; 
}

.hero h1 { 
  font-size: clamp(36px, 4.8vw, 62px); 
  line-height: 1.05; 
  font-weight: 800; 
  margin: 12px 0 14px; 
  color: white !important; 
}

.hero h1 span { 
  color: #8ce1ff; 
}

.hero p { 
  max-width: 620px; 
  color: #e6f5ff !important; 
  font-size: 16px; 
  line-height: 1.7; 
}

.hero-foot { 
  color: #c9eaf7; 
  font-size: 11px; 
  letter-spacing: 1.2px; 
  margin-top: 24px; 
  font-weight: 600;
  text-transform: uppercase;
}

/* Cards & Panels */
.panel {
  background: linear-gradient(150deg, rgba(255, 255, 255, 0.92), rgba(235, 247, 253, 0.8));
  border: 1px solid var(--border-subtle); 
  border-radius: 20px; 
  padding: 24px;
  box-shadow: var(--shadow-smooth); 
  backdrop-filter: blur(16px);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.panel:hover {
  transform: translateY(-2px);
  box-shadow: 0 20px 45px rgba(40, 98, 128, 0.12);
}

.section-kicker { 
  color: #0b80b0; 
  font-size: 11px; 
  font-weight: 800; 
  letter-spacing: 2px; 
  text-transform: uppercase; 
  margin-bottom: 6px;
}

.muted { 
  color: var(--muted); 
  font-size: 13px; 
  line-height: 1.7; 
}

/* Result Cards */
.result-damage { 
  border: 1px solid #f5c2c2; 
  background: linear-gradient(135deg, #fff5f5, #fffcfc); 
  border-radius: 20px; 
  padding: 26px; 
  box-shadow: 0 12px 35px rgba(212, 60, 60, 0.08);
}

.result-safe { 
  border: 1px solid #b8e6ca; 
  background: linear-gradient(135deg, #f0fbf4, #fcfffd); 
  border-radius: 20px; 
  padding: 26px; 
  box-shadow: 0 12px 35px rgba(46, 153, 94, 0.08);
}

/* Custom Buttons */
div.stButton>button {
  border-radius: 14px; 
  min-height: 48px; 
  border: 1px solid #52badf;
  background: linear-gradient(100deg, #189ccc, #4fb8e4); 
  color: #ffffff; 
  font-weight: 700; 
  font-size: 14px;
  letter-spacing: 0.3px;
  transition: all 0.25s ease;
  box-shadow: 0 6px 20px rgba(24, 156, 204, 0.25);
}

div.stButton>button:hover { 
  border-color: #097fb0; 
  background: linear-gradient(100deg, #1387b5, #3da8d4);
  color: #ffffff; 
  transform: translateY(-2px); 
  box-shadow: 0 10px 28px rgba(24, 156, 204, 0.35); 
}

div.stButton>button[kind="secondary"] { 
  background: rgba(255, 255, 255, 0.85); 
  border: 1px solid #c2e2ee; 
  color: #1b4b66; 
  box-shadow: none;
}
div.stButton>button[kind="secondary"]:hover {
  background: #ffffff;
  border-color: #94d0e7;
  color: #0f3246;
}

/* File Uploader */
[data-testid="stFileUploader"] {
  border: 2px dashed #92d0e7; 
  border-radius: 16px; 
  background: rgba(255, 255, 255, 0.75);
  padding: 16px;
  transition: border-color 0.2s ease;
}
[data-testid="stFileUploader"]:hover {
  border-color: #38a5d1;
}

/* Metrics & Images */
[data-testid="stMetric"] { 
  background: rgba(255, 255, 255, 0.88); 
  border: 1px solid #d2eaf4; 
  border-radius: 16px; 
  padding: 16px; 
  box-shadow: 0 8px 20px rgba(50, 110, 140, 0.05);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 800; }

[data-testid="stImage"] img { 
  border-radius: 16px; 
  box-shadow: 0 12px 30px rgba(30, 80, 110, 0.12);
}

/* Navigation Radio Tabs */
[data-testid="stRadio"] > div { 
  gap: 8px; 
}
[data-testid="stRadio"] label { 
  background: rgba(255, 255, 255, 0.82); 
  border: 1px solid #d0e7f2; 
  border-radius: 12px; 
  padding: 10px 16px; 
  font-weight: 600;
  color: var(--ink2);
  transition: all 0.2s ease;
}
[data-testid="stRadio"] label:hover {
  background: #ffffff;
  border-color: #85cde6;
}

[data-testid="stCaptionContainer"] p { 
  color: #618399 !important; 
}

hr { 
  border-color: #d1e7f2; 
  margin: 32px 0;
}

.footer { 
  text-align: center; 
  color: #7091a6; 
  border-top: 1px solid #d1e7f2; 
  padding: 30px 0 10px; 
  font-size: 11px; 
  letter-spacing: 0.5px;
  line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# Top brand and navigation
nav1, nav2 = st.columns([1.2, 3.8], vertical_alignment="center")
with nav1:
    st.markdown('<div class="brand">🛣️ RoadGuard <span>AI</span><div class="brandtag">SEE THE ROAD. DETECT THE RISK.</div></div>', unsafe_allow_html=True)
with nav2:
    st.session_state.page = st.radio(
        "Navigation",
        ["Dashboard", "How It Works", "About Project", "Model Information"],
        horizontal=True,
        label_visibility="collapsed",
        index=["Dashboard", "How It Works", "About Project", "Model Information"].index(st.session_state.page),
    )
st.markdown('<div class="topbar"><span style="color:#9db4c5;font-size:12px">DEEP LEARNING · ROAD CONDITION CLASSIFICATION</span><span class="status">● MODEL APP</span></div>', unsafe_allow_html=True)

page = st.session_state.page

if page == "Dashboard":
    view = ROAD_VIEWS[st.session_state.road_view]
    st.markdown(f"""
    <div class="hero" style="background-image:linear-gradient(90deg,rgba(13,43,61,.82),rgba(18,58,79,.55) 58%,rgba(20,66,88,.18)),url('{view[2]}')">
      <div class="eyebrow">✦ AI-POWERED ROAD INTELLIGENCE</div>
      <h1>Smarter roads.<br><span>Safer journeys.</span></h1>
      <p>Upload a road image and let our trained deep-learning classifier assess whether the image shows road damage.</p>
      <div class="hero-foot">PAKISTAN CITY THEME · {view[0].upper()} &nbsp; / &nbsp; {view[1].upper()}</div>
    </div>
    """, unsafe_allow_html=True)
    ch1, ch2, _ = st.columns([1.25, 1.35, 3.4])
    with ch1:
        if st.button("↻ Change road background", use_container_width=True):
            st.session_state.road_view = random.choice([i for i in range(len(ROAD_VIEWS)) if i != st.session_state.road_view])
            st.rerun()
    with ch2:
        st.caption("City labels are Pakistan-focused themes; background photos are illustrative, not verified photos from those cities.")
    st.markdown("")
    st.markdown('<div class="section-kicker">01 / IMAGE INSPECTION</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.08], gap="large")
    image = None
    with left:
        st.markdown("### Upload your road image")
        st.markdown('<div class="muted">Choose a clear JPG, JPEG or PNG photo showing the road surface. The image is analyzed by the deployed Keras model.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"], key="bolt_road_upload")
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
            except Exception:
                st.error("Unable to open this file. Please upload a valid image.")
        st.markdown('<div class="muted">Tip: a clear road-surface image generally gives the classifier more useful visual information.</div>', unsafe_allow_html=True)
    with right:
        st.markdown("### Image preview")
        if image is not None:
            st.image(image, use_container_width=True, caption="Selected road image")
        else:
            st.markdown('<div class="panel" style="text-align:center;padding:65px 18px"><div style="font-size:46px">⌁</div><h3>Ready for inspection</h3><div class="muted">Your uploaded image preview will appear here.</div></div>', unsafe_allow_html=True)
    st.markdown("")
    run = st.button("✦  ANALYZE ROAD IMAGE", type="primary", use_container_width=True, disabled=image is None)
    if run and image is not None:
        try:
            with st.spinner("Running the trained RoadGuard AI model..."):
                score, elapsed = predict_image(image)
            st.markdown("---")
            st.markdown('<div class="section-kicker">02 / CLASSIFICATION REPORT</div>', unsafe_allow_html=True)
            c1, c2 = st.columns([1.15, 1], gap="large")
            with c1:
                if score >= THRESHOLD:
                    st.markdown(f'<div class="result-damage"><div class="eyebrow" style="color:#d9534f">⚠ CLASSIFICATION RESULT</div><h2 style="margin:12px 0;color:#902b2b">Road damage detected</h2><p>The trained model classified this image as the damage class.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-safe"><div class="eyebrow" style="color:#2e995e">✓ CLASSIFICATION RESULT</div><h2 style="margin:12px 0;color:#1b5e36">No road damage detected</h2><p>The trained model classified this image as the no-damage class.</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.metric("Damage-class model score", f"{score:.1%}")
                st.metric("Inference time", f"{elapsed:.0f} ms")
                st.progress(min(max(score, 0.0), 1.0))
                st.caption("Decision threshold: 50%. Score is not a calibrated probability.")
                st.markdown('</div>', unsafe_allow_html=True)
            st.info("This is whole-image binary classification. It does not draw bounding boxes or identify damage location, type, or severity. Human review is recommended.")
        except Exception as exc:
            st.error("Prediction failed. Check the model file and Streamlit logs.")
            st.exception(exc)
    st.markdown("---")
    st.markdown("### Built for smarter road awareness")
    a,b,c = st.columns(3)
    with a:
        st.markdown('<div class="panel"><div class="section-kicker">01 / DEEP LEARNING</div><h3>MobileNetV2</h3><div class="muted">A trained image-classification model powers the prediction.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="panel"><div class="section-kicker">02 / SIMPLE INPUT</div><h3>Image-based</h3><div class="muted">Upload a road photo to receive a binary classification.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="panel"><div class="section-kicker">03 / CLEAR OUTPUT</div><h3>Explainable result</h3><div class="muted">View the damage score and inference time for each prediction.</div></div>', unsafe_allow_html=True)

elif page == "How It Works":
    st.markdown('<div class="hero"><div class="eyebrow">THE PROCESS</div><h1>How it <span style="color:#36d9e8">works.</span></h1><p>From a road image to a model classification in a few simple steps.</p></div>', unsafe_allow_html=True)
    steps = [
        ("01","Upload image","Select a JPG or PNG road photograph."),
        ("02","Preprocessing","The image is converted to RGB, resized to 224 × 224 pixels and processed using MobileNetV2 preprocessing."),
        ("03","AI inference","The trained Keras model analyzes the image and returns one sigmoid score."),
        ("04","Classification","The app applies a 0.50 threshold to classify the image as damage or no damage."),
        ("05","View result","The interface displays the predicted class, model score and inference time."),
    ]
    for n,t,d in steps:
        st.markdown(f'<div class="panel" style="margin:12px 0"><div class="section-kicker">STEP {n}</div><h3>{t}</h3><div class="muted">{d}</div></div>', unsafe_allow_html=True)

elif page == "About Project":
    st.markdown('<div class="hero"><div class="eyebrow">ABOUT ROADGUARD AI</div><h1>Technology for <span style="color:#36d9e8">road awareness.</span></h1><p>RoadGuard AI is an academic deep-learning prototype for classifying road images into damage and no-damage categories.</p></div>', unsafe_allow_html=True)
    x,y = st.columns(2, gap="large")
    with x:
        st.markdown('<div class="panel"><div class="section-kicker">PROJECT MISSION</div><h3>Support preliminary inspection</h3><div class="muted">Explore how computer vision can assist early awareness of possible road-surface damage.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="section-kicker">INTENDED USE</div><h3>Academic demonstration</h3><div class="muted">Image classification for learning, demonstration and preliminary screening.</div></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="panel"><div class="section-kicker">WHAT IT CAN DO</div><h3>Image-level classification</h3><div class="muted">Accept an image, preprocess it, predict one of two classes and display the score.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel"><div class="section-kicker">CURRENT LIMITATIONS</div><h3>Not object detection</h3><div class="muted">It cannot locate a pothole, measure damage, identify damage type or certify road safety.</div></div>', unsafe_allow_html=True)

elif page == "Model Information":
    st.markdown('<div class="hero"><div class="eyebrow">MODEL CARD</div><h1>Inside the <span style="color:#36d9e8">AI model.</span></h1><p>Technical information about the deployed image classifier.</p></div>', unsafe_allow_html=True)
    x,y = st.columns(2, gap="large")
    with x:
        st.markdown('<div class="panel"><div class="section-kicker">ARCHITECTURE</div><h3>MobileNetV2</h3><p>Pretrained MobileNetV2 backbone fine-tuned for binary road-image classification.</p><hr><b>Input shape</b><p class="muted">224 × 224 × 3 RGB</p><b>Output</b><p class="muted">One sigmoid score</p></div>', unsafe_allow_html=True)
    with y:
        st.markdown('<div class="panel"><div class="section-kicker">CONFIGURATION</div><h3>Runtime details</h3><b>Framework</b><p class="muted">TensorFlow / Keras</p><b>Preprocessing</b><p class="muted">MobileNetV2 preprocess_input</p><b>Decision threshold</b><p class="muted">0.50</p><b>Model file</b><p class="muted">roadguard_final.keras</p></div>', unsafe_allow_html=True)
    st.warning("The interface assumes the model's sigmoid output corresponds to class 1 = Road Damage and class 0 = No Road Damage. Confirm this mapping against the original training labels before final submission. Model scores are not calibrated probabilities.")

st.markdown('<div class="footer"><b>ROADGUARD AI</b> · DEEP LEARNING PROJECT · MOBILENETV2<br>Academic prototype — predictions should be reviewed by a human.</div>', unsafe_allow_html=True)
