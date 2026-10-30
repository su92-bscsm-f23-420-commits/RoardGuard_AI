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

# Enhanced Modern CSS Styling
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
