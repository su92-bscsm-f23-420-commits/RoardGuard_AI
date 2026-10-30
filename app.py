
import random
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ------------------------------------------------------------
# ROADGUARD AI — Intelligent Road Inspection
# Original prediction logic preserved.
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
    (
        "Mountain Highway",
        "Northern Pakistan inspired",
        "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=2000&q=85"
    ),
    (
        "Open Road",
        "Scenic highway landscape",
        "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=2000&q=85"
    ),
    (
        "Highland Route",
        "Mountain road landscape",
        "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=2000&q=85"
    ),
    (
        "Road Journey",
        "Open-road scenery",
        "https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=2000&q=85"
    ),
]

if "road_view" not in st.session_state:
    st.session_state.road_view = random.randrange(len(ROAD_VIEWS))

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "roadguard_final.keras must be in the same folder as app.py."
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ------------------------------------------------------------
# IMAGE PREDICTION
# Original inference logic preserved.
# ------------------------------------------------------------

def predict_image(image: Image.Image):

    model = load_model()

    resized = image.convert("RGB").resize((224, 224))

    arr = np.asarray(resized, dtype=np.float32)

    batch = preprocess_input(
        np.expand_dims(arr, axis=0)
    )

    start = time.perf_counter()

    output = model.predict(batch, verbose=0)

    elapsed = (time.perf_counter() - start) * 1000

    score = float(np.asarray(output).reshape(-1)[0])

    return score, elapsed


# ------------------------------------------------------------
# CUSTOM FRONTEND DESIGN
# ------------------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #07111d;
    --panel: #0c1b2b;
    --panel2: #10243a;
    --line: #1d3a50;
    --cyan: #36d9e8;
    --lime: #c7f36b;
    --muted: #9db1c3;
}

.stApp {
    background:
        radial-gradient(ellipse at 80% -10%, #163650 0%, transparent 38%),
        linear-gradient(145deg, #050c15, #081827 55%, #07111d);

    color: #edf6fb;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1420px;
    padding-top: 1.3rem;
    padding-bottom: 2.8rem;
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f4f8fb;
    letter-spacing: -0.6px;
}

p, li, label, span, div {
    font-family: 'DM Sans', sans-serif;
}

div[data-testid="stMarkdownContainer"] p {
    color: #bdcbd6;
}

/* TOP NAVIGATION */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 22px;
    border: 1px solid #1b3b50;
    border-radius: 17px;
    background: rgba(8,23,37,.92);
    margin-bottom: 25px;
}

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 800;
    font-size: 25px;
    color: #f5fbff;
    letter-spacing: -1px;
}

.brand span {
    color: var(--cyan);
}

.brandtag {
    color: #87a2b7;
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 700;
    margin-top: 3px;
}

.status {
    color: #c7f36b;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid #49623a;
    background: #17291c;
    border-radius: 30px;
    padding: 8px 14px;
}

/* HERO SECTION */

.hero {
    border: 1px solid #2a5066;
    border-radius: 24px;
    padding: 42px 40px;
    min-height: 300px;
    background-size: cover;
    background-position: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px #0005;
    margin-bottom: 25px;
}

.eyebrow {
    color: #61e4e7;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.2px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: clamp(38px, 5vw, 66px);
    line-height: 1.02;
    font-weight: 800;
    margin: 15px 0 14px;
    color: white;
}

.hero h1 span {
    color: var(--cyan);
}

.hero p {
    max-width: 650px;
    color: #d5e4ee !important;
    font-size: 16px;
    line-height: 1.75;
}

.hero-foot {
    color: #b9d1df;
    font-size: 11px;
    letter-spacing: 1px;
    margin-top: 22px;
}

/* GENERAL PANELS */

.panel {
    background: linear-gradient(
        145deg,
        rgba(16,37,57,.96),
        rgba(8,23,38,.97)
    );

    border: 1px solid var(--line);
    border-radius: 19px;
    padding: 23px;
    box-shadow: 0 12px 35px #0002;
}

.section-kicker {
    color: var(--cyan);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.8px;
    text-transform: uppercase;
}

.muted {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.8;
}

/* RESULT CARDS */

.result-damage {
    border: 1px solid #a94c4c;
    background: linear-gradient(135deg,#351c25,#1b1b2a);
    border-radius: 18px;
    padding: 24px;
}

.result-safe {
    border: 1px solid #5b8b4d;
    background: linear-gradient(135deg,#1c3428,#102b2c);
    border-radius: 18px;
    padding: 24px;
}

/* BUTTONS */

div.stButton > button {
    border-radius: 12px;
    min-height: 46px;
    border: 1px solid #1da9c4;
    background: linear-gradient(100deg,#087fae,#12b9c8);
    color: white;
    font-weight: 800;
    transition: .2s;
}

div.stButton > button:hover {
    border-color: #8af4ed;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px #0aaac533;
}

div.stButton > button[kind="secondary"] {
    background: #10263a;
    border: 1px solid #29485f;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    border: 1px dashed #34718a;
    border-radius: 14px;
    background: #071827;
    padding: 14px;
}

/* METRICS */

[data-testid="stMetric"] {
    background: #0d2134;
    border: 1px solid #24455d;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #a9bfce !important;
}

/* IMAGES */

[data-testid="stImage"] img {
    border-radius: 14px;
}

/* NAVIGATION */

[data-testid="stRadio"] > div {
    gap: 7px;
}

[data-testid="stRadio"] label {
    background: #0d2134;
    border: 1px solid #1d3d53;
    border-radius: 11px;
    padding: 8px 13px;
}

/* FOOTER */

hr {
    border-color: #1b394e;
}

.footer {
    text-align: center;
    color: #829caf;
    border-top: 1px solid #1b394e;
    padding: 24px 0 5px;
    font-size: 11px;
    line-height: 2;
}

/* RESPONSIVE */

@media (max-width: 768px) {

    .hero {
        padding: 25px 20px;
        min-height: 260px;
    }

    .hero h1 {
        font-size: 38px;
    }

    .brand {
        font-size: 21px;
    }

    .panel {
        padding: 17px;
    }

}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# TOP BRAND AND NAVIGATION
# ------------------------------------------------------------

nav1, nav2 = st.columns(
    [1.2, 3.8],
    vertical_alignment="center"
)

with nav1:

    st.markdown("""
    <div class="brand">
        🛣️ RoadGuard <span>AI</span>
        <div class="brandtag">
            SEE THE ROAD. DETECT THE RISK.
        </div>
    </div>
    """, unsafe_allow_html=True)

with nav2:

    st.session_state.page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "How It Works",
            "About Project",
            "Model Information"
        ],
        horizontal=True,
        label_visibility="collapsed",
        index=[
            "Dashboard",
            "How It Works",
            "About Project",
            "Model Information"
        ].index(st.session_state.page)
    )


page = st.session_state.page


# ------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------

if page == "Dashboard":

    view = ROAD_VIEWS[st.session_state.road_view]

    st.markdown(f"""
    <div class="hero"
    style="background-image:
    linear-gradient(
        90deg,
        rgba(3,12,22,.96),
        rgba(3,12,22,.72) 58%,
        rgba(3,12,22,.25)
    ),
    url('{view[2]}')">

        <div class="eyebrow">
            ✦ AI-POWERED ROAD INTELLIGENCE
        </div>

        <h1>
            Smarter roads.<br>
            <span>Safer journeys.</span>
        </h1>

        <p>
            Upload a road image and let our trained
            deep-learning classifier assess whether
            the image shows road damage.
        </p>

        <div class="hero-foot">
            SCENIC THEME · {view[0].upper()}
            &nbsp; / &nbsp;
            {view[1].upper()}
        </div>

    </div>
    """, unsafe_allow_html=True)

    ch1, ch2, _ = st.columns([1.25, 1.35, 3.4])

    with ch1:

        if st.button(
            "↻ Change road background",
            use_container_width=True
        ):

            st.session_state.road_view = random.choice(
                [
                    i for i in range(len(ROAD_VIEWS))
                    if i != st.session_state.road_view
                ]
            )

            st.rerun()

    with ch2:

        st.caption("Background is decorative only.")

    st.markdown("")

    st.markdown(
        '<div class="section-kicker">01 / IMAGE INSPECTION</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1, 1.08],
        gap="large"
    )

    image = None

    with left:

        st.markdown("### Upload your road image")

        st.markdown("""
        <div class="muted">
            Choose a clear JPG, JPEG or PNG photo
            showing the road surface. The image is
            analyzed by the deployed Keras model.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose image",
            type=["jpg", "jpeg", "png"],
            key="bolt_road_upload"
        )

        if uploaded:

            try:

                image = Image.open(uploaded).convert("RGB")

            except Exception:

                st.error(
                    "Unable to open this file. "
                    "Please upload a valid image."
                )

        st.markdown("""
        <div class="muted">
            Tip: A clear road-surface image generally
            gives the classifier more useful visual information.
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("### Image preview")

        if image is not None:

            st.image(
                image,
                use_container_width=True,
                caption="Selected road image"
            )

        else:

            st.markdown("""
            <div class="panel"
            style="text-align:center;padding:65px 18px">

                <div style="font-size:46px">
                    ⌁
                </div>

                <h3>Ready for inspection</h3>

                <div class="muted">
                    Your uploaded image preview
                    will appear here.
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    run = st.button(
        "✦  ANALYZE ROAD IMAGE",
        type="primary",
        use_container_width=True,
        disabled=image is None
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    if run and image is not None:

        try:

            with st.spinner(
                "Running the trained RoadGuard AI model..."
            ):

                score, elapsed = predict_image(image)

            st.markdown("---")

            st.markdown(
                '<div class="section-kicker">'
                '02 / CLASSIFICATION REPORT'
                '</div>',
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(
                [1.15, 1],
                gap="large"
            )

            with c1:

                if score >= THRESHOLD:

                    st.markdown("""
                    <div class="result-damage">

                        <div class="eyebrow"
                        style="color:#ff9b9b">
                            ⚠ CLASSIFICATION RESULT
                        </div>

                        <h2 style="margin:12px 0;color:#fff">
                            Road damage detected
                        </h2>

                        <p>
                            The trained model classified
                            this image as the damage class.
                        </p>

                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="result-safe">

                        <div class="eyebrow"
                        style="color:#c7f36b">
                            ✓ CLASSIFICATION RESULT
                        </div>

                        <h2 style="margin:12px 0;color:#fff">
                            No road damage detected
                        </h2>

                        <p>
                            The trained model classified
                            this image as the no-damage class.
                        </p>

                    </div>
                    """, unsafe_allow_html=True)

            with c2:

                st.markdown('<div class="panel">', unsafe_allow_html=True)

                st.metric(
                    "Damage-class model score",
                    f"{score:.1%}"
                )

                st.metric(
                    "Inference time",
                    f"{elapsed:.0f} ms"
                )

                st.progress(
                    min(max(score, 0.0), 1.0)
                )

                st.caption(
                    "Decision threshold: 50%. "
                    "Score is not a calibrated probability."
                )

                st.markdown('</div>', unsafe_allow_html=True)

            st.info(
                "This is whole-image binary classification. "
                "It does not draw bounding boxes or identify "
                "damage location, type, or severity. "
                "Human review is recommended."
            )

        except Exception:

            st.error(
                "Prediction failed. Check the model file "
                "and Streamlit logs."
            )

            st.exception(Exception("Model inference failed. See server logs for details."))

    # --------------------------------------------------------
    # FEATURE CARDS
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("### Built for smarter road awareness")

    a, b, c = st.columns(3)

    with a:

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                01 / DEEP LEARNING
            </div>

            <h3>MobileNetV2</h3>

            <div class="muted">
                A trained image-classification model
                powers the prediction.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with b:

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                02 / SIMPLE INPUT
            </div>

            <h3>Image-based</h3>

            <div class="muted">
                Upload a road photo to receive
                a binary classification.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with c:

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                03 / CLEAR OUTPUT
            </div>

            <h3>Explainable result</h3>

            <div class="muted">
                View the damage score and inference
                time for each prediction.
            </div>

        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------
# HOW IT WORKS
# ------------------------------------------------------------

elif page == "How It Works":

    st.markdown("""
    <div class="hero">

        <div class="eyebrow">
            THE PROCESS
        </div>

        <h1>
            How it
            <span style="color:#36d9e8">works.</span>
        </h1>

        <p>
            From a road image to a model classification
            in a few simple steps.
        </p>

    </div>
    """, unsafe_allow_html=True)

    steps = [

        (
            "01",
            "Upload image",
            "Select a JPG or PNG road photograph."
        ),

        (
            "02",
            "Preprocessing",
            "The image is converted to RGB, resized "
            "to 224 × 224 pixels and processed using "
            "MobileNetV2 preprocessing."
        ),

        (
            "03",
            "AI inference",
            "The trained Keras model analyzes the "
            "image and returns one sigmoid score."
        ),

        (
            "04",
            "Classification",
            "The app applies a 0.50 threshold to "
            "classify the image as damage or no damage."
        ),

        (
            "05",
            "View result",
            "The interface displays the predicted class, "
            "model score and inference time."
        )

    ]

    for n, t, d in steps:

        st.markdown(f"""
        <div class="panel" style="margin:12px 0">

            <div class="section-kicker">
                STEP {n}
            </div>

            <h3>{t}</h3>

            <div class="muted">
                {d}
            </div>

        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------
# ABOUT PROJECT
# ------------------------------------------------------------

elif page == "About Project":

    st.markdown("""
    <div class="hero">

        <div class="eyebrow">
            ABOUT ROADGUARD AI
        </div>

        <h1>
            Technology for
            <span style="color:#36d9e8">
                road awareness.
            </span>
        </h1>

        <p>
            RoadGuard AI is an academic deep-learning
            prototype for classifying road images into
            damage and no-damage categories.
        </p>

    </div>
    """, unsafe_allow_html=True)

    x, y = st.columns(2, gap="large")

    with x:

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                PROJECT MISSION
            </div>

            <h3>Support preliminary inspection</h3>

            <div class="muted">
                Explore how computer vision can assist
                early awareness of possible road-surface damage.
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                INTENDED USE
            </div>

            <h3>Academic demonstration</h3>

            <div class="muted">
                Image classification for learning,
                demonstration and preliminary screening.
            </div>

        </div>
        """, unsafe_allow_html=True)

    with y:

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                WHAT IT CAN DO
            </div>

            <h3>Image-level classification</h3>

            <div class="muted">
                Accept an image, preprocess it, predict
                one of two classes and display the score.
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                CURRENT LIMITATIONS
            </div>

            <h3>Not object detection</h3>

            <div class="muted">
                It cannot locate a pothole, measure damage,
                identify damage type or certify road safety.
            </div>

        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------
# MODEL INFORMATION
# ------------------------------------------------------------

elif page == "Model Information":

    st.markdown("""
    <div class="hero">

        <div class="eyebrow">
            MODEL CARD
        </div>

        <h1>
            Inside the
            <span style="color:#36d9e8">
                AI model.
            </span>
        </h1>

        <p>
            Technical information about the deployed
            image classifier.
        </p>

    </div>
    """, unsafe_allow_html=True)

    x, y = st.columns(2, gap="large")

    with x:

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                ARCHITECTURE
            </div>

            <h3>MobileNetV2</h3>

            <p>
                Pretrained MobileNetV2 backbone
                fine-tuned for binary road-image classification.
            </p>

            <hr>

            <b>Input shape</b>

            <p class="muted">
                224 × 224 × 3 RGB
            </p>

            <b>Output</b>

            <p class="muted">
                One sigmoid score
            </p>

        </div>
        """, unsafe_allow_html=True)

    with y:

        st.markdown("""
        <div class="panel">

            <div class="section-kicker">
                INFERENCE CONFIGURATION
            </div>

            <h3>Runtime details</h3>

            <b>Framework</b>

            <p class="muted">
                TensorFlow / Keras
            </p>

            <b>Preprocessing</b>

            <p class="muted">
                MobileNetV2 preprocess_input
            </p>

            <b>Decision threshold</b>

            <p class="muted">
                0.50
            </p>

            <b>Model file</b>

            <p class="muted">
                roadguard_final.keras
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.warning(
        "The interface assumes the model's sigmoid output "
        "corresponds to class 1 = Road Damage and class "
        "0 = No Road Damage. Confirm this mapping against "
        "the original training labels before final submission. "
        "Model scores are not calibrated probabilities."
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("""
<div class="footer">

    <b>ROADGUARD AI</b>
    · DEEP LEARNING PROJECT
    · MOBILENETV2

    <br>

    Academic prototype — predictions should be reviewed by a human.

</div>
""", unsafe_allow_html=True)
