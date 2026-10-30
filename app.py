import streamlit as st
import numpy as np
import time
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RoadGuard AI",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_PATH = Path(__file__).parent / "roadguard_final.keras"
IMAGE_SIZE = (224, 224)
THRESHOLD = 0.50

# Change these labels if your model's class mapping is reversed.
POSITIVE_LABEL = "Damaged Road"
NEGATIVE_LABEL = "Normal Road"


# =========================================================
# ROAD SLIDER IMAGES
# =========================================================

ROAD_VIEWS = [
    {
        "city": "LAHORE, PUNJAB",
        "road": "Road conditions after rainfall",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Farooq_hospital_road_after_raining_image_-_panoramio.jpg"
    },
    {
        "city": "LAHORE, PAKISTAN",
        "road": "Ring Road infrastructure",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Ringroadlahore.jpg"
    },
    {
        "city": "KARACHI, SINDH",
        "road": "Urban road infrastructure",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Karachi_Road.jpg"
    },
    {
        "city": "KARACHI, PAKISTAN",
        "road": "Airport Road infrastructure",
        "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Airport_Road_karachi.jpg"
    }
]


# =========================================================
# SESSION STATE
# =========================================================

if "road_view" not in st.session_state:
    st.session_state.road_view = 0

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):
    model = load_model()

    if model is None:
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    start_time = time.perf_counter()

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    resized_image = image.resize(IMAGE_SIZE)

    # Convert image to array
    image_array = np.array(resized_image, dtype=np.float32)

    # MobileNetV2 preprocessing
    image_array = preprocess_input(image_array)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Model prediction
    prediction = model.predict(image_array, verbose=0)

    # Extract prediction score
    score = float(np.asarray(prediction).reshape(-1)[0])

    # Calculate inference time
    inference_time = time.perf_counter() - start_time

    # Determine result
    if score >= THRESHOLD:
        label = POSITIVE_LABEL
        confidence = score * 100
        status = "damage"
    else:
        label = NEGATIVE_LABEL
        confidence = (1 - score) * 100
        status = "normal"

    return {
        "label": label,
        "score": score * 100,
        "confidence": confidence,
        "inference_time": inference_time,
        "status": status
    }


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #07111f;
    color: #eaf4fb;
}

/* Main content */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #0b1929;
    border-right: 1px solid #1d3448;
}

section[data-testid="stSidebar"] * {
    color: #eaf4fb;
}

.sidebar-logo {
    font-size: 25px;
    font-weight: 800;
    color: #42d9e8;
    letter-spacing: -1px;
    padding: 10px 0 2px 0;
}

.sidebar-subtitle {
    color: #8da7bb;
    font-size: 11px;
    letter-spacing: 2px;
    margin-bottom: 30px;
}

/* Headings */

h1, h2, h3 {
    color: #f2f8fc !important;
}

p {
    color: #a9bdcc;
}

/* Hero slider */

.road-slider {
    position: relative;
    min-height: 390px;
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid #29475d;
    background-size: cover;
    background-position: center;
    display: flex;
    align-items: center;
    padding: 45px;
    box-shadow: 0 25px 70px rgba(0,0,0,.35);
}

.slider-content {
    max-width: 720px;
    position: relative;
    z-index: 2;
}

.slider-label {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 30px;
    background: rgba(54,217,232,.13);
    border: 1px solid rgba(54,217,232,.4);
    color: #57e7ed;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.road-slider h1 {
    font-size: clamp(38px, 5vw, 68px);
    line-height: 1.05;
    margin: 22px 0 15px;
    color: white;
    font-weight: 800;
}

.road-slider h1 span {
    color: #36d9e8;
}

.road-slider p {
    color: #e0edf5 !important;
    font-size: 15px;
    line-height: 1.8;
    max-width: 620px;
}

.slider-location {
    margin-top: 25px;
    color: #d1e6f0;
    font-size: 11px;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.slider-count {
    color: #36d9e8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(135deg, #20b9ce, #287fe0);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 18px;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #35d7e6, #3695f0);
    border: none;
    transform: translateY(-2px);
}

/* Cards */

.info-card {
    background: linear-gradient(145deg, #102338, #0b1929);
    border: 1px solid #203c52;
    border-radius: 17px;
    padding: 22px;
    min-height: 125px;
}

.card-label {
    color: #8eaabd;
    font-size: 11px;
    letter-spacing: 1.5px;
    font-weight: 700;
    text-transform: uppercase;
}

.card-value {
    color: #f1f8fc;
    font-size: 25px;
    font-weight: 800;
    margin-top: 12px;
}

.card-icon {
    font-size: 23px;
    margin-bottom: 10px;
}

/* Section heading */

.section-heading {
    color: #f1f8fc;
    font-size: 23px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 5px;
}

.section-description {
    color: #91aabd;
    font-size: 13px;
    margin-bottom: 20px;
}

/* Result panel */

.result-panel {
    background: #102338;
    border: 1px solid #29475d;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.result-damage {
    border-left: 5px solid #ff6868;
}

.result-normal {
    border-left: 5px solid #36d9a0;
}

.result-title {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 8px;
}

.result-description {
    color: #a9bdcc;
    font-size: 14px;
    line-height: 1.7;
}

/* Upload area */

[data-testid="stFileUploader"] {
    background: #102338;
    border: 1px dashed #31536a;
    border-radius: 15px;
    padding: 15px;
}

/* Progress bar */

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #20b9ce, #36d9a0);
}

/* Divider */

hr {
    border-color: #20394e;
}

/* Footer */

.footer {
    text-align: center;
    color: #708da2;
    font-size: 11px;
    padding: 30px 0 10px 0;
    letter-spacing: 1px;
}

@media(max-width: 768px) {
    .road-slider {
        min-height: 330px;
        padding: 25px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-logo">🛣️ RoadGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">INTELLIGENT ROAD INSPECTION</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    selected_page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "How It Works",
            "About Project",
            "Model Information"
        ],
        index=[
            "Dashboard",
            "How It Works",
            "About Project",
            "Model Information"
        ].index(st.session_state.page)
    )

    st.session_state.page = selected_page

    st.markdown("---")

    st.markdown("**PROJECT STATUS**")

    st.markdown(
        """
        <div style="
            background:#102b35;
            border:1px solid #236052;
            padding:12px;
            border-radius:10px;
            color:#65e4b2;
            font-size:12px;
            font-weight:700;">
            ● MODEL-BASED DETECTION
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("Academic Project")
    st.caption("RoadGuard AI © 2026")


# =========================================================
# DASHBOARD PAGE
# =========================================================

if st.session_state.page == "Dashboard":

    st.markdown(
        '<div class="section-heading">Road Inspection Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">AI-powered road condition analysis for safer roads.</div>',
        unsafe_allow_html=True
    )

    # Road image slider
    view = ROAD_VIEWS[st.session_state.road_view]

    st.markdown(f"""
    <div class="road-slider"
    style="
        background-image:
        linear-gradient(
            90deg,
            rgba(3,12,22,.96) 0%,
            rgba(3,12,22,.78) 55%,
            rgba(3,12,22,.30) 100%
        ),
        url('{view["image"]}');
    ">
        <div class="slider-content">
            <div class="slider-label">
                ✦ ROADGUARD AI / ROAD SHOWCASE
            </div>

            <h1>
                Smarter roads.<br>
                <span>Safer Pakistan.</span>
            </h1>

            <p>
                Intelligent road inspection powered by artificial
                intelligence and deep learning. Upload a road
                photograph to identify potential road damage.
            </p>

            <div class="slider-location">
                📍 {view["city"]} &nbsp; | &nbsp;
                {view["road"]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    previous_col, slider_info_col, next_col = st.columns([1, 5, 1])

    with previous_col:
        if st.button("❮ Previous", use_container_width=True):
            st.session_state.road_view = (
                st.session_state.road_view - 1
            ) % len(ROAD_VIEWS)
            st.rerun()

    with slider_info_col:
        st.markdown(
            f"""
            <div style="
                display:flex;
                justify-content:center;
                padding:12px;
                color:#91aabd;
                font-size:12px;">
                <span class="slider-count">
                    {st.session_state.road_view + 1:02d}
                    / {len(ROAD_VIEWS):02d}
                </span>
                &nbsp;&nbsp; PAKISTAN ROAD INFRASTRUCTURE SHOWCASE
            </div>
            """,
            unsafe_allow_html=True
        )

    with next_col:
        if st.button("Next ❯", use_container_width=True):
            st.session_state.road_view = (
                st.session_state.road_view + 1
            ) % len(ROAD_VIEWS)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Dashboard information cards
    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">🤖</div>
            <div class="card-label">AI Technology</div>
            <div class="card-value">MobileNetV2</div>
        </div>
        """, unsafe_allow_html=True)

    with card2:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">📷</div>
            <div class="card-label">Input Type</div>
            <div class="card-value">Road Image</div>
        </div>
        """, unsafe_allow_html=True)

    with card3:
        st.markdown("""
        <div class="info-card">
            <div class="card-icon">🎯</div>
            <div class="card-label">Classification</div>
            <div class="card-value">Binary</div>
        </div>
        """, unsafe_allow_html=True)

    # Image inspection section
    st.markdown(
        '<div class="section-heading">Road Image Inspection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Upload a road image and let the AI model analyze its condition.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a road image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear road photograph in JPG or PNG format."
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        preview_col, details_col = st.columns([1.2, 1])

        with preview_col:
            st.image(
                image,
                caption="Uploaded Road Image",
                use_container_width=True
            )

        with details_col:
            st.markdown("### Image Details")

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-label">FILE NAME</div>
                    <div style="color:#f1f8fc;font-weight:700;margin-top:8px;">
                        {uploaded_file.name}
                    </div>
                    <br>
                    <div class="card-label">IMAGE DIMENSIONS</div>
                    <div style="color:#f1f8fc;font-weight:700;margin-top:8px;">
                        {image.width} × {image.height}
                    </div>
                    <br>
                    <div class="card-label">IMAGE FORMAT</div>
                    <div style="color:#f1f8fc;font-weight:700;margin-top:8px;">
                        {image.format or "Unknown"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        analyze_col, clear_col = st.columns([2, 1])

        with analyze_col:
            analyze_button = st.button(
                "🔍 Analyze Road Condition",
                use_container_width=True
            )

        with clear_col:
            clear_button = st.button(
                "Clear Result",
                use_container_width=True
            )

        if clear_button:
            st.session_state.prediction_result = None
            st.rerun()

        if analyze_button:

            try:
                with st.spinner("AI model is analyzing the road image..."):

                    result = predict_image(image)

                    st.session_state.prediction_result = result

            except FileNotFoundError as error:
                st.error(str(error))

            except Exception as error:
                st.error(f"Prediction failed: {error}")

        # Prediction result
        if st.session_state.prediction_result is not None:

            result = st.session_state.prediction_result

            st.markdown("---")
            st.markdown(
                '<div class="section-heading">Analysis Report</div>',
                unsafe_allow_html=True
            )

            if result["status"] == "damage":

                st.markdown(
                    f"""
                    <div class="result-panel result-damage">
                        <div style="color:#ff8585;font-size:12px;font-weight:800;letter-spacing:2px;">
                            ⚠ ROAD CONDITION RESULT
                        </div>
                        <div class="result-title" style="color:#ff8585;">
                            {result["label"]}
                        </div>
                        <div class="result-description">
                            The model classified this image as a potentially
                            damaged road. Further inspection may be required.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-panel result-normal">
                        <div style="color:#55e6ad;font-size:12px;font-weight:800;letter-spacing:2px;">
                            ✓ ROAD CONDITION RESULT
                        </div>
                        <div class="result-title" style="color:#55e6ad;">
                            {result["label"]}
                        </div>
                        <div class="result-description">
                            The model classified this image as a normal road.
                            This result is based on the model's prediction.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-label">PREDICTION CONFIDENCE</div>
                        <div class="card-value">{result["confidence"]:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric2:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-label">MODEL SCORE</div>
                        <div class="card-value">{result["score"]:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with metric3:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="card-label">INFERENCE TIME</div>
                        <div class="card-value">{result["inference_time"]:.3f}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### Prediction Confidence")

            st.progress(
                min(max(result["confidence"] / 100, 0.0), 1.0)
            )

            st.caption(
                "Confidence is calculated from the model's binary output. "
                "It should not be treated as a guarantee of actual road condition."
            )

    else:
        st.info("Upload a road image to start the inspection.")

    st.markdown(
        '<div class="footer">ROADGUARD AI • INTELLIGENT ROAD INSPECTION • ACADEMIC PROJECT</div>',
        unsafe_allow_html=True
    )


# =========================================================
# HOW IT WORKS PAGE
# =========================================================

elif st.session_state.page == "How It Works":

    st.markdown("# How RoadGuard AI Works")

    st.markdown(
        "RoadGuard AI uses a deep learning model to classify road images."
    )

    st.markdown("---")

    steps = [
        (
            "01",
            "Upload Road Image",
            "The user uploads a road photograph through the application."
        ),
        (
            "02",
            "Image Preprocessing",
            "The image is converted to RGB, resized to 224 × 224 pixels, "
            "and prepared using MobileNetV2 preprocessing."
        ),
        (
            "03",
            "AI Model Analysis",
            "The trained MobileNetV2-based model processes the image "
            "and generates a prediction score."
        ),
        (
            "04",
            "Classification",
            "The application compares the model score with the configured "
            "threshold to determine the predicted road category."
        ),
        (
            "05",
            "Display Results",
            "The dashboard displays the predicted class, confidence score, "
            "model score, and inference time."
        )
    ]

    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="info-card" style="margin-bottom:15px;">
                <div style="color:#36d9e8;font-size:13px;font-weight:800;">
                    STEP {number}
                </div>
                <div style="font-size:20px;font-weight:800;color:#f1f8fc;margin:8px 0;">
                    {title}
                </div>
                <div style="color:#a9bdcc;font-size:14px;line-height:1.7;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# ABOUT PROJECT PAGE
# =========================================================

elif st.session_state.page == "About Project":

    st.markdown("# About RoadGuard AI")

    st.markdown(
        """
        RoadGuard AI is an academic project designed to demonstrate
        how artificial intelligence and deep learning can support
        road condition inspection.

        The application analyzes uploaded road photographs and
        classifies them into two categories based on the trained
        model's output.
        """
    )

    st.markdown("---")

    st.markdown("### Project Objectives")

    objectives = [
        "Demonstrate image classification using deep learning.",
        "Explore the use of AI in road condition inspection.",
        "Provide a simple interface for uploading and analyzing images.",
        "Display prediction results in an understandable format.",
        "Show how computer vision can support infrastructure monitoring."
    ]

    for objective in objectives:
        st.markdown(f"- {objective}")

    st.markdown("### Project Scope")

    st.info(
        "This is an academic prototype. Its predictions depend on "
        "the training data and performance of the model. It does not "
        "replace professional road inspection or on-site verification."
    )


# =========================================================
# MODEL INFORMATION PAGE
# =========================================================

elif st.session_state.page == "Model Information":

    st.markdown("# Model Information")

    st.markdown(
        "Technical information about the model used in RoadGuard AI."
    )

    st.markdown("---")

    model_details = [
        ("Model Architecture", "MobileNetV2"),
        ("Model File", "roadguard_final.keras"),
        ("Input Image Size", "224 × 224 pixels"),
        ("Input Color Format", "RGB"),
        ("Preprocessing", "MobileNetV2 preprocess_input"),
        ("Classification Type", "Binary Classification"),
        ("Decision Threshold", str(THRESHOLD)),
        ("Model Loading", "TensorFlow / Keras"),
        ("Output", "Single prediction score")
    ]

    for label, value in model_details:

        left, right = st.columns([1, 2])

        with left:
            st.markdown(
                f"**{label}**"
            )

        with right:
            st.markdown(
                f"<span style='color:#36d9e8;font-weight:600;'>{value}</span>",
                unsafe_allow_html=True
            )

        st.markdown("---")

    st.markdown("### Model File Status")

    if MODEL_PATH.exists():
        st.success("Model file found: roadguard_final.keras")
    else:
        st.error(
            "Model file not found. Place roadguard_final.keras "
            "in the same folder as app.py."
        )

    st.markdown(
        """
        **Note:** The model's output interpretation depends on how
        its binary classes were encoded during training. Confirm that
        a higher output score represents the damaged-road class before
        using this application for evaluation.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ROADGUARD AI © 2026 — BUILT FOR ACADEMIC AND DEMONSTRATION PURPOSES
    </div>
    """,
    unsafe_allow_html=True
)
