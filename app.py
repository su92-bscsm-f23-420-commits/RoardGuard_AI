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
    <div class="hero" style="background-image:linear-gradient(90deg,rgba(3,12,22,.96),rgba(3,12,22,.72) 58%,rgba(3,12,22,.25)),url('{view[2]}')">
      <div class="eyebrow">✦ AI-POWERED ROAD INTELLIGENCE</div>
      <h1>Smarter roads.<br><span>Safer journeys.</span></h1>
      <p>Upload a road image and let our trained deep-learning classifier assess whether the image shows road damage.</p>
      <div class="hero-foot">SCENIC THEME · {view[0].upper()} &nbsp; / &nbsp; {view[1].upper()}</div>
    </div>
    """, unsafe_allow_html=True)
    ch1, ch2, _ = st.columns([1.25, 1.35, 3.4])
    with ch1:
        if st.button("↻ Change road background", use_container_width=True):
            st.session_state.road_view = random.choice([i for i in range(len(ROAD_VIEWS)) if i != st.session_state.road_view])
            st.rerun()
    with ch2:
        st.caption("Background is decorative only.")
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
                    st.markdown(f'<div class="result-damage"><div class="eyebrow" style="color:#ff9b9b">⚠ CLASSIFICATION RESULT</div><h2 style="margin:12px 0;color:#fff">Road damage detected</h2><p>The trained model classified this image as the damage class.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-safe"><div class="eyebrow" style="color:#c7f36b">✓ CLASSIFICATION RESULT</div><h2 style="margin:12px 0;color:#fff">No road damage detected</h2><p>The trained model classified this image as the no-damage class.</p></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="panel"><div class="section-kicker">INFERENCE CONFIGURATION</div><h3>Runtime details</h3><b>Framework</b><p class="muted">TensorFlow / Keras</p><b>Preprocessing</b><p class="muted">MobileNetV2 preprocess_input</p><b>Decision threshold</b><p class="muted">0.50</p><b>Model file</b><p class="muted">roadguard_final.keras</p></div>', unsafe_allow_html=True)
    st.warning("The interface assumes the model's sigmoid output corresponds to class 1 = Road Damage and class 0 = No Road Damage. Confirm this mapping against the original training labels before final submission. Model scores are not calibrated probabilities.")

st.markdown('<div class="footer"><b>ROADGUARD AI</b> · DEEP LEARNING PROJECT · MOBILENETV2<br>Academic prototype — predictions should be reviewed by a human.</div>', unsafe_allow_html=True)
