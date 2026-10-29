import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template_string
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "roadguard_final.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>RoadGuard AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #101827;
            color: white;
            text-align: center;
            padding: 35px 15px;
        }
        .card {
            max-width: 520px;
            margin: 30px auto;
            padding: 28px;
            background: #1e293b;
            border-radius: 16px;
        }
        input, button {
            margin: 15px 0;
            padding: 12px;
            max-width: 100%;
        }
        button {
            background: #38bdf8;
            border: 0;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
        }
        .result { color: #7dd3fc; font-size: 22px; }
        .note { color: #cbd5e1; font-size: 14px; }
    </style>
</head>
<body>
    <h1>RoadGuard AI</h1>
    <p>AI-powered Road Damage Classification</p>
    <div class="card">
        <h2>Upload a Road Image</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image"
                   accept="image/*" required>
            <br>
            <button type="submit">Analyze Road</button>
        </form>
        {% if result %}
            <h2 class="result">{{ result }}</h2>
            <p>Model damage score: {{ score }}</p>
        {% endif %}
        {% if error %}
            <p>{{ error }}</p>
        {% endif %}
        <p class="note">
            This is an academic prototype. Predictions may be incorrect
            and should be reviewed by a person.
        </p>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    score_text = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("image")

        if not uploaded or not uploaded.filename:
            error = "Please select an image."
        else:
            try:
                image = Image.open(uploaded.stream).convert("RGB")
                image = image.resize((224, 224))

                array = np.asarray(image, dtype=np.float32)
                batch = preprocess_input(
                    np.expand_dims(array, axis=0)
                )

                score = float(model.predict(batch, verbose=0)[0][0])

                if score >= 0.5:
                    result = "ROAD DAMAGE DETECTED"
                else:
                    result = "NO ROAD DAMAGE DETECTED"

                score_text = f"{score:.2%} (not a calibrated probability)"

            except Exception:
                error = "Unable to process this image. Please try another."

    return render_template_string(
        PAGE, result=result, score=score_text, error=error
    )
