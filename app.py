
import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = os.environ.get("MODEL_PATH", "roadguard_final.keras")
model = tf.keras.models.load_model(MODEL_PATH)

THRESHOLD = 0.50

def predict_road(image):
    if image is None:
        return {"Please upload a road image": 1.0}

    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32)
    batch = preprocess_input(np.expand_dims(arr, axis=0))
    score = float(model.predict(batch, verbose=0)[0][0])

    label = "ROAD DAMAGE" if score >= THRESHOLD else "NO ROAD DAMAGE DETECTED"

    return {
        label: 1.0,
        f"Model damage score (not a calibrated probability): {score:.2%}": 0.0
    }

demo = gr.Interface(
    fn=predict_road,
    inputs=gr.Image(type="pil", label="Upload a road image"),
    outputs=gr.Label(label="RoadGuard AI Result"),
    title="RoadGuard AI",
    description="Upload a road image for classification. Predictions may be incorrect."
)

if __name__ == "__main__":
    demo.launch()
