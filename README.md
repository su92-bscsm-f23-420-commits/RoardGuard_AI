# RoadGuard AI

A deep-learning prototype that classifies road images as ROAD DAMAGE or
NO ROAD DAMAGE DETECTED using a fine-tuned MobileNetV2 model.

## Run locally
1. Install the required packages using:
   pip install -r requirements.txt
2. Keep roadguard_final.keras in the same folder as app.py.
3. Run:
   python app.py

## Technical details
- Input image: RGB, resized to 224x224
- Preprocessing: MobileNetV2 preprocess_input
- Dataset: RDD2020 subset
- Output: binary classification score

## Limitations
This is an academic prototype. Predictions can be wrong and require human review.
NO ROAD DAMAGE DETECTED does not guarantee a road is damage-free.
The displayed score is not a calibrated probability.
