import joblib
import numpy as np
import os

MODEL_PATH = os.path.join("models", "metaprep_srm.pkl")

model = joblib.load(MODEL_PATH)

LABELS = [
    "Fill Missing",
    "Remove Duplicates",
    "Advanced Text Cleaning",
    "OpenCV Image Enhancement"
]

def predict_strategy(C, S, TQ, IQ, num_numeric_cols):

    features = np.array([[C, S, TQ, IQ, num_numeric_cols]])

    prediction = model.predict(features)[0]

    strategy = {}

    for i, val in enumerate(prediction):
        if val == 1:
            strategy[LABELS[i]] = True

    if not strategy:
        strategy["Dataset Quality Satisfactory"] = True

    confidence = float(np.mean(prediction))

    return strategy, confidence