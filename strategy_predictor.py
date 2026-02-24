import joblib
import numpy as np


class StrategyPredictor:

    def __init__(self, model_path="models/metaprep_srm.pkl"):
        self.models = joblib.load(model_path)

    def predict(self, metadata_vector):

        X = np.array(metadata_vector).reshape(1, -1)

        predictions = {}

        for name, model in self.models.items():
            prob = model.predict_proba(X)[0][1]

            predictions[name] = {
                "prediction": int(prob > 0.5),
                "confidence": round(float(prob), 3)
            }

        return predictions