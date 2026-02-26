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
    




    # -----------------------------
    # EXECUTE STEPS
    # -----------------------------
    for step in st.session_state.strategy:

        if step == "Remove Duplicates":
            df = df.drop_duplicates()
            logs.append("✔ Duplicates Removed")

        if step == "Fill Missing":
            df = df.fillna("")
            logs.append("✔ Missing Values Filled")

        if step == "Advanced Text Cleaning":
            df = clean_text_dataset(df)
            logs.append("✔ Text Normalization Applied")

        if step == "Numeric Processing":
            df, pipeline_used, nq_score = numeric_agent(df)

            logs.append("📊 Numeric Agent Output")
            logs.append(f"Pipeline Selected: {pipeline_used}")
            logs.append(f"Numeric Quality Score: {nq_score}")
            logs.append("-----------------------------------")

        if step == "OpenCV Image Enhancement":

            if st.session_state.image_folder:

                result = enhance_images(
                    st.session_state.image_folder,
                    os.path.join(st.session_state.image_folder,"enhanced")
                )

                if result["status"] == "success":

                    st.session_state.image_folder = result["enhanced_folder"]

                    logs.append("🖼 Image Agent Output")
                    logs.append(f"IQ Before: {result['IQ_before']}")
                    logs.append(f"IQ After: {result['IQ_after']}")
                    logs.append(f"Improvement: {result['improvement']}")
                    logs.append("-----------------------------------")

    st.session_state.processed = df
    st.success("Pipeline Execution Complete")

    st.markdown("### 🧠 Model Output Console")
    for log in logs:
        st.write(log)

    st.dataframe(df)