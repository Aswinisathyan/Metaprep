import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import os
import zipfile
import tempfile
from datetime import datetime

from src.profiling import profile_text_dataset, profile_images
from src.dqi import DataQualityIndex
from src.text_cleaning import clean_text_dataset
from src.image_enhancement import enhance_images
from src.numeric_agent import numeric_agent

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="MetaPrep AI Console",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# ANIMATED STYLE
# =====================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg,#050816,#0a0f2c,#001f3f,#050816);
    background-size:400% 400%;
    animation:gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
.neon-box{
    background:rgba(0,255,255,0.05);
    border:1px solid #00F5FF;
    border-radius:15px;
    padding:25px;
    box-shadow:0 0 25px #00F5FF55;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#00F5FF;'>🧠 MetaPrep Multimodal AI Console</h1>", unsafe_allow_html=True)
st.divider()

# =====================================
# SESSION STATE
# =====================================
for key in ["dataset","image_folder","processed","strategy","confidence","logs"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.logs is None:
    st.session_state.logs = []

# =====================================
# TABS
# =====================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Upload ZIP",
    "🧠 Strategy",
    "⚙ Execution",
    "📊 Analytics"
])

# =====================================
# TAB 1 – ZIP UPLOAD
# =====================================
with tab1:

    uploaded_zip = st.file_uploader("Upload Multimodal ZIP", type=["zip"])

    if uploaded_zip:

        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "dataset.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        csv_file = None
        image_folder = None

        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(".csv"):
                    csv_file = os.path.join(root, file)

            for d in dirs:
                if d.lower() == "images":
                    image_folder = os.path.join(root, d)

        if csv_file:
            df = pd.read_csv(csv_file)
            st.session_state.dataset = df
            st.session_state.image_folder = image_folder
            st.success("ZIP Extracted Successfully")
            st.dataframe(df)
        else:
            st.error("No CSV file found inside ZIP.")

# =====================================
# TAB 2 – STRATEGY
# =====================================
with tab2:

    if st.session_state.dataset is not None:

        text_profile = profile_text_dataset(st.session_state.dataset)

        if st.session_state.image_folder:
            image_profile = profile_images(st.session_state.image_folder)
        else:
            image_profile = {
                "blur_score":1.0,
                "noise_score":1.0,
                "resolution_score":1.0
            }

        metrics = DataQualityIndex(text_profile, image_profile).compute_dqi()

        C,S,TQ,IQ = metrics["C"],metrics["S"],metrics["TQ"],metrics["IQ"]

        strategy = {}

        if C < 0.90:
            strategy["Fill Missing"] = True
        if S < 0.95:
            strategy["Remove Duplicates"] = True
        if TQ < 0.90:
            strategy["Advanced Text Cleaning"] = True
        if IQ < 0.80:
            strategy["GAN Image Enhancement"] = True

        if not strategy:
            strategy["Dataset Quality Satisfactory"] = True

        ideal = np.array([1,1,1,1])
        actual = np.array([C,S,TQ,IQ])
        confidence = round(1 - np.mean(np.abs(ideal-actual)),2)

        st.session_state.strategy = strategy
        st.session_state.confidence = confidence

        st.metric("Strategy Confidence", f"{confidence*100:.1f}%")
        st.json(strategy)

# =====================================
# TAB 3 – EXECUTION
# =====================================
with tab3:

    if st.session_state.strategy:

        if st.button("🚀 Execute Agents"):

            df = st.session_state.dataset.copy()
            logs = []

            for step in st.session_state.strategy:

                # -------------------------
                # TEXT AGENT
                # -------------------------
                if step == "Remove Duplicates":
                    df = df.drop_duplicates()
                    logs.append("✔ Duplicates Removed")

                if step == "Fill Missing":
                    df = df.fillna("")
                    logs.append("✔ Missing Values Filled")

                if step == "Advanced Text Cleaning":
                    df = clean_text_dataset(df)
                    logs.append("✔ Text Normalization Applied")

                # -------------------------
                # IMAGE ENHANCEMENT AGENT
                # -------------------------
                if step == "Image Enhancement":

                    if st.session_state.image_folder:

                        result = enhance_images(
                            st.session_state.image_folder,
                            os.path.join(
                                st.session_state.image_folder,
                                "enhanced"
                            )
                        )

                        if result["status"] == "success":
                            st.session_state.image_folder = result["enhanced_folder"]

                            logs.append(
                                f"✔ Image Enhanced | IQ Improvement: {result['improvement']}"
                            )
                        else:
                            logs.append("⚠ Image Enhancement Failed")
                        if step == "Numeric Processing":

                            df, pipeline_used, nq_score = numeric_agent(df)
                            logs.append(f"✔ Numeric Agent Applied | Pipeline: {pipeline_used} | NQ: {nq_score}")

            st.session_state.processed = df
            st.dataframe(st.session_state.processed)
            st.success("Pipeline Execution Complete")

            for log in logs:
                st.write(log)

# =====================================
# TAB 4 – ANALYTICS
# =====================================
with tab4:

    if st.session_state.processed is not None:

        df_before = st.session_state.dataset
        df_after = st.session_state.processed

        text_before = profile_text_dataset(df_before)
        text_after = profile_text_dataset(df_after)

        image_profile = profile_images(st.session_state.image_folder)

        dqi_before = DataQualityIndex(text_before,image_profile).compute_dqi()
        dqi_after = DataQualityIndex(text_after,image_profile).compute_dqi()

        improvement = round(dqi_after["DQI"] - dqi_before["DQI"],3)

        st.metric("DQI Before", dqi_before["DQI"])
        st.metric("DQI After", dqi_after["DQI"])
        st.metric("Improvement", improvement)

        categories=["C","S","TQ","IQ"]

        fig=go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[dqi_before["C"],dqi_before["S"],dqi_before["TQ"],dqi_before["IQ"]],
            theta=categories,
            fill='toself',
            name="Before"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[dqi_after["C"],dqi_after["S"],dqi_after["TQ"],dqi_after["IQ"]],
            theta=categories,
            fill='toself',
            name="After"
        ))

        st.plotly_chart(fig)
