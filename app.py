import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import zipfile
import tempfile

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
# STYLE
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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#00F5FF;'>🧠 MetaPrep Multimodal AI Console</h1>", unsafe_allow_html=True)
st.divider()

# =====================================
# SESSION STATE
# =====================================
for key in ["dataset","image_folder","processed","strategy","confidence"]:
    if key not in st.session_state:
        st.session_state[key] = None

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

                if file.lower().endswith((".jpg",".jpeg",".png")):
                    image_folder = root

            for d in dirs:
                if d.lower() in ["images","image","imgs","pictures"]:
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
            image_profile = {"blur_score":1.0,"noise_score":1.0,"resolution_score":1.0}

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
            strategy["OpenCV Image Enhancement"] = True

        # Numeric strategy trigger
        if len(st.session_state.dataset.select_dtypes(include=np.number).columns) > 0:
            strategy["Numeric Processing"] = True

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
if st.button("🚀 Execute Pipeline"):

    df = st.session_state.dataset.copy()
    logs = []

    # -----------------------------
    # PRINT STRATEGY ENGINE OUTPUT
    # -----------------------------
    logs.append("🔍 Strategy Engine Output")
    logs.append(f"Confidence: {st.session_state.confidence}")
    logs.append("Selected Actions:")

    for action in st.session_state.strategy:
        logs.append(f" - {action}")

    logs.append("-----------------------------------")

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
# =====================================
# TAB 4 – ANALYTICS
# =====================================
with tab4:

    if st.session_state.processed is not None:

        df_before = st.session_state.dataset
        df_after = st.session_state.processed

        text_before = profile_text_dataset(df_before)
        text_after = profile_text_dataset(df_after)

        if st.session_state.image_folder:
            image_profile = profile_images(st.session_state.image_folder)
        else:
            image_profile = {"blur_score":1.0,"noise_score":1.0,"resolution_score":1.0}

        dqi_before = DataQualityIndex(text_before,image_profile).compute_dqi()
        dqi_after = DataQualityIndex(text_after,image_profile).compute_dqi()

        # -----------------------------
        # METRICS DISPLAY
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        col1.metric("TQ Before → After", 
                    f"{dqi_before['TQ']} → {dqi_after['TQ']}")
        col2.metric("IQ Before → After", 
                    f"{dqi_before['IQ']} → {dqi_after['IQ']}")
        col3.metric("DQI Before → After", 
                    f"{dqi_before['DQI']} → {dqi_after['DQI']}")

        # -----------------------------
        # BAR CHART (ONLY TQ, IQ, DQI)
        # -----------------------------
        categories = ["TQ", "IQ", "DQI"]

        before_values = [
            dqi_before["TQ"],
            dqi_before["IQ"],
            dqi_before["DQI"]
        ]

        after_values = [
            dqi_after["TQ"],
            dqi_after["IQ"],
            dqi_after["DQI"]
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=categories,
            y=before_values,
            name="Before",
            opacity=0.7
        ))

        fig.add_trace(go.Bar(
            x=categories,
            y=after_values,
            name="After",
            opacity=0.7
        ))

        fig.update_layout(
            title="Quality Improvement (TQ, IQ, DQI)",
            barmode="group",
            yaxis=dict(range=[0,1]),
            xaxis_title="Metric",
            yaxis_title="Score"
        )

        st.plotly_chart(fig, use_container_width=True)