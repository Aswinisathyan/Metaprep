import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="MetaPrep AI Console",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# KEEPING YOUR ORIGINAL STYLE (UNCHANGED)
# =====================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
    color: white;
}
.stApp {
    background: linear-gradient(-45deg, #050816, #0a0f2c, #001f3f, #050816);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.typewriter {
    overflow: hidden;
    border-right: .15em solid #00F5FF;
    white-space: nowrap;
    animation: typing 3s steps(40, end), blink-caret .75s step-end infinite;
}
@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}
@keyframes blink-caret {
    from, to { border-color: transparent }
    50% { border-color: #00F5FF; }
}
.glow-text { animation: glow 2s infinite alternate; }
@keyframes glow {
    from { text-shadow: 0 0 5px #00F5FF; }
    to { text-shadow: 0 0 20px #00F5FF; }
}
.slide-in { animation: slideIn 1s ease forwards; }
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-40px); }
    to { opacity: 1; transform: translateX(0); }
}
.neon-box {
    background: rgba(0,255,255,0.05);
    border: 1px solid #00F5FF;
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 0 15px #00F5FF44;
    margin-bottom: 20px;
}
.log-box {
    background: black;
    border: 1px solid #00F5FF;
    padding: 15px;
    height: 230px;
    overflow-y: auto;
    border-radius: 10px;
    font-size: 13px;
}
.stButton>button {
    background: #00F5FF;
    color: black;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown("<h1 class='typewriter glow-text'>🧠 MetaPrep AI Console</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='slide-in glow-text'>AI-Driven Multimodal Strategy Prediction & Validation Framework</h3>", unsafe_allow_html=True)
st.divider()

# =====================================
# SESSION STATE
# =====================================
for key in ["dataset", "processed", "strategy", "logs", "confidence"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.logs is None:
    st.session_state.logs = []

# =====================================
# ADAPTIVE QUALITY VECTOR
# =====================================

def compute_quality_vector(df):

    total_cells = df.size
    rows = len(df)

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    # Completeness
    C = 1 - missing/(total_cells + 1)

    # Consistency
    S = 1 - duplicates/(rows + 1)

    # Text Quality (abnormal characters ratio)
    text_cols = df.select_dtypes(include="object")

    if not text_cols.empty:
        abnormal_chars = text_cols.apply(
            lambda col: col.astype(str).str.count(r'[^\x00-\x7F]')
        ).sum().sum()

        total_text_chars = text_cols.apply(
            lambda col: col.astype(str).str.len()
        ).sum().sum()

        abnormal_ratio = abnormal_chars/(total_text_chars+1)
        TQ = 1 - abnormal_ratio
    else:
        TQ = 1

    # Placeholder image quality (upgrade when ESRGAN integrated)
    IQ = 0.85

    return C, S, TQ, IQ


def compute_adaptive_dqi(C, S, TQ, IQ):
    return round(
        0.3*C +
        0.2*S +
        0.25*TQ +
        0.25*IQ,
        3
    )


def adaptive_pipeline_predictor(C, S, TQ, IQ):

    strategy = {}

    if C < 0.90:
        strategy["Fill Missing"] = True

    if S < 0.95:
        strategy["Remove Duplicates"] = True

    if TQ < 0.85:
        strategy["Text Normalization"] = True

    if IQ < 0.80:
        strategy["GAN Image Enhancement"] = True

    if not strategy:
        strategy["Dataset Quality Satisfactory"] = True

    ideal = np.array([1,1,1,1])
    actual = np.array([C,S,TQ,IQ])
    deviation = np.mean(np.abs(ideal - actual))

    confidence = round(1 - deviation, 2)

    return strategy, confidence


# =====================================
# TABS
# =====================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Upload ✨",
    "🧠 Strategy ⚡",
    "⚙ Execution 🚀",
    "📊 Analytics 📈"
])

# =====================================
# TAB 1 - UPLOAD
# =====================================
with tab1:

    st.markdown('<div class="neon-box slide-in"><h3 class="glow-text">📂 Upload Dataset</h3></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state.dataset = df
        st.session_state.processed = None
        st.success("Dataset Loaded Successfully")
        st.dataframe(df, use_container_width=True)


# =====================================
# TAB 2 - STRATEGY (MODEL-LIKE)
# =====================================
with tab2:

    if st.session_state.dataset is not None:

        st.markdown('<div class="neon-box slide-in"><h3 class="glow-text">🧠 Adaptive Pipeline Prediction</h3></div>', unsafe_allow_html=True)

        C, S, TQ, IQ = compute_quality_vector(st.session_state.dataset)

        strategy, confidence = adaptive_pipeline_predictor(C, S, TQ, IQ)

        st.session_state.strategy = strategy
        st.session_state.confidence = confidence

        # Confidence Ring
        fig = go.Figure(go.Pie(
            values=[confidence, 1-confidence],
            hole=0.75,
            marker_colors=["#00F5FF", "#111"],
            textinfo="none"
        ))

        fig.update_layout(
            annotations=[dict(
                text=f"{confidence*100:.1f}%",
                x=0.5, y=0.5,
                font_size=28,
                font_color="#00F5FF",
                showarrow=False
            )],
            showlegend=False,
            paper_bgcolor="#050816"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.json(strategy)


# =====================================
# TAB 3 - EXECUTION
# =====================================
with tab3:

    if st.session_state.dataset is not None and st.session_state.strategy is not None:

        st.markdown('<div class="neon-box slide-in"><h3 class="glow-text">⚙ Pipeline Execution</h3></div>', unsafe_allow_html=True)

        manual_mode = st.checkbox("🛠 Enable Manual Cleaning Override")

        if manual_mode:
            remove_dup = st.checkbox("Remove Duplicates")
            fill_missing = st.checkbox("Fill Missing")
            normalize_text = st.checkbox("Text Normalization")
        else:
            remove_dup = "Remove Duplicates" in st.session_state.strategy
            fill_missing = "Fill Missing" in st.session_state.strategy
            normalize_text = "Text Normalization" in st.session_state.strategy

        if st.button("🚀 Run Pipeline"):

            st.session_state.logs = []
            df = st.session_state.dataset.copy()
            progress = st.progress(0)

            steps = []

            if remove_dup:
                steps.append("Removing Duplicates")
            if fill_missing:
                steps.append("Filling Missing Values")
            if normalize_text:
                steps.append("Text Normalization")

            for i, step in enumerate(steps):
                time.sleep(0.8)
                st.session_state.logs.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {step} Completed"
                )
                progress.progress((i+1)/len(steps))

            if remove_dup:
                df = df.drop_duplicates()

            if fill_missing:
                df = df.fillna("")

            if normalize_text:
                for col in df.select_dtypes(include="object").columns:
                    df[col] = df[col].str.lower()

            st.session_state.processed = df
            st.success("Pipeline Execution Complete")

        st.markdown('<div class="log-box">', unsafe_allow_html=True)
        for log in st.session_state.logs:
            st.markdown(f"<p class='glow-text'>{log}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =====================================
# TAB 4 - ANALYTICS (ADAPTIVE DQI)
# =====================================
with tab4:

    if st.session_state.processed is not None:

        df_before = st.session_state.dataset
        df_after = st.session_state.processed

        C_before, S_before, TQ_before, IQ_before = compute_quality_vector(df_before)
        C_after, S_after, TQ_after, IQ_after = compute_quality_vector(df_after)

        dqi_before = compute_adaptive_dqi(C_before, S_before, TQ_before, IQ_before)
        dqi_after = compute_adaptive_dqi(C_after, S_after, TQ_after, IQ_after)

        categories = ["Completeness", "Consistency", "Text Quality", "Image Quality"]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[C_before, S_before, TQ_before, IQ_before],
            theta=categories,
            fill='toself',
            name='Before',
            line=dict(color="#FF4C4C")
        ))

        fig.add_trace(go.Scatterpolar(
            r=[C_after, S_after, TQ_after, IQ_after],
            theta=categories,
            fill='toself',
            name='After',
            line=dict(color="#00F5FF")
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#00F5FF")),
            paper_bgcolor="#050816",
            font=dict(color="#00F5FF")
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Adaptive DQI Before", dqi_before)
        col2.metric("Adaptive DQI After", dqi_after)

        st.download_button(
            label="📥 Download Cleaned Dataset",
            data=df_after.to_csv(index=False),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    else:
        st.warning("Run pipeline first.")
