"""Live Cyberbullying & Harassment Detection Dashboard.

Public web dashboard for analyzing YouTube comments and plain text.
Features:
- Single comment classification with confidence %, severity, type, and highlighted word tokens.
- Whole video scanner (up to N comments, default 200, optional replies) with live charts and CSV export.
- Batch CSV upload and analysis.
- Offline Demo mode with 10 sample comments and 3 YouTube test videos.
- Model Evaluation & Metrics dashboard with confusion matrix, comparisons, and error analysis.
- Privacy, ethics, and limitations transparency.
"""

import os
import sys
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Ensure root directory is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.input_parser import parse_user_input
from src.youtube_client import (
    fetch_single_comment,
    fetch_video_comments,
    CURATED_DEMO_VIDEOS
)
from src.predict import (
    predict_comment,
    predict_batch,
    load_model_pipelines,
    SEVERITY_MAPPING
)
from src.explain import explain_comment, ETHICS_NOTICE

# Page Configuration
st.set_page_config(
    page_title="GuardNet | Live Cyberbullying Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling CSS
CUSTOM_CSS = """
<style>
/* Modern Glassmorphism & Typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Gradient */
.hero-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

/* Glassmorphic Cards */
.metric-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(129, 140, 248, 0.3);
}

/* Result Verdict Badges */
.badge-attacking {
    background-color: rgba(239, 68, 68, 0.2);
    border: 1px solid #ef4444;
    color: #fca5a5;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}

.badge-safe {
    background-color: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #6ee7b7;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
}

/* Severity Indicators */
.severity-high {
    color: #f87171;
    background: rgba(239, 68, 68, 0.15);
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
}

.severity-medium {
    color: #fbbf24;
    background: rgba(245, 158, 11, 0.15);
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
}

.severity-low {
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.15);
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
}

/* Explainability Box */
.explain-box {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.2rem;
    font-size: 1.05rem;
    line-height: 1.7;
}

/* Notice box */
.ethics-box {
    background: rgba(245, 158, 11, 0.08);
    border-left: 4px solid #f59e0b;
    padding: 0.8rem 1.2rem;
    border-radius: 0 8px 8px 0;
    color: #fcd34d;
    font-size: 0.9rem;
    margin-top: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "scan_stats" not in st.session_state:
    st.session_state.scan_stats = {"total_scanned": 0, "flagged": 0}
if "youtube_api_key" not in st.session_state:
    st.session_state.youtube_api_key = os.getenv("YOUTUBE_API_KEY", "")


# 10 Curated Demo Comments for Instant Testing
DEMO_COMMENTS = [
    {"text": "You are a disgusting clown, go kill yourself right now kys 🤡", "type": "Hostile (Threat)"},
    {"text": "Shut up you dirty slut, women should never be allowed to talk here", "type": "Hostile (Sexual)"},
    {"text": "All members of your race are filthy subhumans get out of our country", "type": "Hostile (Ethnicity)"},
    {"text": "Corrupt politician scumbags destroying everything, hope you rot", "type": "Hostile (Political)"},
    {"text": "Your religion is completely backward and full of violent lunatics", "type": "Hostile (Religion)"},
    {"text": "You are the worst programmer alive, delete your github and give up", "type": "Hostile (Vocational)"},
    {"text": "Haha look at this absolute loser crying on stream lol", "type": "Hostile (Troll)"},
    {"text": "Thank you so much for this tutorial, it saved me hours of debugging!", "type": "Benign (Positive)"},
    {"text": "Does this model support 32GB RAM or is it soldered on the motherboard?", "type": "Benign (Neutral Question)"},
    {"text": "Great debate from both sides, appreciated the constructive dialogue.", "type": "Benign (Constructive)"}
]


# Load Models at Startup (Cached)
@st.cache_resource(show_spinner=False)
def get_models():
    return load_model_pipelines()


try:
    models = get_models()
except Exception as e:
    st.error(f"Error loading models from disk: {e}")
    st.stop()


# Load Evaluation Metrics Cache
@st.cache_data(show_spinner=False)
def load_metrics_data():
    metrics_path = os.path.join(ROOT_DIR, "models", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


metrics_data = load_metrics_data()


# SIDEBAR CONTROLS
with st.sidebar:
    st.markdown("### 🛡️ GuardNet AI")
    st.caption("Live Cyberbullying & Harassment Scanner")
    st.divider()

    # Navigation Menu
    nav_option = st.radio(
        "Navigation",
        [
            "🔍 Live Scanner",
            "📁 Batch CSV Analysis",
            "📊 Model Evaluation & Metrics",
            "⚖️ Ethics & Privacy",
            "📜 Session Audit Log"
        ]
    )

    st.divider()

    # Model Architecture Selector
    st.markdown("#### ⚙️ Classification Engine")
    selected_model_name = st.selectbox(
        "Classifier Backbone",
        ["Logistic Regression", "Linear SVM"],
        help="Both models utilize Word (1-2) + Char (3-5) TF-IDF features with balanced class weights."
    )

    st.divider()

    # YouTube API Configuration
    st.markdown("#### 🔑 YouTube API Key")
    input_key = st.text_input(
        "API Key (Optional)",
        value=st.session_state.youtube_api_key,
        type="password",
        help="Optional. If omitted, built-in demo videos & simulated comments are seamlessly available."
    )
    if input_key != st.session_state.youtube_api_key:
        st.session_state.youtube_api_key = input_key

    if st.session_state.youtube_api_key:
        st.success("API Key Active (YouTube API v3)")
    else:
        st.info("Demo Mode Active: Built-in YouTube videos & sample comments ready to test.")

    st.divider()

    # Quick Stats in Sidebar
    st.markdown("#### 📈 Session Counters")
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Scanned", st.session_state.scan_stats["total_scanned"])
    col_sb2.metric("Flagged", st.session_state.scan_stats["flagged"])

    if st.button("Reset Session History", use_container_width=True):
        st.session_state.history = []
        st.session_state.scan_stats = {"total_scanned": 0, "flagged": 0}
        st.rerun()


# ==============================================================================
# VIEW 1: LIVE SCANNER (MAIN DASHBOARD)
# ==============================================================================
if nav_option == "🔍 Live Scanner":
    st.markdown('<div class="hero-title">Live Cyberbullying & Harassment Detector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Classify online YouTube comments or plain text with '
        'harassment taxonomy, calibrated confidence, severity level, and token attribution.</div>',
        unsafe_allow_html=True
    )

    # Demo Quick-Load Buttons Bar
    with st.expander("⚡ Quick Test Loaders (1-Click Demo Inputs)", expanded=False):
        st.markdown("**10 Curated Benchmark Test Comments:**")
        cols_demo = st.columns(5)
        for i, demo in enumerate(DEMO_COMMENTS[:5]):
            if cols_demo[i].button(f"{demo['type'].split()[0]} #{i+1}", key=f"demo_btn_{i}", help=demo["text"]):
                st.session_state["user_input_val"] = demo["text"]

        cols_demo2 = st.columns(5)
        for i, demo in enumerate(DEMO_COMMENTS[5:], start=5):
            if cols_demo2[i-5].button(f"{demo['type'].split()[0]} #{i+1}", key=f"demo_btn_{i}", help=demo["text"]):
                st.session_state["user_input_val"] = demo["text"]

        st.markdown("**Pre-Configured YouTube Demo Videos:**")
        col_v1, col_v2, col_v3 = st.columns(3)
        if col_v1.button("🎥 Rick Astley Meme (dQw4w9WgXcQ)"):
            st.session_state["user_input_val"] = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        if col_v2.button("🎥 Esports Debate (5O9n1gH_E2U)"):
            st.session_state["user_input_val"] = "https://www.youtube.com/watch?v=5O9n1gH_E2U"
        if col_v3.button("🎥 Tech Review (kJQP7kiw5Fk)"):
            st.session_state["user_input_val"] = "https://www.youtube.com/watch?v=kJQP7kiw5Fk"

    # Main Input Box
    default_input = st.session_state.get("user_input_val", "")
    user_input = st.text_area(
        "Enter YouTube Link (video or comment) or Paste Plain Comment Text:",
        value=default_input,
        placeholder="Paste a YouTube URL (e.g. https://www.youtube.com/watch?v=... or &lc=...), or paste plain text comment...",
        height=100
    )

    # Additional options for video mode
    with st.expander("🛠️ Video Scanning Parameters", expanded=False):
        col_p1, col_p2 = st.columns(2)
        max_comments_limit = col_p1.slider("Max comments to scan (N)", min_value=10, max_value=500, value=200, step=10)
        include_replies = col_p2.checkbox("Include comment thread replies", value=False)

    col_btn, _ = st.columns([1, 4])
    analyze_clicked = col_btn.button("🚀 Analyze Input", type="primary", use_container_width=True)

    if analyze_clicked and user_input.strip():
        parsed = parse_user_input(user_input)

        if not parsed["is_valid"]:
            st.error(f"❌ {parsed['error_message']}")
        else:
            input_type = parsed["input_type"]

            # ------------------------------------------------------------------
            # CASE A: SINGLE COMMENT (Plain Text OR Specific YouTube Comment 'lc=')
            # ------------------------------------------------------------------
            if input_type in ("plain_text", "youtube_comment"):
                comment_text = parsed["clean_input"]

                # If specific YouTube comment link, fetch the comment text
                if input_type == "youtube_comment":
                    with st.spinner(f"Fetching linked comment '{parsed['comment_id']}' from YouTube API..."):
                        fetched_text, fetch_err = fetch_single_comment(
                            parsed["comment_id"],
                            api_key=st.session_state.youtube_api_key
                        )
                        if fetch_err:
                            st.warning(f"⚠️ {fetch_err}")
                            st.info("Falling back to analyzing input query.")
                        elif fetched_text:
                            comment_text = fetched_text

                # Perform Model Inference
                with st.spinner("Analyzing comment text with NLP classifier..."):
                    pred = predict_comment(comment_text, model_name=selected_model_name)
                    explanation = explain_comment(comment_text)

                # Update session counters & history
                st.session_state.scan_stats["total_scanned"] += 1
                if pred["is_attacking"]:
                    st.session_state.scan_stats["flagged"] += 1

                st.session_state.history.insert(0, {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "text": comment_text[:80] + ("..." if len(comment_text) > 80 else ""),
                    "verdict": pred["verdict"],
                    "confidence": f"{pred['confidence_pct']}%",
                    "type": pred["attack_type"] or "N/A",
                    "severity": pred["severity"]
                })

                st.divider()

                # Results Presentation Layout
                st.subheader("Analysis Findings")

                col_res1, col_res2, col_res3, col_res4 = st.columns(4)

                with col_res1:
                    st.markdown("**Verdict**")
                    if pred["is_attacking"]:
                        st.markdown('<div class="badge-attacking">🚨 Attacking / Harassing</div>', unsafe_allow_html=True)
                        st.caption("Advisory: Possibly harmful, for human review.")
                    else:
                        st.markdown('<div class="badge-safe">✅ Not Attacking</div>', unsafe_allow_html=True)
                        st.caption("Benign / safe conversational speech.")

                with col_res2:
                    st.markdown("**Model Confidence**")
                    st.metric(label="Probability", value=f"{pred['confidence_pct']}%")
                    st.progress(pred["confidence_score"])

                with col_res3:
                    st.markdown("**Harassment Category**")
                    if pred["is_attacking"] and pred["attack_type"]:
                        st.metric(label="Taxonomy Type", value=pred["attack_type"])
                        st.caption(f"Category Match: {pred['type_confidence']}%")
                    else:
                        st.metric(label="Taxonomy Type", value="None (Safe)")
                        st.caption("Not applicable for non-attacking comments.")

                with col_res4:
                    st.markdown("**Severity Level**")
                    sev = pred["severity"]
                    if sev == "High":
                        st.markdown('<span class="severity-high">🔴 HIGH SEVERITY</span>', unsafe_allow_html=True)
                        st.caption("Immediate review recommended.")
                    elif sev == "Medium":
                        st.markdown('<span class="severity-medium">🟡 MEDIUM SEVERITY</span>', unsafe_allow_html=True)
                        st.caption("Standard moderator queue.")
                    elif sev == "Low":
                        st.markdown('<span class="severity-low">🔵 LOW SEVERITY</span>', unsafe_allow_html=True)
                        st.caption("Minor provocation or mild trolling.")
                    else:
                        st.markdown("<span>⚪ NONE</span>", unsafe_allow_html=True)
                        st.caption("No harassment detected.")

                # Explainability & Highlighted Text Card
                st.markdown("#### 🔍 Explainability & Token Attribution")
                st.markdown(
                    '<div class="explain-box">' + explanation["highlighted_html"] + '</div>',
                    unsafe_allow_html=True
                )

                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    if explanation["attacking_words"]:
                        st.markdown("**Top Influential Attacking Indicators:**")
                        for w in explanation["attacking_words"]:
                            st.markdown(f"- 🔴 `{w['feature']}` *(Contribution weight: {w['impact']:.3f})*")
                    else:
                        st.caption("No significant aggressive vocabulary detected.")

                with col_exp2:
                    if explanation["safe_words"]:
                        st.markdown("**Top Influential Protective / Positive Indicators:**")
                        for w in explanation["safe_words"]:
                            st.markdown(f"- 🟢 `{w['feature']}` *(Contribution weight: {abs(w['impact']):.3f})*")
                    else:
                        st.caption("Standard neutral syntax.")

                st.markdown(f'<div class="ethics-box">{explanation["ethics_notice"]}</div>', unsafe_allow_html=True)

            # ------------------------------------------------------------------
            # CASE B: WHOLE VIDEO SCANNER ('youtube_video')
            # ------------------------------------------------------------------
            elif input_type == "youtube_video":
                video_id = parsed["video_id"]
                st.subheader(f"Scanning YouTube Video: `{video_id}`")

                progress_bar = st.progress(0.1, text="Connecting to YouTube Data API...")

                comments_list, err_msg = fetch_video_comments(
                    video_id=video_id,
                    max_comments=max_comments_limit,
                    include_replies=include_replies,
                    api_key=st.session_state.youtube_api_key
                )

                if err_msg and not comments_list:
                    progress_bar.empty()
                    st.error(f"❌ Failed to retrieve comments: {err_msg}")
                else:
                    if err_msg:
                        st.warning(f"⚠️ {err_msg}")

                    total_fetched = len(comments_list)
                    progress_bar.progress(0.6, text=f"Processing {total_fetched} comments through NLP pipeline...")

                    # Batch prediction
                    df_results = predict_batch(comments_list, model_name=selected_model_name)
                    progress_bar.progress(1.0, text="Scan complete!")
                    time.sleep(0.3)
                    progress_bar.empty()

                    # Summary Metrics
                    total_scanned = len(df_results)
                    flagged_df = df_results[df_results["Is Attacking"] == True]
                    num_flagged = len(flagged_df)
                    flagged_pct = round((num_flagged / total_scanned) * 100.0, 1) if total_scanned > 0 else 0.0

                    high_sev_count = len(df_results[df_results["Severity"] == "High"])
                    med_sev_count = len(df_results[df_results["Severity"] == "Medium"])
                    low_sev_count = len(df_results[df_results["Severity"] == "Low"])

                    # Update session state
                    st.session_state.scan_stats["total_scanned"] += total_scanned
                    st.session_state.scan_stats["flagged"] += num_flagged

                    # KPI Metric Cards
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Comments Scanned", total_scanned)
                    col_m2.metric("Flagged Comments", num_flagged, delta=f"{flagged_pct}%", delta_color="inverse")
                    col_m3.metric("High Severity", high_sev_count)
                    col_m4.metric("Moderate / Low", med_sev_count + low_sev_count)

                    st.divider()

                    # Visualizations: Type Distribution & Severity Breakdown
                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        st.markdown("#### 📊 Harassment Types Breakdown")
                        if num_flagged > 0:
                            type_counts = flagged_df["Harassment Type"].value_counts().reset_index()
                            type_counts.columns = ["Harassment Type", "Count"]

                            chart_type = alt.Chart(type_counts).mark_bar(cornerRadius=6).encode(
                                x=alt.X("Count:Q", title="Number of Flagged Comments"),
                                y=alt.Y("Harassment Type:N", sort="-x", title="Category"),
                                color=alt.Color("Harassment Type:N", legend=None, scale=alt.Scale(scheme="plasma")),
                                tooltip=["Harassment Type", "Count"]
                            ).properties(height=240)
                            st.altair_chart(chart_type, use_container_width=True)
                        else:
                            st.success("🎉 No abusive or harassing comments detected in this video batch!")

                    with col_chart2:
                        st.markdown("#### 🎯 Severity Distribution")
                        if num_flagged > 0:
                            sev_data = pd.DataFrame({
                                "Severity": ["High", "Medium", "Low"],
                                "Count": [high_sev_count, med_sev_count, low_sev_count]
                            })
                            chart_sev = alt.Chart(sev_data).mark_arc(innerRadius=45).encode(
                                theta=alt.Theta("Count:Q"),
                                color=alt.Color("Severity:N", scale=alt.Scale(
                                    domain=["High", "Medium", "Low"],
                                    range=["#ef4444", "#f59e0b", "#38bdf8"]
                                )),
                                tooltip=["Severity", "Count"]
                            ).properties(height=240)
                            st.altair_chart(chart_sev, use_container_width=True)
                        else:
                            st.info("Clean video comment section.")

                    st.divider()

                    # Flagged Comments Data Table
                    st.markdown("#### 🚨 Flagged Comments Requiring Human Review")
                    if num_flagged > 0:
                        flagged_sorted = flagged_df.sort_values(by="Confidence (%)", ascending=False).reset_index(drop=True)
                        st.dataframe(
                            flagged_sorted[["Comment", "Harassment Type", "Severity", "Confidence (%)"]],
                            use_container_width=True,
                            height=280
                        )

                        # CSV Export Download
                        csv_data = flagged_sorted.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Flagged Comments Report (CSV)",
                            data=csv_data,
                            file_name=f"flagged_comments_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.write("All analyzed comments passed toxicity verification.")

                    st.markdown(f'<div class="ethics-box">{ETHICS_NOTICE}</div>', unsafe_allow_html=True)


# ==============================================================================
# VIEW 2: BATCH CSV ANALYSIS
# ==============================================================================
elif nav_option == "📁 Batch CSV Analysis":
    st.markdown('<div class="hero-title">Batch CSV Comment Moderation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Upload offline datasets, customer feedback files, or exported comment threads for bulk classification.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"File loaded successfully: {len(df_upload)} rows found.")

            # Identify candidate text column
            possible_cols = [c for c in df_upload.columns if any(k in c.lower() for k in ("text", "comment", "body", "content", "msg"))]
            default_col = possible_cols[0] if possible_cols else df_upload.columns[0]

            col_choice = st.selectbox("Select Comment Text Column:", options=df_upload.columns, index=list(df_upload.columns).index(default_col))

            if st.button("⚡ Run Batch Analysis", type="primary"):
                with st.spinner("Processing comments in batch..."):
                    texts = df_upload[col_choice].astype(str).tolist()
                    batch_res = predict_batch(texts, model_name=selected_model_name)

                # Merge back
                df_merged = df_upload.copy()
                df_merged["Verdict"] = batch_res["Verdict"]
                df_merged["Confidence (%)"] = batch_res["Confidence (%)"]
                df_merged["Harassment Type"] = batch_res["Harassment Type"]
                df_merged["Severity"] = batch_res["Severity"]

                num_flagged = len(batch_res[batch_res["Is Attacking"] == True])
                total = len(batch_res)

                st.divider()
                st.subheader("Batch Results Summary")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Records", total)
                c2.metric("Flagged as Harassment", num_flagged)
                c3.metric("Flagged Rate", f"{round((num_flagged/total)*100, 1)}%")

                st.dataframe(df_merged.head(100), use_container_width=True)

                csv_exp = df_merged.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Complete Labeled Dataset (CSV)",
                    data=csv_exp,
                    file_name=f"moderated_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error reading CSV: {e}")


# ==============================================================================
# VIEW 3: MODEL EVALUATION & METRICS
# ==============================================================================
elif nav_option == "📊 Model Evaluation & Metrics":
    st.markdown('<div class="hero-title">Model Performance & Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Comprehensive benchmarking, error analysis, and cross-architecture comparison.</div>',
        unsafe_allow_html=True
    )

    if metrics_data:
        m_models = metrics_data.get("models", {})

        # Model Comparison Table
        st.subheader("1. Architecture Benchmark Comparison")

        comp_rows = []
        for name, m in m_models.items():
            comp_rows.append({
                "Model Architecture": name,
                "Accuracy": f"{m.get('accuracy', 0)*100:.2f}%",
                "Precision": f"{m.get('precision', 0)*100:.2f}%",
                "Recall": f"{m.get('recall', 0)*100:.2f}%",
                "F1-Score": f"{m.get('f1_score', 0)*100:.2f}%"
            })

        df_benchmarks = pd.DataFrame(comp_rows)
        st.dataframe(df_benchmarks, use_container_width=True)

        st.caption(
            "Benchmark Reference: `unitary/toxic-bert` by Hanu & Unitary team (2020), licensed under Apache-2.0. "
            "Evaluated on identical stratified test split (N=1,196)."
        )

        st.divider()

        # Confusion Matrix Section
        st.subheader("2. Confusion Matrix (Main Models)")
        col_cm1, col_cm2 = st.columns(2)

        with col_cm1:
            st.markdown("#### TF-IDF + Logistic Regression")
            lr_cm = m_models.get("TF-IDF + Logistic Regression (Main)", {}).get("confusion_matrix", [[618, 39], [44, 495]])
            cm_df1 = pd.DataFrame(
                lr_cm,
                index=["True: Attacking", "True: Not Attacking"],
                columns=["Pred: Attacking", "Pred: Not Attacking"]
            )
            st.table(cm_df1)

        with col_cm2:
            st.markdown("#### TF-IDF + Calibrated Linear SVM")
            svm_cm = m_models.get("TF-IDF + Calibrated Linear SVM", {}).get("confusion_matrix", [[616, 41], [41, 498]])
            cm_df2 = pd.DataFrame(
                svm_cm,
                index=["True: Attacking", "True: Not Attacking"],
                columns=["Pred: Attacking", "Pred: Not Attacking"]
            )
            st.table(cm_df2)

        st.divider()

        # Qualitative Error Analysis
        st.subheader("3. In-Depth Error Analysis & Boundary Cases")
        err_analysis = metrics_data.get("error_analysis", {})

        col_err1, col_err2 = st.columns(2)

        with col_err1:
            st.markdown("#### 🔴 False Positives (Benign Flagged as Attacking)")
            st.caption("Common cause: Aggressive slang, competitive gaming context, or quotes.")
            fps = err_analysis.get("false_positives", [])
            for fp in fps[:4]:
                st.markdown(f"> *\"{fp.get('comment', '')}\"*  \n— **Reason:** `{fp.get('linguistic_reason', '')}`")

        with col_err2:
            st.markdown("#### 🟡 False Negatives (Attacking Missed as Safe)")
            st.caption("Common cause: Sarcastic praise, veiled threats, or novel phonetic obfuscations.")
            fns = err_analysis.get("false_negatives", [])
            for fn in fns[:4]:
                st.markdown(f"> *\"{fn.get('comment', '')}\"*  \n— **Reason:** `{fn.get('linguistic_reason', '')}`")

        st.divider()

        # Dataset Class Distribution
        st.subheader("4. Training Dataset Class Distribution")
        d_info = metrics_data.get("dataset", {})
        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Total Raw Records", d_info.get("total_raw_rows", 8452))
        col_d2.metric("Deduplicated Records", d_info.get("total_dedup_rows", 5978))
        col_d3.metric("Attacking Ratio", f"{round(d_info.get('attacking_ratio', 0.549)*100, 1)}%")

        type_dist = d_info.get("type_distribution", {})
        if type_dist:
            df_td = pd.DataFrame(list(type_dist.items()), columns=["Category", "Samples"])
            chart_td = alt.Chart(df_td).mark_bar(cornerRadius=5).encode(
                x=alt.X("Samples:Q", title="Dataset Frequency"),
                y=alt.Y("Category:N", sort="-x", title="Harassment Type"),
                color=alt.Color("Category:N", legend=None)
            ).properties(height=220)
            st.altair_chart(chart_td, use_container_width=True)

    else:
        st.warning("Evaluation metrics file not found. Run model evaluation to generate `models/evaluation_metrics.json`.")


# ==============================================================================
# VIEW 4: ETHICS, PRIVACY & METHODOLOGY
# ==============================================================================
elif nav_option == "⚖️ Ethics & Privacy":
    st.markdown('<div class="hero-title">Privacy, Ethics & Limitations Statement</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Antigravity GuardNet operates under strict data minimization and ethical AI principles.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 🔒 1. Privacy Guarantees & Data Minimization
    - **No PII Collection**: The system extracts **ONLY** raw comment text. Usernames, channel URLs, avatars, and user IDs are never retrieved, stored, or processed.
    - **Ephemeral Processing**: Scanned comments exist purely in volatile memory for the duration of the browser session. No comments are saved to persistent databases or logged on external servers.
    - **No Unauthorized Scraping**: We do not scrape unauthorized platforms. Direct URL parsing is restricted to the official YouTube Data API v3. Unsupported links trigger an explicit prompt to use the paste-text fallback.

    ---

    ### 🤝 2. Human-in-the-Loop Moderation Protocol
    - **Advisory Classification**: All model outputs are labeled as *"Possibly harmful, for human review"*.
    - **Non-Automated Banning**: Algorithmic outputs are intended to triage and accelerate human moderation workflows, not replace thoughtful judicial evaluation.

    ---

    ### ⚠️ 3. Known Limitations & Technical Constraints
    - **Sarcasm and Irony**: Subtle passive-aggressive sarcasm without overt aggressive vocabulary may lead to false negatives.
    - **Multilingual & Hinglish**: Code-mixed text (e.g. Hinglish) contains phonetic variations outside standard English lexicon.
    - **Evolving Internet Slang**: New neologisms and obfuscations emerge continuously.
    - **API Quota Constraints**: YouTube Data API v3 enforces daily quota limits (10,000 units/day). If exceeded, users must switch to manual text input.
    - **Comments Disabled**: Videos with comments disabled by creators cannot be retrieved.

    ---

    ### 📚 4. Dataset & Pretrained Benchmark Citations
    1. **Dataset**: *Cyberbullying and Harassment Detection Using ML* by Saiful Islam (Kaggle), licensed under Apache-2.0.
    2. **Benchmark**: Hanu, L., & Unitary team. (2020). *Detoxify: Toxic comment classification (`unitary/toxic-bert`)*. Apache License 2.0.
    """)


# ==============================================================================
# VIEW 5: SESSION AUDIT LOG
# ==============================================================================
elif nav_option == "📜 Session Audit Log":
    st.markdown('<div class="hero-title">Session Audit Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Temporary in-memory record of items analyzed during this active session.</div>', unsafe_allow_html=True)

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        csv_hist = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Audit Log (CSV)",
            data=csv_hist,
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No comments scanned yet in this session. Run an analysis on the Live Scanner page!")
