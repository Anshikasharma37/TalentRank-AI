
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
import time

# Make src/ importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(
    page_title="TalentRank AI — Recruiter Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1a1d2e;
        border-right: 1px solid #2d3061;
    }

    /* Cards */
    .candidate-card {
        background: linear-gradient(135deg, #1e2140 0%, #252847 100%);
        border: 1px solid #3d4270;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        transition: border-color 0.2s;
    }
    .candidate-card:hover { border-color: #6c72ff; }

    /* Score badge */
    .score-badge {
        background: linear-gradient(90deg, #6c72ff, #a855f7);
        color: white;
        border-radius: 20px;
        padding: 4px 14px;
        font-weight: bold;
        font-size: 1.1em;
        display: inline-block;
    }

    /* Rank number */
    .rank-number {
        font-size: 2.5em;
        font-weight: 900;
        color: #6c72ff;
        line-height: 1;
    }

    /* Flag badges */
    .flag-warning {
        background: #7c2d12;
        color: #fca5a5;
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 0.75em;
        margin: 2px;
        display: inline-block;
    }
    .flag-ok {
        background: #14532d;
        color: #86efac;
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 0.75em;
        margin: 2px;
        display: inline-block;
    }

    /* Reasoning text */
    .reasoning-text {
        color: #94a3b8;
        font-style: italic;
        font-size: 0.95em;
        border-left: 3px solid #6c72ff;
        padding-left: 12px;
        margin: 8px 0;
    }

    /* Metric */
    .metric-label {
        color: #64748b;
        font-size: 0.75em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #e2e8f0;
        font-size: 1.1em;
        font-weight: 600;
    }

    h1, h2, h3 { color: #e2e8f0 !important; }
    p, li { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 30px 0 20px 0;">
    <h1 style="font-size:2.8em; margin:0; background: linear-gradient(90deg, #6c72ff, #a855f7);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🎯 TalentRank AI
    </h1>
    <p style="color:#64748b; font-size:1.1em; margin:8px 0 0 0;">
        Intelligent Candidate Discovery & Ranking · Powered by Semantic Search + Gemini AI
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")

    run_mode = st.radio(
        "Mode",
        ["📊 View Results", "🚀 Run Pipeline"],
        help="View existing results or run a new ranking"
    )

    st.divider()
    st.markdown("### 🔍 Filter Results")

    min_score = st.slider("Minimum Score", 0.0, 1.0, 0.0, 0.05)
    max_rank  = st.slider("Max Rank", 1, 100, 100)

    st.divider()
    st.markdown("### 📋 Columns Displayed")
    show_radar   = st.checkbox("Radar Charts",   value=True)
    show_signals = st.checkbox("Behavioral Signals", value=True)
    show_flags   = st.checkbox("Disqualifier Flags", value=True)


# ─── Run Pipeline Mode ────────────────────────────────────────────────────────
if run_mode == "🚀 Run Pipeline":
    st.markdown("## 🚀 Run Full Pipeline")

    col1, col2 = st.columns(2)
    with col1:
        skip_llm = st.checkbox("Skip Gemini Re-ranking (faster)", value=False)
    with col2:
        no_cache = st.checkbox("Ignore Cache", value=False)

    if st.button("▶ Run Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running pipeline... (this takes ~20-30 min on CPU)"):
            import subprocess
            cmd = [sys.executable, "rank.py"]
            if skip_llm:
                cmd.append("--skip-llm")
            if no_cache:
                cmd.append("--no-cache")

            progress_bar = st.progress(0)
            status_text  = st.empty()

            # Run as subprocess so we can show live progress
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(os.path.dirname(__file__))
            )

            log_area = st.empty()
            log_lines = []
            stage = 0

            for line in process.stdout:
                log_lines.append(line.rstrip())
                log_area.code("\n".join(log_lines[-30:]))   # show last 30 lines

                # Update progress bar based on stage markers
                if "[1/6]" in line: progress_bar.progress(0.10); status_text.text("Parsing JD...")
                elif "[2/6]" in line: progress_bar.progress(0.15); status_text.text("Fast prefilter (100K candidates)...")
                elif "[3/6]" in line: progress_bar.progress(0.40); status_text.text("Embedding candidates...")
                elif "[4/6]" in line: progress_bar.progress(0.65); status_text.text("Semantic search...")
                elif "[5/6]" in line: progress_bar.progress(0.75); status_text.text("5D scoring...")
                elif "[6/6]" in line: progress_bar.progress(0.85); status_text.text("Gemini re-ranking...")
                elif "[7/7]" in line: progress_bar.progress(0.95); status_text.text("Writing submission.csv...")

            process.wait()
            progress_bar.progress(1.0)

            if process.returncode == 0:
                status_text.success("✅ Pipeline complete! Switch to 'View Results' mode.")
            else:
                status_text.error("❌ Pipeline failed. Check logs above.")

    st.stop()


# ─── View Results Mode ────────────────────────────────────────────────────────
st.markdown("## 📊 Ranking Results")

# Load submission CSV
submission_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "submission.csv"
)

if not os.path.exists(submission_path):
    st.warning(
        "No submission.csv found. Switch to '🚀 Run Pipeline' mode to generate results, "
        "or run `python rank.py` from the terminal."
    )
    st.stop()

df = pd.read_csv(submission_path)

# Apply filters
df_filtered = df[
    (df["score"] >= min_score) &
    (df["rank"]  <= max_rank)
].copy()

# ─── Summary Stats ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Candidates Ranked", len(df))
with col2:
    st.metric("Showing", len(df_filtered))
with col3:
    st.metric("Top Score", f"{df['score'].max():.4f}")
with col4:
    st.metric("Median Score", f"{df['score'].median():.4f}")

st.divider()

# ─── Score Distribution Chart ─────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    fig_bar = px.bar(
        df,
        x="rank",
        y="score",
        color="score",
        color_continuous_scale=["#3d4270", "#6c72ff", "#a855f7"],
        labels={"rank": "Rank", "score": "Score"},
        title="Score Distribution Across Top 100 Candidates",
    )
    fig_bar.update_layout(
        paper_bgcolor="#1e2140",
        plot_bgcolor="#1e2140",
        font_color="#94a3b8",
        title_font_color="#e2e8f0",
        showlegend=False,
        height=300,
    )
    fig_bar.update_traces(showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("### 🎯 About This Ranking")
    st.markdown("""
    **5-Dimension Scoring:**
    - 🔵 **Semantic Fit** (30%) — Career text vs JD embedding
    - 🟣 **Career Fit** (25%) — Role family + company type
    - 🟡 **Trajectory** (20%) — Moving toward AI/ML?
    - 🟢 **Behavioral** (15%) — Platform activity signals
    - 🔴 **Education** (10%) — Degree + certs + assessments

    **Anti-Trap Built-in:**
    Marketing Managers with AI keywords score near zero regardless of skill list.
    """)

st.divider()

# ─── Candidate Cards ──────────────────────────────────────────────────────────
st.markdown(f"### 🏆 Top Candidates ({len(df_filtered)} shown)")

# Load detailed scoring cache if available
detailed_cache_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    ".cache", "detailed_scores.json"
)
detailed_scores = {}
if os.path.exists(detailed_cache_path):
    with open(detailed_cache_path) as f:
        detailed_scores = {d["candidate_id"]: d for d in json.load(f)}

for _, row in df_filtered.iterrows():
    cid       = row["candidate_id"]
    rank      = int(row["rank"])
    score     = float(row["score"])
    reasoning = str(row["reasoning"])

    # Get detailed scores if available
    detail = detailed_scores.get(cid, {})

    with st.container():
        col_rank, col_info, col_chart = st.columns([1, 3, 2])

        with col_rank:
            st.markdown(
                f'<div class="rank-number">#{rank}</div>'
                f'<div class="score-badge">{score:.4f}</div>',
                unsafe_allow_html=True
            )

        with col_info:
            title   = detail.get("current_title", "").title() or "—"
            years   = detail.get("years_exp", "")
            years_str = f"{years:.1f} yrs" if years else ""

            st.markdown(f"**{cid}**")
            st.markdown(f"*{title}* {'· ' + years_str if years_str else ''}")
            st.markdown(
                f'<div class="reasoning-text">"{reasoning}"</div>',
                unsafe_allow_html=True
            )

            # Disqualifier flags
            if show_flags and detail.get("disqualifier_flags"):
                for flag in detail["disqualifier_flags"]:
                    st.markdown(
                        f'<span class="flag-warning">⚠ {flag.replace("_", " ")}</span>',
                        unsafe_allow_html=True
                    )
            elif show_flags and detail:
                st.markdown(
                    '<span class="flag-ok">✓ No disqualifiers</span>',
                    unsafe_allow_html=True
                )

            # Behavioral signals
            if show_signals and detail:
                sig_cols = st.columns(4)
                signals = [
                    ("Open to Work", "✅" if detail.get("open_to_work") else "❌"),
                    ("Days Inactive", str(detail.get("days_since_active", "?"))),
                    ("Career Fit", f"{detail.get('career_fit', 0):.0%}"),
                    ("Trajectory", f"{detail.get('trajectory', 0):.0%}"),
                ]
                for i, (label, value) in enumerate(signals):
                    with sig_cols[i]:
                        st.markdown(
                            f'<div class="metric-label">{label}</div>'
                            f'<div class="metric-value">{value}</div>',
                            unsafe_allow_html=True
                        )

        # Radar chart
        if show_radar and detail:
            with col_chart:
                dimensions = ["Semantic", "Career", "Trajectory", "Behavioral", "Education"]
                values = [
                    detail.get("semantic_fit", 0),
                    detail.get("career_fit", 0),
                    detail.get("trajectory", 0),
                    detail.get("behavioral", 0),
                    detail.get("education_cert", 0),
                ]

                fig = go.Figure(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=dimensions + [dimensions[0]],
                    fill="toself",
                    fillcolor="rgba(108, 114, 255, 0.2)",
                    line=dict(color="#6c72ff", width=2),
                    marker=dict(color="#a855f7", size=6),
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="#1e2140",
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            tickfont=dict(size=8, color="#64748b"),
                            gridcolor="#2d3061",
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=9, color="#94a3b8"),
                            gridcolor="#2d3061",
                        ),
                    ),
                    paper_bgcolor="#1e2140",
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=200,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"radar_{cid}")

        st.markdown('<hr style="border-color: #2d3061; margin: 8px 0;">', unsafe_allow_html=True)

# ─── Download Button ──────────────────────────────────────────────────────────
with open(submission_path, "rb") as f:
    st.download_button(
        label="⬇ Download submission.csv",
        data=f,
        file_name="submission.csv",
        mime="text/csv",
        type="primary",
        use_container_width=False,
    )
