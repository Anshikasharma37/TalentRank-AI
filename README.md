<div align="center">

# 🎯 TalentRank AI

### Intelligent Candidate Discovery & Ranking

*Redrob AI Hackathon — India Runs Data & AI Challenge*

<br/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini_1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

[![Run Pipeline](https://img.shields.io/badge/▶_Run_Pipeline-python_rank.py-success?style=for-the-badge)](rank.py)
[![Live Dashboard](https://img.shields.io/badge/📊_Live_Dashboard-Render-4385F4?style=for-the-badge&logo=render&logoColor=white)](https://talentrank-ai-24gb.onrender.com/)
[![Submission](https://img.shields.io/badge/📄_Submission-submission.csv-blue?style=for-the-badge)](submission.csv)

</div>

---

## 🚨 The Trap We Were Warned About

> *"The right answer is NOT to find candidates with the most AI keywords. That's a trap."*
> — Redrob Hackathon Organizers

The sample submission ranked an **Accountant at #2** and a **Graphic Designer at #4** — simply because they had AI keywords in their skills. Our system is built specifically to defeat this.

| | ❌ Sample Submission (Bad) | ✅ TalentRank AI (Ours) |
|---|---|---|
| Rank 1 | Civil Engineer | **Lead AI Engineer** |
| Rank 2 | Accountant | **Staff ML Engineer** |
| Rank 4 | Graphic Designer | **Senior ML Engineer** |
| Non-technical in Top 10 | **10 / 10** | **0 / 10** |

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key in config.py
GEMINI_API_KEY = "AIzaSy..."

# 3. Run the full pipeline
python rank.py

# 4. Launch the recruiter dashboard
streamlit run app/streamlit_app.py
```

> ⏱️ First run ~25 min &nbsp;|&nbsp; Re-runs ~5 min (embeddings cached)

---

## 🏗️ How It Works

```
100,000 Candidates (candidates.jsonl)
          │
          ▼  Rule-based Fast Prefilter          ⏱ 28 seconds
        3,000  ← scored by role title + behavioral signals
          │     ← Marketing Managers, Accountants eliminated here
          │
          ▼  Sentence-Transformer Embedding + FAISS    ⏱ 3 minutes
          500  ← top by semantic similarity to JD
          │     ← finds engineers who built retrieval systems
          │       even without using exact JD keywords
          │
          ▼  Full 5-Dimension Weighted Scoring    ⏱ instant
          200  ← best by career fit, trajectory, behavioral signals
          │
          ▼  Gemini 1.5 Flash LLM Re-Ranking     ⏱ 10 minutes
          100  → submission.csv  ✅
```

---

## 📐 5-Dimension Scoring Engine

| Dimension | Weight | What It Actually Measures |
|---|:---:|---|
| 🔵 **Semantic Fit** | 30% | Cosine similarity of career descriptions vs JD embedding |
| 🟣 **Career Fit** | 25% | ML/AI title family + product company ratio + experience range |
| 🟡 **Trajectory** ⭐ | 20% | Are recent roles MORE AI-relevant than older ones? |
| 🟢 **Behavioral** | 15% | Platform activity, response rate, GitHub score, availability |
| 🔴 **Education & Certs** | 10% | Degree tier, relevant field, certifications, assessment scores |

> ⭐ **Trajectory** is our unique dimension — no other team asks *"is this person moving toward AI?"* instead of just *"are they there now?"*

---

## 🛡️ Anti-Trap Disqualifiers

The JD listed exactly who NOT to hire. We encoded every one:

| Disqualifier | Penalty |
|---|:---:|
| Non-technical role (Marketing, HR, Accountant...) | ×0.10 |
| Entire career at consulting firms only (TCS, Wipro...) | ×0.15 |
| Ghost candidate (inactive 6+ months, <10% response rate) | ×0.40 |
| CV/Speech only, no NLP/Information Retrieval exposure | ×0.60 |

---

## 🤖 Gemini LLM Reasoning

Every row in `submission.csv` has a sentence like a real recruiter wrote it:

> *"Built hybrid retrieval pipeline at Swiggy serving 50M+ users; trajectory clearly moving toward ML systems over last 3 roles; active on platform and open to Pune relocation."*

This is what makes our submission stand out — judges read reasoning, not just scores.

---

## 📁 Project Structure

```
TalentRank-AI/
│
├── 📄 rank.py                    ← Single command: runs everything
├── ⚙️  config.py                  ← All paths, weights, keyword lists
├── 📋 requirements.txt
├── 📊 submission.csv             ← Final output (100 ranked candidates)
│
├── src/
│   ├── jd_parser.py              ← Parse JD docx → weighted sections
│   ├── profile_parser.py         ← Stream 100K JSONL line-by-line
│   ├── signals_parser.py         ← Normalize 23 behavioral signals → 0–1
│   ├── disqualifier.py           ← JD-explicit hard filter + penalties
│   ├── scorer.py                 ← 5-dimension weighted scoring engine
│   ├── embedder.py               ← all-MiniLM-L6-v2 embeddings
│   ├── vector_store.py           ← FAISS IndexFlatIP + disk cache
│   └── llm_ranker.py             ← Gemini 1.5 Flash re-ranking
│
└── app/
    └── streamlit_app.py          ← Recruiter dashboard (dark mode + radar charts)
```

---

## 🏆 Top 10 Results

All top 10 are real ML/AI engineers. Zero non-technical roles — confirmed.

| Rank | Title | Years | Score |
|:---:|---|:---:|:---:|
| 🥇 1 | Lead AI Engineer | 6.7 | 0.7951 |
| 🥈 2 | Staff ML Engineer | 7.0 | 0.7905 |
| 🥉 3 | ML Engineer | 3.9 | 0.7888 |
| 4 | Senior ML Engineer | 6.1 | 0.7883 |
| 5 | Senior Applied Scientist | 16.2 | 0.7816 |
| 6 | Senior AI Engineer | 5.9 | 0.7771 |
| 7 | Senior ML Engineer | 7.2 | 0.7771 |
| 8 | Senior AI Engineer | 7.8 | 0.7736 |
| 9 | Senior NLP Engineer | 8.9 | 0.7732 |
| 10 | Senior AI Engineer | 5.9 | 0.7722 |

---

## 🚀 Deployment

### Deploy Dashboard on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → Sign in with GitHub
3. Click **"New app"**
4. Select your repo → set **Main file path** to `app/streamlit_app.py`
5. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "AIzaSy..."
   ```
6. Click **Deploy** → your app is live at `https://yourname-talentrank.streamlit.app`

### Run Locally

```bash
streamlit run app/streamlit_app.py
```

### Run on Google Colab

Upload `candidates.jsonl` and `job_description.docx` to Google Drive, then:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install packages
!pip install -q google-generativeai sentence-transformers faiss-cpu python-docx

# Run pipeline
import config
config.CANDIDATES_FILE = '/content/drive/MyDrive/TalentRank/candidates.jsonl'
config.JD_FILE         = '/content/drive/MyDrive/TalentRank/job_description.docx'

from rank import run_pipeline
run_pipeline()
```

---

## ✅ Validation

```bash
python validate_submission.py submission.csv
# Submission is valid.
```

---

<div align="center">

Built for the **Redrob AI Hackathon** — India Runs Data & AI Challenge

*Defeating keyword-stuffing, one trajectory score at a time.*

</div>
