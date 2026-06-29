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
[![Submission](https://img.shields.io/badge/📄_Submission-submission.xlsx-blue?style=for-the-badge)](submission.xlsx)

</div>

---

## 👥 Target Audience
This system is designed for **technical recruiters, talent acquisition teams, and engineering managers** who need to efficiently identify top-tier AI/ML engineering candidates from large talent pools.

---

## 🎯 Purpose & Goals
The core purpose of this project is to automate candidate screening at scale. Rather than relying on simple keyword matching, the pipeline is engineered to assess a candidate's actual history of building systems, their career growth, and their likelihood of successful placement.

---

## ⚠️ Challenges & Issues We Solve

Standard resume parsers and Applicant Tracking Systems (ATS) suffer from several critical limitations:
1. **Keyword Stuffing**: Candidates often list popular ML libraries and buzzwords (like "LLM", "PyTorch", "RAG") without having actual engineering experience. Traditional parsers rank these profiles highly, leading to poor matches.
2. **Out-of-Date/Ghost Candidates**: Profiles that have been inactive for months or candidates who rarely respond to recruiter outreach waste recruitment resources.
3. **Misaligned Career Growth**: Hiring managers need to know if a candidate is actively progressing *towards* ML engineering or moving *away* from it (e.g., shifting into non-technical management).
4. **Context-Free Filtering**: Simple rule-based systems might filter out an exceptional candidate who built a complex recommendation system simply because they did not use exact keywords from the job description.

---

## 💡 Our Solution
TalentRank AI introduces a **multi-stage screening and ranking pipeline** that combines rule-based filters, semantic search, a multi-dimensional scoring engine, and Large Language Model (LLM) reasoning:

### 1. Hard & Soft Disqualifiers
Eliminates profiles that do not match technical baselines, identify candidates from non-technical backgrounds, flag inactive profiles, and discount pure-consulting backgrounds.

### 2. Semantic Search (Dense Retrieval)
We convert candidate career descriptions into vector embeddings using a sentence-transformer model and run a similarity search against the job description using **FAISS**. This surfaces candidates who built relevant systems even if they used different terminology.

### 3. 5-Dimension Scoring Engine
Evaluates candidates across five distinct pillars:
* **Semantic Fit (30%)** — Cosine similarity of career history against the job description.
* **Career Fit (25%)** — Professional title relevance, experience range, and product company ratio.
* **Career Trajectory (20%)** — Measures whether the candidate's recent roles are progressively more AI/ML focused.
* **Behavioral Engagement (15%)** — Availability flags, notice periods, and response rates.
* **Education & Certifications (10%)** — Institutional tiers, relevant majors, assessments, and certifications.

### 4. Recruiter-Quality Explanations
Utilizes **Gemini 1.5 Flash** to analyze the top candidates and write a concise, professional summary explaining *why* each candidate was selected.

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

---

## 🏗️ How It Works

```
100,000 Candidates (candidates.jsonl)
          │
          ▼  Rule-based Fast Prefilter
        3,000  ← filtered by title family and activity signals
          │
          ▼  Sentence-Transformer Embedding + FAISS
          500  ← top candidates matching JD semantically
          │
          ▼  Full 5-Dimension Weighted Scoring
          200  ← top candidates based on technical & behavioral composite
          │
          ▼  Gemini 1.5 Flash LLM Re-Ranking
          100  → final submission (submission.csv & submission.xlsx)
```

---

## 🛡️ Anti-Trap Disqualifiers
Explicit requirements and negative signals defined in the job description are encoded as multiplier penalties:

| Disqualifier | Penalty Multiplier |
|---|:---:|
| Non-technical role (Marketing, HR, Accountant, etc.) | ×0.10 |
| Entire career at consulting firms only | ×0.15 |
| Inactive candidate (6+ months inactive, low response) | ×0.40 |
| CV/Speech-only background without NLP/IR exposure | ×0.60 |

---

## 🤖 Gemini LLM Reasoning
Each shortlisted candidate is annotated with a concise evaluation summarizing their fit:

> *"Experienced Machine Learning Engineer with 6.1 years of experience at Razorpay; strong background in building NLP pipelines and vector search systems; highly active and immediately available."*

---

## 📁 Project Structure

```
TalentRank-AI/
│
├── 📄 rank.py                    ← Pipeline entry point
├── ⚙️  config.py                  ← Central settings, weights, and keywords
├── 📋 requirements.txt           ← Local dependencies
├── 📋 requirements-app.txt       ← Streamlit deployment dependencies
├── 📊 submission.csv             ← Final ranked candidate CSV
├── 📊 submission.xlsx            ← Final ranked candidate Excel Sheet
│
├── src/
│   ├── jd_parser.py              ← Parses the Word JD document
│   ├── profile_parser.py         ← Streams candidates from raw dataset
│   ├── signals_parser.py         ← Normalizes candidate behavioral signals
│   ├── disqualifier.py           ← Applies hard filters and penalties
│   ├── scorer.py                 ← Computes the 5D weighted score
│   ├── embedder.py               ← Text embedding helper
│   ├── vector_store.py           ← Manages FAISS vector index
│   └── llm_ranker.py             ← Gemini re-ranking & reasoning
│
└── app/
    └── streamlit_app.py          ← Visual recruiter dashboard
```

---

## 🏆 Top Candidates Preview

| Rank | Title | Years of Exp | Score |
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

## 🚀 Deployment & Local Execution

### Run Locally
```bash
streamlit run app/streamlit_app.py
```

### Run on Google Colab
Upload the dataset files to Google Drive, then execute:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install packages
!pip install -q google-generativeai sentence-transformers faiss-cpu python-docx numpy pandas openpyxl

# Set configurations and execute
import config
config.CANDIDATES_FILE = '/content/drive/MyDrive/TalentRank-AI_dataset/candidates.jsonl'
config.JD_FILE         = '/content/drive/MyDrive/TalentRank-AI_dataset/job_description.docx'

from rank import run_pipeline
run_pipeline()
```

---

## ✅ Validation

Verify the structure and formatting of the output sheet using the validator tool:
```bash
python validate_submission.py submission.csv
# Output: Submission is valid.
```

---

<div align="center">

Built for the **Redrob AI Hackathon** — India Runs Data & AI Challenge

</div>
