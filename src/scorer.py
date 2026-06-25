"""
Dimensions:
  semantic_fit    (30%) — cosine similarity: candidate text vs JD embedding
  career_fit      (25%) — role family + company type + years of experience
  trajectory      (20%) — are recent roles moving TOWARD AI/ML?
  behavioral      (15%) — platform activity signals
  education_cert  (10%) — degree tier + certs + assessment scores

"""

import re
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    WEIGHTS, AI_ROLE_KEYWORDS, NON_TECHNICAL_ROLE_KEYWORDS,
    CONSULTING_COMPANIES, EDUCATION_TIER_SCORES, RELEVANT_FIELDS,
    RELEVANT_CERT_KEYWORDS
)
from src.signals_parser import normalize_signals, composite_behavioral_score
from src.disqualifier import get_disqualifier_info


# keyword sets for AI-relevance scoring

AI_CORE_SKILLS = {
    "machine learning", "deep learning", "nlp", "natural language processing",
    "llm", "large language model", "retrieval", "rag", "vector database",
    "embedding", "faiss", "pinecone", "weaviate", "elasticsearch", "opensearch",
    "ranking", "recommendation system", "information retrieval",
    "pytorch", "tensorflow", "transformers", "bert", "gpt",
    "fine-tuning", "fine tuning", "lora", "qlora",
    "a/b testing", "ndcg", "mrr", "evaluation framework",
    "python", "spark", "mlops", "model serving", "inference",
}

AI_PRODUCTION_SIGNALS = [
    # These phrases in descriptions indicate REAL production work (not tutorials)
    "deployed to production", "production system", "serving real users",
    "at scale", "million", "billion", "latency", "throughput",
    "ranking system", "recommendation engine", "search system",
    "embedding pipeline", "retrieval pipeline", "hybrid search",
    "a/b test", "online experiment", "offline evaluation",
    "model retraining", "embedding drift", "index refresh",
    "shipped", "launched", "built and deployed", "end-to-end",
]


def score_candidate(candidate: dict, jd_embedding=None, candidate_embedding=None) -> dict:
    
    #  Normalize behavioral signals
    signals_raw  = candidate.get("redrob_signals", {})
    signals_norm = normalize_signals(signals_raw)
    behavioral   = composite_behavioral_score(signals_norm)

    #  Get disqualifier penalty 
    disq = get_disqualifier_info(candidate, signals_norm)
    penalty = disq["penalty_multiplier"]

    # Compute all 5 dimensions 
    sem_fit     = _semantic_fit_score(candidate_embedding, jd_embedding)
    career      = _career_fit_score(candidate)
    trajectory  = _trajectory_score(candidate)
    education   = _education_cert_score(candidate, signals_raw)

    # Weighted total (before penalty)
    raw_score = (
        sem_fit    * WEIGHTS["semantic_fit"]    +
        career     * WEIGHTS["career_fit"]      +
        trajectory * WEIGHTS["trajectory"]      +
        behavioral * WEIGHTS["behavioral"]      +
        education  * WEIGHTS["education_cert"]
    )

    # Apply disqualifier penalty
    final_score = round(raw_score * penalty, 4)

    return {
        "candidate_id":       candidate["candidate_id"],
        "current_title":      candidate["current_title"],
        "years_exp":          candidate["years_exp"],
        # Individual dimension scores
        "semantic_fit":       round(sem_fit,    4),
        "career_fit":         round(career,     4),
        "trajectory":         round(trajectory, 4),
        "behavioral":         round(behavioral, 4),
        "education_cert":     round(education,  4),
        # Final
        "raw_score":          round(raw_score,  4),
        "penalty_multiplier": penalty,
        "final_score":        final_score,
        # Metadata for Streamlit display
        "disqualifier_flags": disq["flags"],
        "is_disqualified":    disq["is_hard_disqualified"],
        "open_to_work":       signals_norm["open_to_work"],
        "days_since_active":  signals_norm["days_since_active"],
    }


def fast_prefilter_score(candidate: dict) -> float:
    """
    Quick score using ONLY career_fit + behavioral — no embeddings.
    Used to reduce 100K to top 3000 before expensive embedding step.
    """
    signals_raw  = candidate.get("redrob_signals", {})
    signals_norm = normalize_signals(signals_raw)
    behavioral   = composite_behavioral_score(signals_norm)
    career       = _career_fit_score(candidate)
    disq         = get_disqualifier_info(candidate, signals_norm)

    fast_score = (career * 0.65) + (behavioral * 0.35)
    return fast_score * disq["penalty_multiplier"]


# Dimension Scorers 

def _semantic_fit_score(candidate_emb, jd_emb) -> float:
    
    if candidate_emb is None or jd_emb is None:
        return 0.5   # neutral — will be updated in semantic phase

    import numpy as np
    # Cosine similarity
    c_norm = candidate_emb / (np.linalg.norm(candidate_emb) + 1e-9)
    j_norm = jd_emb        / (np.linalg.norm(jd_emb)        + 1e-9)
    sim = float(np.dot(c_norm, j_norm))
    # Map from [-1,1] to [0,1]
    return max(0.0, min((sim + 1) / 2, 1.0))


def _career_fit_score(candidate: dict) -> float:
    """
    Score based on:
    1. Is the current title in the AI/ML role family?
    2. Is their career history at product companies (not pure consulting)?
    3. Is years of experience in the right range (4-12)?
    4. Do their career descriptions show production ML work?
    """
    title        = candidate.get("current_title", "").lower()
    titles_hist  = candidate.get("titles_history", [])
    companies    = candidate.get("companies_history", [])
    years_exp    = candidate.get("years_exp", 0)
    descriptions = " ".join(candidate.get("career_descriptions", [])).lower()

    score = 0.0

    # Title matching 
    # Check current title + history titles
    all_titles = [title] + titles_hist
    max_title_score = 0.0
    for t in all_titles:
        ts = _title_ai_score(t)
        max_title_score = max(max_title_score, ts)

    score += max_title_score * 0.40   # title is 40% of career_fit

    #  Product vs consulting company history
    product_company_fraction = _product_company_ratio(companies)
    score += product_company_fraction * 0.25

    # Years of experience in range
    year_score = _years_score(years_exp)
    score += year_score * 0.15

    #  Production ML signals in descriptions
    production_score = _production_signals_score(descriptions)
    score += production_score * 0.20

    return min(score, 1.0)


def _trajectory_score(candidate: dict) -> float:
    """
    This catches candidates who started in other fields and transitioned to ML.
    It also penalizes candidates who are MOVING AWAY from ML (e.g., into management).

    Method:
    - Score each role 0-1 for AI-relevance based on title + description
    - If score[recent] > score[older] → positive trajectory
    - If score[recent] < score[older] → negative trajectory
    """
    career_raw = candidate.get("career_raw", [])
    if not career_raw:
        return 0.3   # no history 

    # Already sorted by date by profile_parser
    role_scores = []
    for role in career_raw[:5]:    
        role_title = role.get("title", "").lower()
        role_desc  = role.get("description", "").lower()
        role_score = _role_ai_relevance(role_title, role_desc)
        role_scores.append(role_score)

    if len(role_scores) == 0:
        return 0.3

    if len(role_scores) == 1:
        return role_scores[0]

    # Most recent role score
    recent_score = role_scores[0]

    # Average of older roles
    older_avg = sum(role_scores[1:]) / len(role_scores[1:])

    # Trajectory = current relevance + improvement bonus
    trajectory = recent_score * 0.6 + max(0, recent_score - older_avg) * 0.4

    return min(max(trajectory, 0.0), 1.0)


def _education_cert_score(candidate: dict, signals_raw: dict) -> float:
    """
    Score based on:
    1. Education tier
    2. Relevant field of study (CS, Maths, Stats, EE)
    3. Relevant certifications
    4. Skill assessment scores on ML topics (from Redrob platform)
    """
    score = 0.0
    best_edu = candidate.get("best_education", {})

    # Education tier 
    tier_score = EDUCATION_TIER_SCORES.get(best_edu.get("tier", "unknown"), 0.4)
    score += tier_score * 0.35

    # Relevant field of study 
    field = best_edu.get("field", "").lower()
    field_relevant = any(rf in field for rf in RELEVANT_FIELDS)
    score += (0.4 if field_relevant else 0.1) * 0.25

    # Certifications 
    certs = candidate.get("certs", [])
    recent_year_threshold = 2021
    relevant_cert_score = 0.0
    for cert in certs:
        cert_name = cert.get("name", "").lower()
        cert_year = cert.get("year", 0)
        is_relevant = any(kw in cert_name for kw in RELEVANT_CERT_KEYWORDS)
        is_recent   = cert_year >= recent_year_threshold
        if is_relevant and is_recent:
            relevant_cert_score = max(relevant_cert_score, 1.0)
        elif is_relevant:
            relevant_cert_score = max(relevant_cert_score, 0.6)
    score += relevant_cert_score * 0.20

    # Skill assessment scores 
    assessment_scores = signals_raw.get("skill_assessment_scores", {})
    relevant_assessments = []
    for skill_name, skill_score in assessment_scores.items():
        skill_lower = skill_name.lower()
        if any(kw in skill_lower for kw in ["machine learning", "nlp", "python",
                                              "deep learning", "data science",
                                              "algorithms", "retrieval"]):
            relevant_assessments.append(float(skill_score) / 100.0)

    if relevant_assessments:
        assessment_avg = sum(relevant_assessments) / len(relevant_assessments)
        score += assessment_avg * 0.20

    return min(score, 1.0)


# Helper Functions

def _title_ai_score(title: str) -> float:
    """Score a job title for AI/ML relevance. Returns 0.0-1.0."""
    title = title.lower()

    # Top-tier: exactly what the JD is looking for
    if any(t in title for t in ["ml engineer", "machine learning engineer",
                                  "ai engineer", "nlp engineer", "ranking engineer",
                                  "search engineer", "applied scientist",
                                  "recommendation engineer", "research engineer"]):
        return 1.0

    # High relevance: strong signal
    if any(t in title for t in ["data scientist", "deep learning", "mlops",
                                  "ml platform", "senior ml", "senior ai",
                                  "staff ml", "principal ml"]):
        return 0.85

    # Medium: could be relevant depending on descriptions
    if any(t in title for t in ["data engineer", "software engineer", "sde",
                                  "backend engineer", "senior engineer",
                                  "full stack", "platform engineer",
                                  "tech lead", "engineering lead"]):
        return 0.55

    # Low: tangential roles
    if any(t in title for t in ["junior", "intern", "analyst", "consultant",
                                  "architect"]): 
        return 0.25

    # Non-technical roles — score near zero
    if any(t in title for t in NON_TECHNICAL_ROLE_KEYWORDS):
        return 0.0

    # Unknown / other
    return 0.30


def _role_ai_relevance(title: str, description: str) -> float:
    """Score a single role (title + description) for AI/ML relevance."""
    combined = title + " " + description

    # Count AI core skill mentions in description
    hits = sum(1 for skill in AI_CORE_SKILLS if skill in combined)
    skill_score = min(hits / 8.0, 1.0)   # 8+ hits = full score

    # Title score
    title_score = _title_ai_score(title)

    # Production signal bonus
    prod_hits = sum(1 for sig in AI_PRODUCTION_SIGNALS if sig in description)
    prod_bonus = min(prod_hits / 3.0, 0.3)

    return min((title_score * 0.5) + (skill_score * 0.35) + prod_bonus, 1.0)


def _product_company_ratio(companies: list) -> float:
    """
    What fraction of the candidate's companies are product companies?
   
    """
    if not companies:
        return 0.5   # unknown → neutral

    product_count = 0
    for company in companies:
        company_lower = company.lower()
        is_consulting = any(c in company_lower for c in CONSULTING_COMPANIES)
        if not is_consulting:
            product_count += 1

    return product_count / len(companies)


def _years_score(years: float) -> float:
    """
    Score years of experience for the 5-9 year sweet spot.
    JD says: "consider outside the band if other signals are strong" — so we don't
    hard-cutoff, we just give lower scores outside the sweet spot.
    """
    if 4 <= years <= 10:
        return 1.0   
    elif 3 <= years < 4 or 10 < years <= 13:
        return 0.75
    elif 2 <= years < 3 or 13 < years <= 15:
        return 0.5
    elif years >= 15:
        return 0.35   # over-experienced — harder to sell on startup culture
    else:
        return 0.25   # under-experienced


def _production_signals_score(descriptions: str) -> float:
    """How much evidence of real production ML work is in the descriptions?"""
    hits = sum(1 for sig in AI_PRODUCTION_SIGNALS if sig in descriptions)
    return min(hits / 5.0, 1.0)   # 5+ hits = full score
