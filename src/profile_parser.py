
import json
import re
from datetime import date, datetime
from typing import Iterator


def parse_candidate(line: str) -> dict | None:
   
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None

    cid       = raw.get("candidate_id", "")
    profile   = raw.get("profile", {})
    career    = raw.get("career_history", [])
    education = raw.get("education", [])
    skills    = raw.get("skills", [])
    certs     = raw.get("certifications", [])
    signals   = raw.get("redrob_signals", {})

    # ── Profile basics ────────────────────────────────────────────────────────
    current_title   = profile.get("current_title", "").lower()
    current_company = profile.get("current_company", "").lower()
    current_industry = profile.get("current_industry", "")
    years_exp       = float(profile.get("years_of_experience") or 0)
    summary         = profile.get("summary", "")
    headline        = profile.get("headline", "")

    # ── Career history ────────────────────────────────────────────────────────
   
    def parse_date(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return date.min

    sorted_career = sorted(
        career,
        key=lambda r: parse_date(r.get("start_date", "")),
        reverse=True
    )

    # Extract companies and titles from history
    companies_history = [r.get("company", "").lower() for r in sorted_career]
    titles_history    = [r.get("title", "").lower()   for r in sorted_career]
    industries_history = [r.get("industry", "")       for r in sorted_career]

    # what they actually DID
    career_descriptions = []
    for r in sorted_career:
        role_text = f"{r.get('title', '')} at {r.get('company', '')}: {r.get('description', '')}"
        career_descriptions.append(role_text)

    #  Skills 
    skill_list = []
    for s in skills:
        name        = s.get("name", "")
        proficiency = s.get("proficiency", "beginner")
        duration_m  = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)
        skill_list.append({
            "name":         name,
            "proficiency":  proficiency,
            "duration_m":   duration_m,
            "endorsements": endorsements,
        })
    skills_text = " | ".join(s["name"] for s in skill_list)

    # Education 
    best_edu = _get_best_education(education)

    # Certifications 
    certs_list = [
        {"name": c.get("name", ""), "year": c.get("year", 0)}
        for c in certs
    ]

    # raw, normalized later by signals_parser.py
    return {
        "candidate_id":       cid,
        "current_title":      current_title,
        "current_company":    current_company,
        "current_industry":   current_industry,
        "years_exp":          years_exp,
        "summary":            summary,
        "headline":           headline,
        "companies_history":  companies_history,
        "titles_history":     titles_history,
        "industries_history": industries_history,
        "career_descriptions": career_descriptions,   
        "career_raw":         sorted_career,          
        "skill_list":         skill_list,
        "skills_text":        skills_text,
        "best_education":     best_edu,
        "certs":              certs_list,
        "redrob_signals":     signals,
    }


def _get_best_education(education: list) -> dict:
    
    tier_order = {"tier_1": 5, "tier_2": 4, "tier_3": 3, "tier_4": 2, "unknown": 1}

    best = {
        "tier": "unknown",
        "field": "",
        "degree": "",
        "institution": "",
    }
    best_score = 0

    for edu in education:
        tier  = edu.get("tier", "unknown")
        score = tier_order.get(tier, 1)
        if score > best_score:
            best_score = score
            best = {
                "tier":        tier,
                "field":       edu.get("field_of_study", "").lower(),
                "degree":      edu.get("degree", ""),
                "institution": edu.get("institution", ""),
            }
    return best


def stream_candidates(filepath: str) -> Iterator[dict]:
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = parse_candidate(line)
            if candidate:
                yield candidate


def build_candidate_text_for_embedding(candidate: dict) -> str:
   
    parts = []

    if candidate.get("headline"):
        parts.append(candidate["headline"])

    if candidate.get("summary"):
        parts.append(candidate["summary"])

    # Career descriptions 
    for desc in candidate.get("career_descriptions", []):
        if desc.strip():
            parts.append(desc.strip())

    # Skills text as supporting context
    if candidate.get("skills_text"):
        parts.append("Skills: " + candidate["skills_text"])

    return " ".join(parts)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from config import CANDIDATES_FILE

    print("Reading first 3 candidates...")
    for i, cand in enumerate(stream_candidates(CANDIDATES_FILE)):
        print(f"\n--- {cand['candidate_id']} ---")
        print(f"  Title:    {cand['current_title']}")
        print(f"  Company:  {cand['current_company']}")
        print(f"  Years:    {cand['years_exp']}")
        print(f"  Skills:   {cand['skills_text'][:80]}")
        print(f"  Edu tier: {cand['best_education']['tier']}")
        if i >= 2:
            break
