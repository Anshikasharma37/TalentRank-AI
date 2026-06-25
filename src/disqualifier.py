"""
  1.0 = clean, no penalty
  0.5 = soft disqualifier (penalized but not eliminated)
  0.1 = hard disqualifier (effectively eliminated)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import NON_TECHNICAL_ROLE_KEYWORDS, CONSULTING_COMPANIES


def get_disqualifier_info(candidate: dict, behavioral_signals: dict) -> dict:
   
    flags = []
    penalty = 1.0   # start clean

    title          = candidate.get("current_title", "").lower()
    companies      = candidate.get("companies_history", [])
    titles_history = candidate.get("titles_history", [])
    years_exp      = candidate.get("years_exp", 0)
    skills_text    = candidate.get("skills_text", "").lower()
    career_raw     = candidate.get("career_raw", [])
    descriptions   = " ".join(candidate.get("career_descriptions", [])).lower()

    # Hard disqualifier 1: Clearly non-technical role 

    # The JD says: Marketing Manager with AI keywords = wrong answer
    is_non_technical = any(kw in title for kw in NON_TECHNICAL_ROLE_KEYWORDS)
    if is_non_technical:
        # Check if their descriptions actually show engineering work
        engineering_in_descriptions = _has_engineering_substance(descriptions)
        if engineering_in_descriptions:
            flags.append("non_technical_title_but_eng_background")
            penalty *= 0.5   # soft: penalized but not eliminated
        else:
            flags.append("non_technical_role")
            penalty *= 0.1   # hard: Marketing Manager without ML substance

    # Hard disqualifier 2: Pure consulting career

    # "People who have only worked at consulting firms in their entire career"
    all_companies_are_consulting = _is_pure_consulting(companies)
    if all_companies_are_consulting:
        flags.append("pure_consulting_career")
        penalty *= 0.15

    # Soft disqualifier 3: Ghost candidate 
    # "A perfect-on-paper candidate who hasn't logged in for 6 months and has
    #  a 5% recruiter response rate is, for hiring purposes, not actually available."
    days_inactive = behavioral_signals.get("days_since_active", 0)
    response_rate = behavioral_signals.get("response_score", 0)
    if days_inactive > 180 and response_rate < 0.15:
        flags.append("ghost_candidate")
        penalty *= 0.4

    # Soft disqualifier 4: Extreme over-experience 
    
    if years_exp > 15 and all_companies_are_consulting:
        flags.append("over_experienced_consulting")
        penalty *= 0.7

    # Soft disqualifier 5: CV/Speech-only engineer
    # JD: "People whose primary expertise is CV, speech, robotics without NLP/IR"
    is_cv_only = _is_cv_speech_only(skills_text, descriptions)
    if is_cv_only:
        flags.append("cv_speech_only_no_nlp_ir")
        penalty *= 0.6

    return {
        "penalty_multiplier": round(max(0.0, min(penalty, 1.0)), 4),
        "flags":              flags,
        "is_hard_disqualified": penalty <= 0.2,
    }


def _has_engineering_substance(descriptions: str) -> bool:
    
    engineering_signals = [
        "model", "algorithm", "pipeline", "deployment", "api",
        "machine learning", "neural", "training", "inference",
        "python", "sql", "database", "engineering", "system",
        "retrieval", "embedding", "vector", "ranking", "recommendation",
        "built", "developed", "implemented", "shipped", "production",
    ]
    hit_count = sum(1 for sig in engineering_signals if sig in descriptions)
    return hit_count >= 4   # at least 4 engineering signals


def _is_pure_consulting(companies: list) -> bool:
   
    if not companies:
        return False

    consulting_count = 0
    for company in companies:
        company_lower = company.lower()
        if any(c in company_lower for c in CONSULTING_COMPANIES):
            consulting_count += 1

    # All companies are consulting → hard disqualifier
    return consulting_count == len(companies) and len(companies) > 0


def _is_cv_speech_only(skills_text: str, descriptions: str) -> bool:
   
    cv_speech_signals = [
        "computer vision", "image classification", "object detection",
        "speech recognition", "speech synthesis", "robotics", "opencv",
        "yolo", "cnn for image",
    ]
    nlp_ir_signals = [
        "nlp", "natural language", "text", "retrieval", "ranking",
        "recommendation", "search", "embedding", "bert", "transformer",
        "question answering", "information retrieval", "vector",
    ]

    combined = skills_text + " " + descriptions
    has_cv_speech = sum(1 for s in cv_speech_signals if s in combined) >= 2
    has_nlp_ir    = sum(1 for s in nlp_ir_signals    if s in combined) >= 2

    # Only flag if they have CV/speech signals but zero NLP/IR signals
    return has_cv_speech and not has_nlp_ir


if __name__ == "__main__":
   
    test_candidate = {
        "current_title":    "marketing manager",
        "companies_history": ["tcs", "infosys"],
        "titles_history":   ["marketing manager", "marketing executive"],
        "years_exp":        8,
        "skills_text":      "RAG LLMs Python TensorFlow GPT",
        "career_raw":       [],
        "career_descriptions": [
            "Ran marketing campaigns. Managed social media. Used ChatGPT for content."
        ],
    }
    behavioral = {"days_since_active": 45, "response_score": 0.3}
    result = get_disqualifier_info(test_candidate, behavioral)
    print("Disqualifier result (should be heavily penalized):")
    print(f"  Penalty multiplier: {result['penalty_multiplier']}")
    print(f"  Flags: {result['flags']}")
    print(f"  Hard disqualified: {result['is_hard_disqualified']}")