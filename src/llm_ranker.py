
import json
import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GEMINI_API_KEY, GEMINI_MODEL

# JD context we always include in the prompt
JD_CONTEXT = """
Role: Senior AI Engineer (Founding Team) at Redrob AI
Location: Pune/Noida, India | Hybrid | 5-9 years experience

MUST HAVE (hard requirements):
- Production experience with embedding-based retrieval systems (sentence-transformers, BGE, E5)
- Production experience with vector databases / hybrid search (FAISS, Pinecone, Weaviate, Elasticsearch)
- Strong Python, production-grade code
- Hands-on evaluation framework design (NDCG, MRR, MAP, A/B testing)

IDEAL PROFILE:
- 6-8 years, of which 4-5 years in applied ML at PRODUCT companies (not IT services)
- Built and shipped end-to-end ranking/search/recommendation systems to real users
- Active on platform, willing to relocate to Noida/Pune

EXPLICIT DISQUALIFIERS (from JD):
- Pure consulting career (only TCS/Infosys/Wipro/Accenture/Cognizant etc.)
- AI experience = only LangChain wrappers calling OpenAI with no pre-LLM production ML
- Not written production code in 18 months (moved to architecture/tech-lead only)
- Pure research without production deployment
- Only CV/Speech/Robotics with no NLP/IR

CRITICAL ANTI-TRAP INSTRUCTION:
The right answer is NOT the candidate with the most AI keywords.
A Marketing Manager with RAG/LLM/Pinecone in their skills = NOT a fit.
A backend engineer who built a recommendation system "without saying RAG" = IS a fit.
Reason about SUBSTANCE and PRODUCTION TRACK RECORD, not keyword presence.
"""


def rerank_with_gemini(
    candidates: list[dict],
    batch_size: int = 5,
    delay_between_batches: float = 1.0,
) -> list[dict]:
    """
    Re-rank candidates using Gemini.

    For each candidate, Gemini returns:
      - score: 0.0–1.0 (hiring quality score)
      - reasoning: one sentence, recruiter voice

    Args:
        candidates: list of scored candidate dicts (top-200 from scorer)
        batch_size: process N candidates per API call (cheaper, faster)
        delay_between_batches: seconds to wait between calls (rate limit)

    Returns:
        Same list, each dict now has 'llm_score' and 'reasoning' added
    """
    if not GEMINI_API_KEY:
        print("  WARNING: GEMINI_API_KEY not set — using rule-based reasoning fallback")
        return _fallback_reasoning(candidates)

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    results = []
    total_batches = (len(candidates) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(candidates), batch_size):
        batch = candidates[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        print(f"  LLM re-ranking batch {batch_num}/{total_batches} "
              f"({len(batch)} candidates)...", end="", flush=True)

        try:
            batch_results = _rank_batch(model, batch)
            results.extend(batch_results)
            print(f" done")
        except Exception as e:
            print(f" ERROR: {e}")
           
            for cand in batch:
                cand["llm_score"] = cand.get("final_score", 0.5)
                cand["reasoning"] = _rule_based_reasoning(cand)
            results.extend(batch)

        # Respect rate limits
        if batch_idx + batch_size < len(candidates):
            time.sleep(delay_between_batches)

    return results


def _rank_batch(model, batch: list[dict]) -> list[dict]:
    """Send a batch of candidates to Gemini and parse the response."""

    
    candidate_summaries = []
    for i, cand in enumerate(batch):
        summary = _build_candidate_summary(cand)
        candidate_summaries.append(f"CANDIDATE_{i+1}:\n{summary}")

    prompt = f"""You are an expert technical recruiter evaluating candidates for the following role.

{JD_CONTEXT}

For each candidate below, provide:
1. A score from 0.0 to 1.0 (where 1.0 = perfect fit for this exact role)
2. A one-sentence reasoning in recruiter voice (direct, specific, honest)

The reasoning should:
- Mention what they've ACTUALLY BUILT (not just their skills list)
- Note if they're at a product company vs consulting firm
- Flag availability/engagement if relevant
- Be honest about gaps or disqualifiers

Return ONLY a JSON array, no markdown, no explanation:
[
  {{"score": 0.85, "reasoning": "Built hybrid retrieval system at a Series B fintech with NDCG-tracked eval infrastructure; active on platform and open to Pune relocation."}},
  ...
]

{chr(10).join(candidate_summaries)}

Return the JSON array with exactly {len(batch)} entries, in order."""

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "max_output_tokens": 1024}
    )

    # Parse JSON response
    text = response.text.strip()
   
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    parsed = json.loads(text)

    # Attach llm_score and reasoning back to candidates
    for i, cand in enumerate(batch):
        if i < len(parsed):
            cand["llm_score"]  = float(parsed[i].get("score", cand.get("final_score", 0.5)))
            cand["reasoning"]  = str(parsed[i].get("reasoning", _rule_based_reasoning(cand)))
        else:
            cand["llm_score"]  = cand.get("final_score", 0.5)
            cand["reasoning"]  = _rule_based_reasoning(cand)

    return batch


def _build_candidate_summary(cand: dict) -> str:
   
    title        = cand.get("current_title", "unknown")
    years        = cand.get("years_exp", 0)
    companies    = cand.get("companies_history", [])[:3]
    descriptions = cand.get("career_descriptions", [])[:2]
    skills_text  = cand.get("skills_text", "")[:200]
    signals      = cand.get("redrob_signals", {})

    days_active  = cand.get("days_since_active", 999)
    active_str   = (f"{days_active} days ago" if days_active < 999 else "unknown")

    desc_text = "\n".join(f"  - {d[:200]}" for d in descriptions)

    return (
        f"Title: {title} | {years:.1f} years experience\n"
        f"Companies: {', '.join(companies)}\n"
        f"Recent role descriptions:\n{desc_text}\n"
        f"Skills: {skills_text}\n"
        f"Last active: {active_str} | "
        f"Open to work: {signals.get('open_to_work_flag', False)} | "
        f"Response rate: {signals.get('recruiter_response_rate', 0):.0%} | "
        f"GitHub score: {signals.get('github_activity_score', -1)}\n"
        f"Disqualifier flags: {cand.get('disqualifier_flags', [])}"
    )


def _rule_based_reasoning(cand: dict) -> str:
    """
    Generate a rule-based reasoning string when Gemini is unavailable.
    Still informative — references actual career data.
    """
    title       = cand.get("current_title", "unknown").title()
    years       = cand.get("years_exp", 0)
    companies   = cand.get("companies_history", [])
    flags       = cand.get("disqualifier_flags", [])
    career_fit  = cand.get("career_fit", 0)
    behavioral  = cand.get("behavioral", 0)
    trajectory  = cand.get("trajectory", 0)

    parts = []

    if career_fit >= 0.7:
        parts.append(f"{title} with strong ML/AI engineering background")
    elif career_fit >= 0.4:
        parts.append(f"{title} with partial ML/AI background")
    else:
        parts.append(f"{title} — limited ML/AI engineering alignment")

    parts.append(f"{years:.1f} years experience")

    if companies:
        parts.append(f"at {companies[0].title()}")

    if trajectory >= 0.7:
        parts.append("trajectory clearly moving toward ML/AI systems")
    elif trajectory >= 0.4:
        parts.append("some trajectory toward AI engineering")

    if behavioral >= 0.7:
        parts.append("active and responsive on platform")
    elif behavioral < 0.3:
        parts.append("low engagement — may be passive candidate")

    if "pure_consulting_career" in flags:
        parts.append("entire career in IT services — no product company experience")
    if "non_technical_role" in flags:
        parts.append("non-technical role — does not align with engineering requirement")
    if "ghost_candidate" in flags:
        parts.append("inactive for 6+ months with low response rate")

    return "; ".join(parts) + "."


def _fallback_reasoning(candidates: list[dict]) -> list[dict]:
    """Apply rule-based reasoning to all candidates (Gemini unavailable)."""
    for cand in candidates:
        cand["llm_score"] = cand.get("final_score", 0.5)
        cand["reasoning"] = _rule_based_reasoning(cand)
    return candidates


if __name__ == "__main__":
    # Quick test with fake candidates
    test_candidates = [
        {
            "candidate_id": "CAND_TEST_001",
            "current_title": "ml engineer",
            "years_exp": 6.5,
            "companies_history": ["razorpay", "flipkart"],
            "career_descriptions": [
                "Built real-time recommendation engine using dense retrieval + BM25 hybrid search. "
                "Deployed to 10M+ users. Maintained FAISS index with weekly refresh pipeline."
            ],
            "skills_text": "Python, PyTorch, FAISS, Elasticsearch, A/B testing, NDCG",
            "final_score": 0.87,
            "career_fit": 0.9,
            "trajectory": 0.85,
            "behavioral": 0.7,
            "disqualifier_flags": [],
            "days_since_active": 15,
            "redrob_signals": {
                "open_to_work_flag": True,
                "recruiter_response_rate": 0.8,
                "github_activity_score": 72,
            }
        }
    ]
    print("Testing LLM ranker with 1 candidate...")
    results = rerank_with_gemini(test_candidates, batch_size=1)
    print(f"Score: {results[0]['llm_score']}")
    print(f"Reasoning: {results[0]['reasoning']}")
