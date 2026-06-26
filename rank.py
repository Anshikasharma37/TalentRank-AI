"""
This script runs the full pipeline end-to-end:
  1. Parse JD
  2. Stream 100K candidates → rule-based fast prefilter → top 3000
  3. Embed top 3000 candidates + JD → FAISS index
  4. Semantic search → top 500
  5. Full 5-dimension scoring → top 200
  6. Gemini LLM re-ranking → top 100 with recruiter reasoning
  7. Write submission.csv
"""

import os
import sys
import csv
import json
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    CANDIDATES_FILE, JD_FILE, OUTPUT_FILE, CACHE_DIR,
    FAST_PREFILTER_KEEP, SEMANTIC_KEEP, LLM_RERANK_KEEP, FINAL_SUBMISSION
)
from src.jd_parser      import parse_jd, get_jd_embedding_text
from src.profile_parser  import stream_candidates, build_candidate_text_for_embedding
from src.scorer          import score_candidate, fast_prefilter_score
from src.embedder        import embed_texts, embed_single, build_jd_embedding
from src.vector_store    import build_index, search_index, save_index, load_index
from src.llm_ranker      import rerank_with_gemini


def run_pipeline(
    use_cache: bool = True,
    skip_llm: bool = False,
):
    """
    Args:
        use_cache:  if True and cache exists, skip embedding step
        skip_llm:   if True, skip Gemini re-ranking (faster, less impressive)
    """
    start_time = time.time()
    print("\n" + "="*60)
    print("  TalentRank AI — Candidate Ranking Pipeline")
    print("="*60)

    # Step 1: Parse JD 
    print("\n[1/6] Parsing job description...")
    jd = parse_jd(JD_FILE)
    print(f"  JD parsed: {len(jd['requirements'])} chars in requirements section")

    # Step 2: Rule-based fast prefilter 
    print(f"\n[2/6] Fast prefilter: scoring all candidates (rule-based, no embeddings)...")
    print(f"  Streaming {CANDIDATES_FILE}...")

    fast_scored = []
    total_read  = 0
    skipped     = 0

    for candidate in stream_candidates(CANDIDATES_FILE):
        total_read += 1

        fast_score = fast_prefilter_score(candidate)

       
        candidate["_fast_score"] = fast_score
        fast_scored.append(candidate)

        if total_read % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"  Read {total_read:,} candidates ({elapsed:.0f}s)...")

    # Sort by fast score, keep top N
    fast_scored.sort(key=lambda c: c["_fast_score"], reverse=True)
    top_candidates = fast_scored[:FAST_PREFILTER_KEEP]

    print(f"  Total read: {total_read:,}")
    print(f"  Kept top {len(top_candidates):,} by fast score")
    print(f"  Top fast score: {top_candidates[0]['_fast_score']:.4f}")
    print(f"  Fast filter done in {time.time()-start_time:.0f}s")

    # Step 3: Embed candidates + build FAISS index
    candidate_ids = [c["candidate_id"] for c in top_candidates]
    index, embeddings, cached_ids = None, None, None

    if use_cache:
        index, embeddings, cached_ids = load_index(CACHE_DIR)

    if index is None or cached_ids != candidate_ids:
        print(f"\n[3/6] Embedding {len(top_candidates):,} candidates...")
        embed_start = time.time()

        texts = [build_candidate_text_for_embedding(c) for c in top_candidates]
        embeddings = embed_texts(texts, batch_size=64, show_progress=True)

        print(f"  Embedding done: shape={embeddings.shape} | {time.time()-embed_start:.0f}s")

        print(f"\n[4/6] Building FAISS index...")
        index = build_index(embeddings)

        if use_cache:
            save_index(index, embeddings, candidate_ids, CACHE_DIR)
    else:
        print(f"\n[3/6] Using cached embeddings ({embeddings.shape})")
        print(f"\n[4/6] Using cached FAISS index")

    # Step 4: Semantic search — top 500 
    print(f"\n[4/6] Semantic search: retrieving top {SEMANTIC_KEEP} candidates...")
    jd_embedding = build_jd_embedding(jd)

    distances, top_indices = search_index(index, jd_embedding, top_k=SEMANTIC_KEEP)

    semantic_top = []
    for rank_idx, (dist, cand_idx) in enumerate(zip(distances, top_indices)):
        if cand_idx >= len(top_candidates):
            continue
        cand = top_candidates[cand_idx]
        cand["_semantic_sim"]       = float(dist)
        cand["_candidate_embedding"] = embeddings[cand_idx]
        semantic_top.append(cand)

    print(f"  Semantic search returned {len(semantic_top)} candidates")
    print(f"  Top semantic sim: {semantic_top[0]['_semantic_sim']:.4f}")

    # Step 5: Full 5-dimension scoring 
    print(f"\n[5/6] Full 5-dimension scoring on {len(semantic_top)} candidates...")

    scored = []
    for cand in semantic_top:
        result = score_candidate(
            cand,
            jd_embedding=jd_embedding,
            candidate_embedding=cand.get("_candidate_embedding"),
        )
        # Merge back — keep full candidate data for LLM step
        result.update({
            "companies_history":   cand.get("companies_history", []),
            "titles_history":      cand.get("titles_history", []),
            "career_descriptions": cand.get("career_descriptions", []),
            "skills_text":         cand.get("skills_text", ""),
            "redrob_signals":      cand.get("redrob_signals", {}),
        })
        scored.append(result)

    
    scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    # Keep top candidates for LLM re-ranking
    top_for_llm = scored[:LLM_RERANK_KEEP]

    print(f"  Scoring done. Top final_score: {top_for_llm[0]['final_score']:.4f}")
    print(f"  Hard disqualified in top {SEMANTIC_KEEP}: "
          f"{sum(1 for s in scored if s['is_disqualified'])}")

    # Step 6: LLM re-ranking 
    if skip_llm:
        print(f"\n[6/6] Skipping LLM re-ranking (--skip-llm flag set)")
        final_ranked = top_for_llm[:FINAL_SUBMISSION]
        for cand in final_ranked:
            cand["reasoning"] = _build_fallback_reasoning(cand)
            cand["llm_score"] = cand["final_score"]
    else:
        print(f"\n[6/6] LLM re-ranking: sending top {LLM_RERANK_KEEP} to Gemini...")
        reranked = rerank_with_gemini(top_for_llm, batch_size=5)

        # Final sort: blend llm_score (60%) + final_score (40%)
        for cand in reranked:
            cand["blend_score"] = (
                cand["llm_score"]  * 0.60 +
                cand["final_score"] * 0.40
            )

        reranked.sort(key=lambda r: (-r["blend_score"], r["candidate_id"]))
        final_ranked = reranked[:FINAL_SUBMISSION]

    # Step 7: Write submission.csv 
    print(f"\n[7/7] Writing {OUTPUT_FILE}...")
    _write_submission(final_ranked)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total_time/60:.1f} minutes")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Top 5 candidates:")
    for i, cand in enumerate(final_ranked[:5], 1):
        score = cand.get("blend_score", cand.get("final_score", 0))
        print(f"    {i}. {cand['candidate_id']} | {cand['current_title']!r} | "
              f"score={score:.4f} | {cand['years_exp']:.1f} yrs")
    print("="*60 + "\n")


def _write_submission(ranked: list[dict]):
    """Write the final submission CSV with required columns."""
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, cand in enumerate(ranked, 1):
            candidate_id = cand["candidate_id"]
            score        = round(cand.get("blend_score", cand.get("final_score", 0)), 4)
            reasoning    = cand.get("reasoning", "").replace("\n", " ").strip()

            writer.writerow([candidate_id, rank, score, reasoning])

    print(f"  Written {len(ranked)} rows to {OUTPUT_FILE}")

    # Also save detailed scores for Streamlit dashboard radar charts
    _save_detailed_scores(ranked)


def _save_detailed_scores(ranked: list[dict]):

    """Save dimension-level scores to .cache/ so Streamlit can show radar charts."""
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "detailed_scores.json")
    
    detailed = []
    for cand in ranked:
        detailed.append({
            "candidate_id":      cand.get("candidate_id"),
            "current_title":     cand.get("current_title"),
            "years_exp":         cand.get("years_exp"),
            "semantic_fit":      cand.get("semantic_fit", 0),
            "career_fit":        cand.get("career_fit", 0),
            "trajectory":        cand.get("trajectory", 0),
            "behavioral":        cand.get("behavioral", 0),
            "education_cert":    cand.get("education_cert", 0),
            "final_score":       cand.get("final_score", 0),
            "disqualifier_flags": cand.get("disqualifier_flags", []),
            "open_to_work":      cand.get("open_to_work", False),
            "days_since_active": cand.get("days_since_active", 999),
        })
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)
    print(f"  Detailed scores saved to {cache_path}")


def _build_fallback_reasoning(cand: dict) -> str:

    """Quick reasoning string if LLM is skipped."""

    title       = (cand.get("current_title") or "unknown").title()
    years       = cand.get("years_exp", 0)
    companies   = cand.get("companies_history", [])
    career_fit  = cand.get("career_fit", 0)
    trajectory  = cand.get("trajectory", 0)
    behavioral  = cand.get("behavioral", 0)

    parts = [f"{title}, {years:.1f} years"]
    if companies:
        parts.append(f"at {companies[0].title()}")
    if career_fit >= 0.7:
        parts.append("strong ML/AI engineering background")
    if trajectory >= 0.6:
        parts.append("clear trajectory toward AI systems")
    if behavioral >= 0.6:
        parts.append("active on platform")
    elif behavioral < 0.3:
        parts.append("low engagement — passive candidate")
    for flag in cand.get("disqualifier_flags", []):
        if flag == "pure_consulting_career":
            parts.append("entire career in IT services")
        elif flag == "non_technical_role":
            parts.append("non-technical role — does not align with JD")

    return "; ".join(parts) + "."


if __name__ == "__main__":
    # Check for --skip-llm flag
    skip_llm = "--skip-llm" in sys.argv
    no_cache  = "--no-cache" in sys.argv

    run_pipeline(
        use_cache=not no_cache,
        skip_llm=skip_llm,
    )
