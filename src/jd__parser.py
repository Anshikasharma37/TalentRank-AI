import docx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import JD_FILE


def parse_jd(jd_path: str = JD_FILE) -> dict:
   
    doc = docx.Document(jd_path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)

    # Split into sections by heading keywords 
    sections = {
        "requirements":      [],
        "responsibilities":  [],
        "preferred":         [],
        "ideal_profile":     [],
        "disqualifiers":     [],
    }

    current_section = None
    for para in paragraphs:
        lower = para.lower()

        # Detect section changes
        if "things you absolutely need" in lower or "skills inventory" in lower:
            current_section = "requirements"
        elif "what you'd actually be doing" in lower or "first 90 days" in lower:
            current_section = "responsibilities"
        elif "things we'd like you to have" in lower:
            current_section = "preferred"
        elif "things we explicitly do not want" in lower:
            current_section = "disqualifiers"
        elif "how to read between the lines" in lower or "ideal candidate" in lower:
            current_section = "ideal_profile"
        elif current_section:
            sections[current_section].append(para)

    
    # Weighted: requirements > responsibilities > preferred
    ideal_text = (
        "Senior AI Engineer role requiring: "
        "Production experience with embeddings-based retrieval systems. "
        "Vector database and hybrid search infrastructure in production. "
        "Designing evaluation frameworks for ranking systems — NDCG, MRR, MAP. "
        "Strong Python. Shipping ranking and recommendation systems to real users. "
        "Applied ML at product companies, not pure consulting. "
        "Hybrid retrieval, dense retrieval, re-ranking, LLM integration. "
        "6-8 years total experience with 4-5 years in applied ML at product companies. "
        "Built end-to-end ranking or recommendation or search system at scale. "
        "Opinions on retrieval architecture, evaluation, and LLM integration. "
        "Active job seeker, willing to relocate to Noida or Pune."
    )

    return {
        "full_text":        full_text,
        "requirements":     "\n".join(sections["requirements"]),
        "responsibilities": "\n".join(sections["responsibilities"]),
        "preferred":        "\n".join(sections["preferred"]),
        "ideal_profile":    ideal_text,  
        "disqualifiers":    sections["disqualifiers"],
    }


def get_jd_embedding_text(jd: dict) -> str:
   
    return (
        # Requirements (40% weight → repeated 2x)
        jd["requirements"] + "\n" + jd["requirements"] + "\n"
        # Ideal profile ( most focused)
        + jd["ideal_profile"] + "\n"
        # Responsibilities (35% weight)
        + jd["responsibilities"] + "\n"
        # Preferred (25% weight)
        + jd["preferred"]
    )


if __name__ == "__main__":
    jd = parse_jd()
    print("=== JD Sections ===")
    for k, v in jd.items():
        if isinstance(v, list):
            print(f"\n[{k}] ({len(v)} items):")
            for item in v[:3]:
                print(f"  - {item[:80]}")
        else:
            print(f"\n[{k}]:\n{v[:300]}...")