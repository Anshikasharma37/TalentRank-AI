
import os

DATA_DIR = os.path.join(
    "dataset", "extracted",
    "[PUB] India_runs_data_and_ai_challenge",
    "India_runs_data_and_ai_challenge"
)

CANDIDATES_FILE  = os.path.join(DATA_DIR, "candidates.jsonl")
JD_FILE          = os.path.join(DATA_DIR, "job_description.docx")
OUTPUT_FILE      = "submission.csv"
CACHE_DIR        = ".cache"   # stores FAISS index + embeddings between runs


WEIGHTS = {
    "semantic_fit":       0.30,   # cosine sim of career text vs JD
    "career_fit":         0.25,   # role family + company type + years
    "trajectory":         0.20,   # are recent roles MORE AI-relevant than older ones?
    "behavioral":         0.15,   # platform activity signals
    "education_cert":     0.10,   # degree tier + relevant certs + assessment scores
}

# Pipeline Stage Limits
FAST_PREFILTER_KEEP  = 3000   # rule-based pass over 100K 
SEMANTIC_KEEP        = 500    # after FAISS search → keep top N
LLM_RERANK_KEEP      = 200    # full-score top N sent to Gemini
FINAL_SUBMISSION     = 100    # final CSV rows



# Titles that signal real ML/AI engineering work
AI_ROLE_KEYWORDS = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "data scientist", "nlp engineer", "nlp researcher",
    "research engineer", "applied scientist", "applied ml",
    "ranking engineer", "search engineer", "recommendation",
    "computer vision engineer",  
    "deep learning engineer", "mlops", "ml ops", "ml platform",
    "data engineer",              
    "software engineer",          
    "sde", "backend engineer",    
    "senior engineer", "staff engineer", "principal engineer",
    "tech lead",                  
]

# Titles that are clearly not what the JD is asking for
NON_TECHNICAL_ROLE_KEYWORDS = [
    "marketing", "accountant", "accounting",
    "hr ", "human resource", "human resources",
    "operations manager",
    "graphic design", "graphic designer",
    "civil engineer", "mechanical engineer",
    "sales executive", "sales manager",
    "content writer", "content manager",
    "customer support", "customer service",
    "finance manager", "financial analyst",
    "business analyst",           # penalized but not eliminated
    "project manager",            # penalized but not eliminated
    "supply chain", "procurement",
    "ui/ux", "ux designer", "product designer",
]

#  Consulting Company Detection 

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "l&t infotech", "mindtree",
    "persistent systems", "mastech", "niit technologies",
    "kpit", "coforge",
}

#  Education Tier Scores 

EDUCATION_TIER_SCORES = {
    "tier_1": 1.00,
    "tier_2": 0.80,
    "tier_3": 0.55,
    "tier_4": 0.30,
    "unknown": 0.40,
}

# Fields of study that are highly relevant for this role
RELEVANT_FIELDS = [
    "computer science", "cs", "software engineering",
    "information technology", "electronics", "electrical",
    "mathematics", "statistics", "physics",
    "data science", "artificial intelligence", "machine learning",
    "computational", "cognitive science",
]

# Relevant Certifications 

RELEVANT_CERT_KEYWORDS = [
    "machine learning", "deep learning", "nlp", "tensorflow",
    "pytorch", "aws ml", "gcp ml", "azure ml",
    "google cloud professional", "databricks", "mlops",
    "coursera ml", "fast.ai",
]

#  Behavioral Signal Thresholds 

ACTIVITY_THRESHOLDS = {
    "very_active":  30,    # logged in within 30 days → score 1.0
    "active":       90,    # within 90 days → score 0.75
    "somewhat":     180,   # within 180 days → score 0.4
    # beyond 180 days → score 0.1 (ghost candidate)
}

#  Gemini API 

GEMINI_MODEL    = "gemini-1.5-flash"  
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")  # set via env variable or in Colab

# Embedding Model

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
