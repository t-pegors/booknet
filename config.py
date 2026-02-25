import os
from dotenv import load_dotenv

# Load local overrides from .env if it exists
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths
# =====================================
GOODREADS_CSV_PATTERN = os.environ.get("GOODREADS_CSV_PATTERN", "data/raw/*goodreads_library_export.csv")
LIBRARY_PATH = os.environ.get("LIBRARY_JSON_PATH", "data/processed/my_library.json")
GOLDEN_PATH = os.environ.get("GOLDEN_DATASET_PATH", "data/processed/golden_dataset.json")
EVAL_DIR = os.environ.get("EVAL_DIR", "data/evals")
LOG_DIR = os.environ.get("LOG_DIR", "data/logs")

ENRICHMENT_DATA_DIR = os.path.join(BASE_DIR, "data", "enrichment")
LOCATIONS_PATH = os.path.join(ENRICHMENT_DATA_DIR, "locations.json")
SPHERES_PATH   = os.path.join(ENRICHMENT_DATA_DIR, "spheres.json")
GENRES_PATH    = os.path.join(ENRICHMENT_DATA_DIR, "top_level_genres.json")
ERAS_PATH      = os.path.join(ENRICHMENT_DATA_DIR, "eras.json")

# Module 1: CSV Extractor Configuration
# =====================================
FILTER_READ_ONLY=os.environ.get("FILTER_READ_ONLY", "true").lower() == "true"

# Module 2: Google Books API Configuration
# =======================================
GOOGLE_FORCE_UPDATE=os.environ.get("GOOGLE_FORCE_UPDATE", "false").lower() == "true"

# Module 3: AI Enrichment
# =========================
OLLAMA_BASE_URL=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL=os.environ.get("OLLAMA_MODEL", "mistral-nemo:12b")
TEMPERATURE=os.environ.get("TEMPERATURE", "0.1")
IS_REASONING_MODEL=os.environ.get("IS_REASONING_MODEL", "false").lower() == "true"
GOLDEN_DATASET_PATH=os.environ.get("GOLDEN_DATASET_PATH", "data/processed/golden_dataset.json")

# Model Comparison Config 
ADD_REASONING_OUTPUT = os.environ.get("ADD_REASONING_OUTPUT", "false").lower() == "true"
ADD_STEP_TIMING = os.environ.get("ADD_STEP_TIMING", "true").lower() == "true"
MODELS_TO_TEST = ["gemma3:12b", "mistral-nemo:12b", "llama3.1:8b"] # non-reasoning models
#MODELS_TO_TEST = ["deepseek-r1:14b", "phi4-reasoning"] # reasoning models
SAMPLE_SIZE=5

