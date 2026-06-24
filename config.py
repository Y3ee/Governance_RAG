import os
from dotenv import load_dotenv

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INDEX_DIR = os.path.join(BASE_DIR, "index")

load_dotenv(os.path.join(BASE_DIR, ".env"))

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# RAG Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-2"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY not found in environment. Please set it in your .env file.")
