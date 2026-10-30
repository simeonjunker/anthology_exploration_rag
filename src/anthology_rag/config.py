"""Basic setup, read from environment variables (or a .env file); see .env.example."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# LLM server (OpenAI-compatible API served by llama.cpp or vLLM)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8345/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
ENABLE_THINKING = False

# Database
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", "chroma_db"))
# Local anthology clone; unset means the acl-anthology package's default location.
ANTHOLOGY_REPO_PATH = Path(p) if (p := os.getenv("ANTHOLOGY_REPO_PATH")) else None
# Index only the N most recent papers (faster builds); unset means all papers.
MAX_PAPERS = int(n) if (n := os.getenv("MAX_PAPERS")) else None
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "256"))

# Retrieval
# Without an explicit count: return papers with cosine similarity >= MIN_SIMILARITY, but at
# least MIN_RESULTS and at most MAX_RESULTS. MAX_RESULTS is also the page size for "more".
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.5"))
MIN_RESULTS = int(os.getenv("MIN_RESULTS", "3"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))
# Authors are matched after the vector search, so fetch more candidates.
NUM_CANDIDATES_WITH_AUTHORS = int(os.getenv("NUM_CANDIDATES_WITH_AUTHORS", "300"))
# Only the last N messages are sent to the LLM, to keep the context small.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
