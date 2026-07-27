from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = ROOT / ".chroma"

COLLECTION_NAME = "groundedops_documents"
EMBED_MODEL = "embed-v4.0"
RERANK_MODEL = "rerank-v4.0-fast"
CHAT_MODEL = "command-a-plus-05-2026"

RETRIEVE_K = 20
RERANK_K = 6
