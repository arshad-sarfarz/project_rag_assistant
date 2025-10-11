from pathlib import Path

# Configuration constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB_NAME = "rag_db"
DEFAULT_COLLECTION_NAME = "design_docs"
BATCH_SIZE = 100
# Change this to your data folder path
DATA_DIR = str(Path.cwd() / "data")
OUTPUTS_DIR = str(Path.cwd() / "output")
