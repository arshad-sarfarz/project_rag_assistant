from pathlib import Path

# Configuration constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DB_NAME = "db"
DEFAULT_COLLECTION_NAME = "documents"
BATCH_SIZE = 100
# Change this to your data folder path
FOLDER_PATH = Path.cwd() / "data"
