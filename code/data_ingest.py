import os
import chromadb
import shutil
import uuid
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_docling.loader import DoclingLoader
from config import *
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import QueryRequest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_design_documents(file_paths: List[str]) -> bool:
    """
    Demonstration of the data loader functionality.

    Args:
        file_paths: List of files to process
    Returns:
        client: The ChromaDB client instance if successful, else None
    """
    try:
        # Initialize database
        coll = initialize_db(clear_dir=True)
        if not coll:
            return None

        # Load and process documents
        texts, mds = load_text_with_docling(file_paths)
        if not texts:
            logger.error("No texts were loaded")
            return None

        # Embed and store
        success = embed_and_store_texts(coll, texts, metadatas=mds)
        if not success:
            logger.error("Failed to embed and store texts")
            return None
        return success

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return None


def initialize_db(db_name: str = DEFAULT_DB_NAME, clear_dir: bool = False, collection_name: str = DEFAULT_COLLECTION_NAME) -> Optional[Collection]:
    """
    Initialize ChromaDB persistent client.

    Args:
        db_name: Name/path of the database directory
        clear_dir: Whether to clear existing database

    Returns:
        ChromaDB client instance or None if initialization fails
    """
    try:
        if clear_dir and os.path.exists(db_name):
            logger.info(f"Clearing existing database directory: {db_name}")
            shutil.rmtree(db_name)

        client = chromadb.PersistentClient(path=db_name)

        collection = client.get_or_create_collection(name=collection_name)

        logger.info(f"Successfully initialized database at: {db_name}")
        logger.info(
            f"Successfully created/accessed collection: {collection_name}")
        return collection

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")


def embed_and_store_texts(collection: Collection,
                          texts: List[str],
                          metadatas: Optional[List[Dict]] = None,
                          batch_size: int = BATCH_SIZE) -> bool:
    """
    Embed texts and store them in ChromaDB collection with batch processing.

    Args:
        collection: ChromaDB collection instance
        texts: List of text documents to embed
        metadatas: Optional list of metadata dictionaries
        batch_size: Number of documents to process in each batch

    Returns:
        True if successful, False otherwise
    """
    if not texts:
        logger.warning("No texts provided for embedding")
        return False

    metadatas = metadatas or [{}] * len(texts)
    metadatas = [_sanitize_metadata(m) for m in metadatas]

    if len(texts) != len(metadatas):
        logger.error("Mismatch between number of texts and metadatas")

    try:
        # Initialize embeddings model
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
        hf = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        # Process in batches to manage memory
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_num = (i // batch_size) + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            # Filter out empty texts
            valid_indices = [j for j, text in enumerate(
                batch_texts) if text.strip()]
            if not valid_indices:
                logger.warning(
                    f"Batch {batch_num} contains no valid texts, skipping")
                continue

            valid_texts = [batch_texts[j] for j in valid_indices]
            valid_metadatas = [batch_metadatas[j] for j in valid_indices]

            # Generate embeddings
            embeddings = hf.embed_documents(valid_texts)

            # Generate deterministic IDs
            ids = [_generate_deterministic_id(
                text, md) for text, md in zip(valid_texts, valid_metadatas)]

            # Check for existing documents to avoid duplicates
            existing_ids = set()
            try:
                existing_results = collection.get(ids=ids)
                existing_ids = set(existing_results['ids'])
            except Exception as e:
                logger.debug(
                    f"Could not check for existing documents: {str(e)}")

            # Filter out existing documents
            new_indices = [j for j, doc_id in enumerate(
                ids) if doc_id not in existing_ids]
            if not new_indices:
                logger.info(
                    f"Batch {batch_num}: All documents already exist, skipping")
                continue

            new_texts = [valid_texts[j] for j in new_indices]
            new_metadatas = [valid_metadatas[j] for j in new_indices]
            new_embeddings = [embeddings[j] for j in new_indices]
            new_ids = [ids[j] for j in new_indices]

            # Store in ChromaDB
            collection.add(
                ids=new_ids,
                documents=new_texts,
                metadatas=new_metadatas,
                embeddings=new_embeddings
            )

            logger.info(
                f"Batch {batch_num}: Added {len(new_ids)} new documents")

        logger.info("Successfully completed embedding and storage")
        return True

    except Exception as e:
        logger.error(f"Failed to embed and store texts: {str(e)}")


def load_text_with_docling(file_paths: List[str]) -> Tuple[List[str], List[Dict]]:
    """
    Use Docling to parse one or more files to raw text.

    Args:
        file_paths: List of file paths to process

    Returns:
        Tuple of (texts, metadatas) lists
    """
    texts = []
    metadatas = []

    for fp in file_paths:
        if not os.path.exists(fp):
            logger.error(f"File does not exist: {fp}")
            continue

        try:
            logger.info(f"Loading file: {fp}")
            docs = DoclingLoader(fp).load()  # each doc is a LangChain Document

            for d in docs:
                if d.page_content and d.page_content.strip():  # Skip empty content
                    texts.append(d.page_content)
                    metadatas.append(d.metadata or {})

            logger.info(
                f"Successfully loaded {len([d for d in docs if d.page_content])} documents from {fp}")

        except Exception as e:
            logger.error(f"Failed to load file {fp}: {str(e)}")
            continue

    logger.info(f"Total documents loaded: {len(texts)}")
    return texts, metadatas

# region Helper Functions


def _sanitize_metadata(md: Optional[Dict]) -> Dict:
    """
    Keep only primitive fields; extract filename/page if present.

    Args:
        md: Raw metadata dictionary

    Returns:
        Sanitized metadata dictionary with only primitive types
    """
    if md is None:
        return {}

    out = {}

    try:
        # Pull useful bits if Docling provided them
        origin = md.get("origin", {})
        if isinstance(origin, dict):
            fn = origin.get("filename")
            if fn:
                out["filename"] = str(fn)

        # Docling often stores page info in a list under 'prov'
        prov = md.get("prov")
        if isinstance(prov, list) and prov:
            pg = prov[0].get("page_no")
            if isinstance(pg, (int, float, str)):
                out["page_no"] = int(pg) if isinstance(
                    pg, (int, float)) else str(pg)

        # Copy any other primitive fields
        for k, v in md.items():
            if k in ("origin", "prov"):  # already handled
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            # Convert other types to string for safety
            elif v is not None:
                out[k] = str(v)

    except Exception as e:
        logger.warning(f"Error sanitizing metadata: {str(e)}")

    return out


def _generate_deterministic_id(text: str, metadata: Dict) -> str:
    """
    Generate deterministic ID based on content and metadata to avoid duplicates.

    Args:
        text: Document text content
        metadata: Document metadata

    Returns:
        Deterministic UUID string
    """
    # Create hash from text + key metadata fields
    content = text + str(metadata.get("filename", "")) + \
        str(metadata.get("page_no", ""))
    hash_obj = hashlib.md5(content.encode('utf-8'))
    return str(uuid.UUID(hash_obj.hexdigest()))

# endregion
