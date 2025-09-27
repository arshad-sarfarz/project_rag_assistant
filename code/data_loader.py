import os
import chromadb
import shutil
import uuid
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_docling import DoclingLoader
from config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Custom exception for data loader errors"""
    pass


def initialize_db(db_name: str = DEFAULT_DB_NAME, clear_dir: bool = False) -> Optional[chromadb.PersistentClient]:
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
        logger.info(f"Successfully initialized database at: {db_name}")
        return client

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise DataLoaderError(f"Database initialization failed: {str(e)}")


def create_collection(client: chromadb.PersistentClient,
                      collection_name: str = DEFAULT_COLLECTION_NAME) -> Optional[chromadb.api.models.Collection]:
    """
    Create or get existing ChromaDB collection.

    Args:
        client: ChromaDB client instance
        collection_name: Name of the collection

    Returns:
        Collection instance or None if creation fails
    """
    try:
        collection = client.get_or_create_collection(name=collection_name)
        logger.info(
            f"Successfully created/accessed collection: {collection_name}")
        return collection

    except Exception as e:
        logger.error(
            f"Failed to create collection {collection_name}: {str(e)}")
        raise DataLoaderError(f"Collection creation failed: {str(e)}")


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


def embed_and_store_texts(collection: chromadb.api.models.Collection,
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
        raise DataLoaderError("Mismatch between number of texts and metadatas")

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
        raise DataLoaderError(f"Embedding and storage failed: {str(e)}")


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


def query_collection(client: chromadb.PersistentClient,
                     query: str,
                     collection_name: str = DEFAULT_COLLECTION_NAME,
                     n_results: int = 3) -> Optional[Dict]:
    """
    Query the ChromaDB collection with semantic search.

    Args:
        client: ChromaDB client instance
        query: Query string
        collection_name: Name of collection to query
        n_results: Number of results to return

    Returns:
        Query results dictionary or None if query fails
    """
    try:
        hf = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        query_embedding = hf.embed_query(query)

        collection = client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        logger.info(f"Query returned {len(results['documents'][0])} results")
        return results

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return None


def get_files_from_folder(folder_path: str,
                          extensions: Optional[List[str]] = None,
                          recursive: bool = True) -> List[str]:
    """
    Get all files from a folder with specified extensions.

    Args:
        folder_path: Path to the folder to scan
        extensions: List of file extensions to include (e.g., ['.pdf', '.txt', '.docx'])
                   If None, includes common document types
        recursive: Whether to scan subfolders recursively

    Returns:
        List of full file paths
    """
    if extensions is None:
        # Document extensions that Docling actually supports
        extensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.html', '.md',
                      '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.wav', '.mp3', '.vtt']

    # Normalize extensions to lowercase
    extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                  for ext in extensions]

    file_paths = []

    try:
        if not os.path.exists(folder_path):
            logger.error(f"Folder does not exist: {folder_path}")
            return []

        if not os.path.isdir(folder_path):
            logger.error(f"Path is not a directory: {folder_path}")
            return []

        logger.info(f"Scanning folder: {folder_path}")
        logger.info(f"Looking for extensions: {extensions}")
        logger.info(f"Recursive scan: {recursive}")

        if recursive:
            # Walk through all subdirectories
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()

                    if file_ext in extensions:
                        file_paths.append(os.path.abspath(file_path))
        else:
            # Only scan the top-level directory
            try:
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)

                    # Skip directories if not recursive
                    if os.path.isfile(item_path):
                        file_ext = os.path.splitext(item)[1].lower()

                        if file_ext in extensions:
                            file_paths.append(os.path.abspath(item_path))
            except PermissionError:
                logger.error(
                    f"Permission denied accessing folder: {folder_path}")
                return []

        # Sort files for consistent ordering
        file_paths.sort()

        logger.info(f"Found {len(file_paths)} files")
        if file_paths:
            logger.info("Sample files found:")
            for i, fp in enumerate(file_paths[:5]):  # Show first 5 files
                logger.info(f"  {i+1}. {fp}")
            if len(file_paths) > 5:
                logger.info(f"  ... and {len(file_paths) - 5} more files")

        return file_paths

    except Exception as e:
        logger.error(f"Error scanning folder {folder_path}: {str(e)}")
        return []
