import os
import logging
from config import DEFAULT_COLLECTION_NAME
from langchain_huggingface import HuggingFaceEmbeddings
from data_ingest import initialize_db
from config import *

logger = logging.getLogger()


def embed_query(documents: list[str]) -> list[list[float]]:
    """
    Embed query using a model.
    """
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    embeddings = model.embed_documents(documents)
    return embeddings


def retrieve_design_documents(
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> list[str]:
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        dict: Query results containing ids, documents, distances, and metadata
    """
    logging.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    logging.info("Embedding query...")
    # Get the first (and only) embedding
    query_embedding = embed_query([query])[0]

    logging.info("Querying collection...")

    collection = initialize_db(clear_dir=False)

    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    logging.info("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    for i in range(len(relevant_results["documents"])):
        print(f"Document {i+1}:")
        print(f"  Distance: {relevant_results['distances'][i]:.4f}")
        print(f"  ID: {relevant_results['ids'][i]}")
        print(f"  Content: {relevant_results['documents'][i]}")
        print(f"\n{'-'*80}\n")

    return relevant_results["documents"]
