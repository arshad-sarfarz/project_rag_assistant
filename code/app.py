from data_ingest import *
from data_rag import *
from utils import *


def query_collection(client: ClientAPI,
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
        if results:
            print("Query Results:")
            for i, doc in enumerate(results['documents'][0]):
                print(f"\nResult {i+1}:")
                print(f"Content: {doc}")
                if results['metadatas'][0][i]:
                    print(f"Metadata: {results['metadatas'][0][i]}")

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return None


if __name__ == "__main__":

    # Get files from folder - you can customize extensions and recursive behavior
    folder_files = get_files_from_folder(
        folder_path=DATA_DIR,
        extensions=['.pdf', '.docx', '.pptx', '.xlsx',
                    '.html', '.md'],  # Docling supported formats
        recursive=True  # Set to False to only scan top-level folder
    )

    # Only run demo if files exist
    existing_files = [f for f in folder_files if os.path.exists(f)]

    if existing_files:
        logger.info(f"Processing {len(existing_files)} files")
        success = load_design_documents(existing_files)
        if success:
            # query_collection(client, "Does the system use Azure Front Door?")
            results = retrieve_design_documents(
                query="What are variational encoders?", threshold=1.5)

    else:
        logger.warning("No files found.")
        logger.info("Please check:")
        logger.info(
            "1. Update the folder_path variable to point to your data folder")
        logger.info("2. Or update sample_files list with specific file paths")
        logger.info(
            "3. Ensure files have supported extensions: .pdf, .docx, .pptx, .xlsx, .html, .md, images")
        logger.info(
            "\nAvailable functions: initialize_db, create_collection, load_text_with_docling, embed_and_store_texts, query_collection")
