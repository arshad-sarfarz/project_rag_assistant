from data_loader import *


def demo_data_load(file_paths: List[str], query: str = "ZUUL"):
    """
    Demonstration of the data loader functionality.

    Args:
        file_paths: List of files to process
        query: Query string for testing
    """
    try:
        # Initialize database
        client = initialize_db("db", clear_dir=True)
        if not client:
            return

        # Create collection
        coll = create_collection(client, "documents")
        if not coll:
            return

        # Load and process documents
        texts, mds = load_text_with_docling(file_paths)
        if not texts:
            logger.error("No texts were loaded")
            return

        # Embed and store
        success = embed_and_store_texts(coll, texts, metadatas=mds)
        if not success:
            logger.error("Failed to embed and store texts")
            return

        # Query the collection
        results = query_collection(client, query)
        if results:
            print("Query Results:")
            for i, doc in enumerate(results['documents'][0]):
                print(f"\nResult {i+1}:")
                print(f"Content: {doc[:200]}...")
                if results['metadatas'][0][i]:
                    print(f"Metadata: {results['metadatas'][0][i]}")

    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")


if __name__ == "__main__":

    # Get files from folder - you can customize extensions and recursive behavior
    folder_files = get_files_from_folder(
        folder_path=FOLDER_PATH,
        extensions=['.pdf', '.docx', '.pptx', '.xlsx',
                    '.html', '.md'],  # Docling supported formats
        recursive=True  # Set to False to only scan top-level folder
    )

    # Only run demo if files exist
    existing_files = [f for f in folder_files if os.path.exists(f)]

    if existing_files:
        logger.info(f"Processing {len(existing_files)} files")
        demo_data_load(existing_files, "ZUUL")
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
