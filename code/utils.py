import os
import logging
from typing import List, Dict, Tuple, Optional
from config import *

logger = logging.getLogger()


def setup_logging():

    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(
        os.path.join(OUTPUTS_DIR, "rag_assistant.log"))
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


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
