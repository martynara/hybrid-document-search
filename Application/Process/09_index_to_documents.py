import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Infrastructure.Services.QdrantManagerService import QdrantManagerService
from Application.Domain.ChunkedDocuments import ChunkedDocuments
from Application.Pipelines.Indexing.IndexingDocumentPipeline import IndexingDocumentPipeline

# Configure logging to display in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def clear_qdrant():
    # Initialize QdrantManagerService
    collection_name = "documents"
    qdrant_service = QdrantManagerService(collection_name=collection_name)
    
    # Clear the collection
    qdrant_service.clear_collection()
    logger.info(f"Collection {collection_name} cleared successfully")
    
    # Close the Qdrant client connection to release resources
    qdrant_service.close()

def main():
    input_dir = "Data/Output"
    Path(input_dir).mkdir(parents=True, exist_ok=True)

    # Get all PDF files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    
    if not json_files:
        logger.warning(f"No json files found in {input_dir}")
        return {"json": None}

    # Index pipeline
    indexing_document_pipeline = IndexingDocumentPipeline()

    for i, json_file in enumerate(json_files):
        logger.info(f"Processing file {i+1}/{len(json_files)}: {json_file}")

        # Load ChunkedDocument from the input file for indexing
        json_file_path = os.path.join(input_dir, json_file)
        chunked_documents = ChunkedDocuments.load_from_file(json_file_path)
        
        for document in chunked_documents.documents:
            indexing_document_pipeline.process_document(document)
            logger.info(f"Successfully indexed document {document.document_id}")

        logger.info(f"Successfully indexed {json_file}")

if __name__ == "__main__":
    clear_qdrant()
    main()