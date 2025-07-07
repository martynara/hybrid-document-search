import asyncio
import logging
import os
import sys
from pathlib import Path

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)

from Application.Domain.ChunkedDocuments import ChunkedDocuments
from Application.Pipelines.Preprocessing.QPipeline import QPipeline

# Configure logging to display in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

async def main():
    input_dir = "Data/Output"
    Path(input_dir).mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    
    if not json_files:
        logger.warning(f"No json files found in {input_dir}")
        return {"json": None}

    # Index pipeline
    query_pipeline = QPipeline()

    for i, json_file in enumerate(json_files):
        logger.info(f"Processing file {i+1}/{len(json_files)}: {json_file}")

        # Load ChunkedDocument from the input file for indexing
        json_file_path = os.path.join(input_dir, json_file)
        chunked_documents = ChunkedDocuments.load_from_file(json_file_path)
        
        for document in chunked_documents.documents:
            document = await query_pipeline.process_document(document)
            logger.info(f"Successfully preprocesed queries document {document.document_id}")

        chunked_documents.save_to_file(json_file_path)
        logger.info(f"Successfully preprocesed queries for {json_file}")

if __name__ == "__main__":
    asyncio.run(main())
