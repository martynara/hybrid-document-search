import asyncio
import time
import logging
import os
from typing import Any, Dict

from Application.Domain.ChunkedDocuments import ChunkedDocuments
from Application.Services.VectorStorage.VectorStorageService import VectorStorageService

# Configure logging to display in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class IndexingDocumentPipeline:
    """
    Pipeline that indexing document to documents index
    """

    def __init__(self, collection_name="documents"):
        self.logger = logging.getLogger(__name__)
        self.vector_service = VectorStorageService(collection_name=collection_name)
       
    def process_document(self, document: ChunkedDocuments) -> None:
        """
        Process a ChunkedDocument through the indexing pipeline.
        
        Args:
            document: ChunkedDocument to process
            
        Returns:
            None
        """
        self.logger.info(f"Processing {document.document_id} through chunking pipeline...")
        
        try:
            self.logger.info("Generating embeddings and storing in vector database...")
        
            # Generate embeddings for all chunks in the document
            document_chunks_with_embeddings = self.vector_service.embeddings_service.generate_embeddings(document.chunks)
        
            # Store each chunk in the vector database
            loop = asyncio.new_event_loop()
        
            try:
                # Initialize progress tracking
                total_chunks = len(document_chunks_with_embeddings)
                start_time = time.time()
            
                logger.info(f"Inserting chunks into vector db...")

                # # Print initial progress bar
                # self._print_progress_bar(0, total_chunks, start_time)
            
                for i, chunk in enumerate(document_chunks_with_embeddings):
                    # Create combined metadata from document metadata and chunk metadata
                    # combined_metadata = {**document.metadata.to_dict(), **chunk.metadata}
                    #combined_metadata = {**chunk.metadata}
                
                    try:
                        result = loop.run_until_complete(
                            self.vector_service.qdrant_service.insert_point(
                                text=chunk.text,
                                embedding=chunk.embedding,
                                metadata=chunk.metadata,
                                chunk_id=chunk.metadata.chunk_id
                            )
                        )

                    except Exception as e:
                        self.logger.error(f"Error storing vector: {str(e)}")
                  
                    # # Update progress bar
                    # self._print_progress_bar(i + 1, total_chunks, start_time)
            
                print()  # Add a newline after the progress bar
            finally:
                loop.close()
        
      
            self.logger.info(f"Document with {len(document.chunks)} chunks stored in vectors index")
        
            
        except Exception as e:
            self.logger.error(f"Failed to index document: {document.document_id}. Error: {str(e)}")
            raise