import asyncio
import time
import logging
import os
from typing import Any, Dict, List

from Application.Domain.ChunkedDocuments import ChunkedDocuments
from Application.Services.VectorStorage.VectorStorageService import VectorStorageService

# Configure logging to display in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class IndexingKeywordsPipeline:
    """
    Pipeline that indexes keywords to documents_keywords index
    """

    def __init__(self, collection_name="documents_keywords"):
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
        self.logger.info(f"Processing {document.document_id} through keywords indexing pipeline...")
        
        try:
            self.logger.info("Generating keyword embeddings and storing in vector database...")
        
            # Get chunks with keywords
            document_chunks_with_keywords = [chunk for chunk in document.chunks if chunk.metadata.keywords]
            
            if not document_chunks_with_keywords:
                self.logger.warning(f"No chunks with keywords found in document {document.document_id}")
                return
                
            # Generate embeddings for keywords from each chunk
            document_chunks_with_embeddings = self.vector_service.embeddings_service.generate_embeddings(document_chunks_with_keywords)
        
            # Store each chunk's keywords in the vector database
            loop = asyncio.new_event_loop()
        
            try:
                # Initialize progress tracking
                total_chunks = len(document_chunks_with_embeddings)
                start_time = time.time()
            
                logger.info(f"Inserting keywords into vector db...")
            
                for i, chunk in enumerate(document_chunks_with_embeddings):
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
            
                print()  # Add a newline after progress
            finally:
                loop.close()
        
            self.logger.info(f"Document with {len(document_chunks_with_embeddings)} chunks stored in keywords index")
        
        except Exception as e:
            self.logger.error(f"Failed to index document keywords: {document.document_id}. Error: {str(e)}")
            raise
