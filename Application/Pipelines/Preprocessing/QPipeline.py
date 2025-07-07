import logging
import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

from Application.Domain.ChunkedDocuments import ChunkedDocuments, Chunk
from Application.InternalServices.LLMService import LLMService
from Application.Services.Metadata.QueryService import QueryService

logger = logging.getLogger(__name__)

class QPipeline:
    """
    Q Pipeline that processes Document objects to add query metadata to chunks.
    """

    def __init__(self, use_llm=True):
        """
        Initialize Q Pipeline.
        
        Args:
            use_llm: Whether to use LLM for generating queries
        """
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm
        
        # Initialize Query service if needed
        if use_llm:
            try:
                self.q_service = QueryService()  # Service will create its own LLM service
                self.logger.info("Query service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Q service: {str(e)}")
                self.use_llm = False
    
    async def process_document(self, document: ChunkedDocuments) -> ChunkedDocuments:
        """
        Process a Document object, adding query metadata to each chunk (synchronous version).
        
        Args:
            document: Document object to process
            
        Returns:
            Updated Document object with query metadata
        """
        self.logger.info(f"=== Processing document: {document.document_id} ===")
        
        # Validate document
        if not document or not hasattr(document, 'chunks') or not document.chunks:
            self.logger.error("Invalid or empty document provided")
            return document  # Return the document as-is even if invalid
        
        # Add query field to each chunk's metadata and generate queries if LLM is enabled
        if self.use_llm:
            self.logger.info("Generating queries for document chunks...")
            total_chunks = len(document.chunks)
            
            # Process each chunk with progress tracking
            for i, chunk in enumerate(document.chunks):
                try:
                    # Print progress
                    print(f"Processing chunk {i+1}/{total_chunks}...", end='\r')
                    
                    # Generate queries for the chunk using synchronous helper method
                    chunk = await self._generate_queries_for_chunk(chunk)
                    
                except Exception as e:
                    self.logger.error(f"Error generating queries for chunk {i+1}: {str(e)}")
            
            print()  # Add a newline after progress
            self.logger.info(f"Completed query generation for {total_chunks} chunks")
        else:
            # Just add empty queries arrays if LLM is not enabled
            self.logger.info("Adding empty queries metadata to each chunk...")
            for chunk in document.chunks:
                # Initialize the queries array if it doesn't exist
                chunk.metadata.queries = []
        
        self.logger.info(f"=== Processing completed for document: {document.document_id} ===")
        
        return document
        
    async def _generate_queries_for_chunk(self, chunk: Chunk) -> Chunk:
        """
        Generate queries for a chunk synchronously, using QueryDocumentService.
        
        Args:
            chunk: DocumentChunk to process
            
        Returns:
            The updated DocumentChunk with queries in metadata
        """
        try:
            # Use the QueryDocumentService's synchronous method
            self.logger.info(f"Using QueryService to generate queries for chunk: {chunk.metadata.chunk_id}")
            
            # Call the synchronous method of the service
            chunk = await self.q_service.generate_queries_for_chunk(chunk)
            
            self.logger.info(f"Generated {len(chunk.metadata.queries)} queries for chunk {chunk.metadata.chunk_id}")
             
            return chunk
            
        except Exception as e:
            self.logger.error(f"Error in synchronous query generation for chunk {chunk.metadata.chunk_id}: {str(e)}")
            return chunk