import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from Application.Domain.ChunkedDocuments import ChunkedDocuments, Chunk
from Application.InternalServices.LLMService import LLMService
from Application.Services.Metadata.SummaryService import SummaryService

logger = logging.getLogger(__name__)

class SummaryPipeline:
    """
    Summary Pipeline that processes Document objects to add summary metadata to chunks.
    """
    
    def __init__(self, use_llm=True):
        """
        Initialize Summary Pipeline.
        
        Args:
            use_llm: Whether to use LLM for generating summaries
        """
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm
        
        # Initialize SummaryService if needed
        if use_llm:
            try:
                self.summary_service = SummaryService(LLMService.create_openai())
                self.logger.info("Summary service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Summary service: {str(e)}")
                self.use_llm = False
    
    async def process_document(self, document: ChunkedDocuments) -> ChunkedDocuments:
        """
        Process a ChunkedDocuments object, adding summary metadata to each chunk.
        
        Args:
            document: ChunkedDocuments object to process
            
        Returns:
            Updated ChunkedDocuments object with summary metadata
        """
        self.logger.info(f"=== Processing document: {document.document_id} ===")
        
        # Validate document
        if not document or not hasattr(document, 'chunks') or not document.chunks:
            self.logger.error("Invalid or empty document provided")
            return document  # Return the document as-is even if invalid
        
        # Add summary field to each chunk's metadata and generate summary if LLM is enabled
        if self.use_llm:
            self.logger.info("Generating summaries for document chunks...")
            total_chunks = len(document.chunks)
            
            # Process each chunk with progress tracking
            for i, chunk in enumerate(document.chunks):
                try:
                    # Print progress
                    print(f"Processing chunk {i+1}/{total_chunks}...", end='\r')
                    
                    # Generate summary for the chunk
                    chunk = await self._generate_summary_for_chunk(chunk)
                    
                except Exception as e:
                    self.logger.error(f"Error generating summary for chunk {i+1}: {str(e)}")
            
            print()  # Add a newline after progress
            self.logger.info(f"Completed summary generation for {total_chunks} chunks")
        else:
            # Just add empty summary strings if LLM is not enabled
            self.logger.info("Adding empty summary metadata to each chunk...")
            for chunk in document.chunks:
                # Initialize the summary field if it doesn't exist
                chunk.metadata.summary = ""
        
        self.logger.info(f"=== Processing completed for document: {document.document_id} ===")
        
        return document
    
    async def _generate_summary_for_chunk(self, chunk: Chunk) -> Chunk:
        """
        Generate summary for a chunk, using SummaryService.
        
        Args:
            chunk: Chunk to process
            
        Returns:
            The updated Chunk with summary in metadata
        """
        try:
            # Use the SummaryService
            self.logger.info(f"Using SummaryService to generate summary for chunk: {chunk.metadata.chunk_id}")
            
            # Since SummaryService might not have an async method, we can adapt the sync method
            summary = self.summary_service.generate_summary_sync(chunk.text)
            
            # Add summary to chunk metadata
            chunk.metadata.summary = summary
            
            self.logger.info(f"Generated summary for chunk {chunk.metadata.chunk_id}")
            
            return chunk
            
        except Exception as e:
            self.logger.error(f"Error in summary generation for chunk {chunk.metadata.chunk_id}: {str(e)}")
            chunk.metadata.summary = ""
            return chunk