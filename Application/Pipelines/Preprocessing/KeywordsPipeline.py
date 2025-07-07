import logging
import os
import time
from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional

from Application.Domain.ChunkedDocuments import ChunkedDocuments, Chunk
from Application.InternalServices.LLMService import LLMService
from Application.Services.Metadata.KeywordsService import KeywordsService

logger = logging.getLogger(__name__)

class KeywordsPipeline:
    """
    Keywords Pipeline that processes Document objects to add keywords metadata to chunks.
    """
    
    def __init__(self, use_llm=True):
        """
        Initialize Keywords Pipeline.
        
        Args:
            use_llm: Whether to use LLM for generating keywords
        """
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm
        
        # Initialize KeywordsService if needed
        if use_llm:
            try:
                self.keywords_service = KeywordsService(LLMService.create_openai())
                self.logger.info("Keywords service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Keywords service: {str(e)}")
                self.use_llm = False
    
    async def process_document(self, document: ChunkedDocuments) -> ChunkedDocuments:
        """
        Process a ChunkedDocuments object, adding keywords metadata to each chunk.
        
        Args:
            document: ChunkedDocuments object to process
            
        Returns:
            Updated ChunkedDocuments object with keywords metadata
        """
        self.logger.info(f"=== Processing document: {document.document_id} ===")
        
        # Validate document
        if not document or not hasattr(document, 'chunks') or not document.chunks:
            self.logger.error("Invalid or empty document provided")
            return document  # Return the document as-is even if invalid
        
        # Add keywords field to each chunk's metadata and generate keywords if LLM is enabled
        if self.use_llm:
            self.logger.info("Generating keywords for document chunks...")
            total_chunks = len(document.chunks)
            
            # Process each chunk with progress tracking
            for i, chunk in enumerate(document.chunks):
                try:
                    # Print progress
                    print(f"Processing chunk {i+1}/{total_chunks}...", end='\r')
                    
                    # Generate keywords for the chunk
                    chunk = await self._extract_keywords_for_chunk(chunk)
                    
                except Exception as e:
                    self.logger.error(f"Error generating keywords for chunk {i+1}: {str(e)}")
            
            print()  # Add a newline after progress
            self.logger.info(f"Completed keywords generation for {total_chunks} chunks")
        else:
            # Just add empty keywords arrays if LLM is not enabled
            self.logger.info("Adding empty keywords metadata to each chunk...")
            for chunk in document.chunks:
                # Initialize the keywords array if it doesn't exist
                chunk.metadata.keywords = []
        
        self.logger.info(f"=== Processing completed for document: {document.document_id} ===")
        
        return document
    
    async def _extract_keywords_for_chunk(self, chunk: Chunk) -> Chunk:
        """
        Extract keywords for a chunk, using KeywordsService.
        
        Args:
            chunk: Chunk to process
            
        Returns:
            The updated Chunk with keywords in metadata
        """
        try:
            # Use the KeywordsService
            self.logger.info(f"Using KeywordsService to generate keywords for chunk: {chunk.metadata.chunk_id}")
            
            # Extract keywords
            keywords = await self.keywords_service.extract_keywords(chunk.text)
            
            # Add keywords to chunk metadata
            chunk.metadata.keywords = keywords
            
            self.logger.info(f"Generated {len(keywords)} keywords for chunk {chunk.metadata.chunk_id}")
            
            return chunk
            
        except Exception as e:
            self.logger.error(f"Error in keyword extraction for chunk {chunk.metadata.chunk_id}: {str(e)}")
            chunk.metadata.keywords = []
            return chunk