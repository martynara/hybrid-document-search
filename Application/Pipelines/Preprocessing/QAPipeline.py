import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from Application.Domain.Document import Document
from Application.Domain.DocumentChunk import DocumentChunk
from Application.InternalServices.LLMService import LLMService
from Application.Services.Metadata.QueryAnswearDocumentService import QueryAnswearDocumentService

logger = logging.getLogger(__name__)

class QAPipeline:
    """
    QA Pipeline that processes Document objects to add QA metadata to chunks.
    """

    def __init__(self, use_llm=True):
        """
        Initialize QA Pipeline.
        
        Args:
            use_llm: Whether to use LLM for generating QA pairs
        """
        self.logger = logging.getLogger(__name__)
        self.use_llm = use_llm
        
        # Initialize QueryAnswer service if needed
        if use_llm:
            try:
                self.qa_service = QueryAnswearDocumentService()  # Service will create its own LLM service
                self.logger.info("QueryAnswer service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize QA service: {str(e)}")
                self.use_llm = False
    
    async def process_document(self, document: Document) -> Document:
        """
        Process a Document object, adding QA metadata to each chunk.
        
        Args:
            document: Document object to process
            
        Returns:
            Updated Document object with QA metadata
        """
        self.logger.info(f"=== Processing document: {document.document_id} ===")
        
        # Validate document
        if not document or not hasattr(document, 'chunks') or not document.chunks:
            self.logger.error("Invalid or empty document provided")
            return document  # Return the document as-is even if invalid
        
        # Add QA field to each chunk's metadata and generate QA pairs if LLM is enabled
        if self.use_llm:
            self.logger.info("Generating QA pairs for document chunks...")
            total_chunks = len(document.chunks)
            
            # Process each chunk with progress tracking
            for i, chunk in enumerate(document.chunks):
                try:
                    # Print progress
                    print(f"Processing chunk {i+1}/{total_chunks}...", end='\r')
                    
                    # Generate QA pairs for the chunk
                    chunk = await self.qa_service.generate_qa_for_chunk(chunk)
                    
                except Exception as e:
                    self.logger.error(f"Error generating QA pairs for chunk {i+1}: {str(e)}")
                    # Ensure QA field exists
                    if 'qa' not in chunk.metadata:
                        chunk.metadata['qa'] = []
            
            print()  # Add a newline after progress
            self.logger.info(f"Completed QA generation for {total_chunks} chunks")
        else:
            # Just add empty QA arrays if LLM is not enabled
            self.logger.info("Adding empty QA metadata to each chunk...")
            for chunk in document.chunks:
                # Initialize the QA array if it doesn't exist
                if 'qa' not in chunk.metadata:
                    chunk.metadata['qa'] = []
        
        # Update document's metadata to reflect QA processing
        document.metadata.updated_at = datetime.now()
        document.set_metadata('qa_processed', True)
        
        self.logger.info(f"=== Processing completed for document: {document.document_id} ===")
        
        return document