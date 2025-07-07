import logging
import json
from typing import List, Dict, Any, Optional
from Application.InternalServices.LLMService import LLMService
from Application.Domain.DocumentChunk import DocumentChunk

logger = logging.getLogger(__name__)

QA_PROMPT = """
Przeanalizuj poniższy tekst i utwórz 3-5 kluczowych pytań wraz z odpowiedziami dotyczących jego głównych tematów i zawartości:
{text}
Zwróć wynik wyłącznie w poniższym formacie JSON, bez dodatkowego tekstu:
[
    {
        "query": "pytanie_1",
        "answer": "odpowiedź_1"
    },
    {
        "query": "pytanie_2",
        "answer": "odpowiedź_2"
    }
]
"""

class QueryAnswearDocumentService:
    """
    Service to generate QA pairs for text using an LLM.
    """
    def __init__(self, llm_service: LLMService = None):
        """
        Initialize QueryAnswerService.
        
        Args:
            llm_service: LLMService instance for text completion (optional)
                        If not provided, a new one will be created
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM service if not provided
        if llm_service is None:
            try:
                self.llm_service = LLMService.create_openai()
                self.logger.info("Created new LLM service")
            except Exception as e:
                self.logger.error(f"Failed to create LLM service: {str(e)}")
                raise
        else:
            self.llm_service = llm_service
    
    async def generate_qa_for_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """
        Generate QA pairs for a document chunk and add them to its metadata.
        
        Args:
            chunk: DocumentChunk to process
            
        Returns:
            The updated DocumentChunk with QA pairs in metadata
        """
        try:
            # Format the prompt with the chunk text
            prompt = QA_PROMPT.replace('{text}', chunk.text)
            
            # Get completion from LLM
            self.logger.info(f"Generating QA pairs for chunk: {chunk.chunk_id}")
            response = self.llm_service.complete(prompt)
            
            # Parse the JSON response
            qa_pairs = self._parse_query_answears(response)
            
            # Add QA pairs to chunk metadata
            if qa_pairs:
                if 'qa' not in chunk.metadata:
                    chunk.metadata['qa'] = []
                chunk.metadata['qa'] = qa_pairs
                self.logger.info(f"Added {len(qa_pairs)} QA pairs to chunk {chunk.chunk_id}")
            else:
                self.logger.warning(f"No valid QA pairs generated for chunk {chunk.chunk_id}")
            
            return chunk
        
        except Exception as e:
            self.logger.error(f"Error generating QA pairs for chunk {chunk.chunk_id[:8]}: {str(e)}")
            # Ensure the QA field exists even if generation failed
            if 'qa' not in chunk.metadata:
                chunk.metadata['qa'] = []
            return chunk
    
    async def generate_qa_for_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate QA pairs for multiple document chunks.
        
        Args:
            chunks: List of DocumentChunk objects to process
            
        Returns:
            List of updated DocumentChunk objects with QA pairs in metadata
        """
        updated_chunks = []
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            updated_chunk = self.generate_qa_for_chunk(chunk)
            updated_chunks.append(updated_chunk)
        
        return updated_chunks
    

    def _parse_query_answears(self, response: str) -> List[str]:
        """Parse LLM response into a list of q&a."""
        try:
            # Clean up the response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.startswith("```"):
                response = response.split("```")[1]
            if response.endswith("```"):
                response = response[:-3]

            response = response.strip()

            # Parse JSON
            qas = json.loads(response)
            if not isinstance(qas, list):
                logger.warning(f"Expected list but got {type(qas).__name__}")
                return []

            return qas

        except Exception as e:
            logger.error(f"Error parsing qas response: {str(e)}")
            logger.error(f"Raw response: {response}")
            return []