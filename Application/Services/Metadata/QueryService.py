import logging
import json
from typing import List, Dict, Any, Optional
from Application.Domain.ChunkedDocuments import Chunk
from Application.InternalServices.LLMService import LLMService
import openai

logger = logging.getLogger(__name__)

QUERY_PROMPT = """
Przeanalizuj poniższy tekst wyszukując kluczowych informacji
Utwórz od 2 do 5 pytań dotyczących jego głównych tematów i zawartości uwzgledniajac kluczowe słowa.
Takst:
{text}


Zwróć wynik wyłącznie w poniższym formacie JSON, bez dodatkowego tekstu:
[
    "pytanie_1",
    "pytanie_2",
    "pytanie_3"
]
"""

class QueryService:
    """
    Service to generate queries for text using an LLM.
    """
    def __init__(self, llm_service: LLMService = None):
        """
        Initialize QueryDocumentService.
        
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
    
    async def generate_queries_for_chunk(self, chunk: Chunk) -> Chunk:
        """
        Generate queries for a document chunk and add them to its metadata.
        
        Args:
            chunk: DocumentChunk to process
            
        Returns:
            The updated DocumentChunk with queries in metadata
        """
        try:
            # Format the prompt with the chunk text
            prompt = QUERY_PROMPT.replace('{text}', chunk.text)
            
            # Get completion from LLM
            self.logger.info(f"Generating queries for chunk: {chunk.metadata.chunk_id}")
            response = await self.llm_service.complete(prompt)
            
            # Parse the JSON response
            queries = self._parse_queries(response)
            
            # Add queries to chunk metadata
            if queries:
                chunk.metadata.queries = queries
                self.logger.info(f"Added {len(queries)} queries to chunk {chunk.metadata.chunk_id}")
            else:
                self.logger.warning(f"No valid queries generated for chunk {chunk.metadata.chunk_id}")

            return chunk
            
        except Exception as e:
            self.logger.error(f"Error generating queries for chunk {chunk.metadata.chunk_id}: {str(e)}")
            return chunk
        
    def _parse_queries(self, response: str) -> List[str]:
        """Parse LLM response into a list of queries."""
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
            queries = json.loads(response)
            if not isinstance(queries, list):
                logger.warning(f"Expected list but got {type(queries).__name__}")
                return []

            return queries

        except Exception as e:
            logger.error(f"Error parsing queries response: {str(e)}")
            logger.error(f"Raw response: {response}")
            return []