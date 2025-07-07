import logging
from typing import List, Dict, Any, Optional

from Application.Domain.SearchResults import SearchResults
from Application.Domain.SearchRow import SearchRow
from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Infrastructure.Services.QdrantManagerService import QdrantManagerService

logger = logging.getLogger(__name__)

class SearchByQueries:
    """
    Service for searching in the document queries collection.
    This service focuses exclusively on searching within the documents_queries collection,
    which contains question-answer pairs.
    """
    
    def __init__(self, qdrant_service: QdrantManagerService, embedding_service: EmbeddingsService):
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        # Store original collection name
        self._original_collection = self.qdrant_service.collection_name
        # Set the specific collection for this service
        self.collection_name = "documents_queries"
    
    def _convert_to_search_results(self, results: List[Dict[str, Any]]) -> SearchResults:
        """Convert raw search results to SearchResults object"""
        search_rows = []
        for result in results:
            # Extract area from metadata if available
            area = result.get('metadata', {}).get('category', 'General')
        
            # Extract or generate links
            links = []
            if 'url' in result:
                links.append(result['url'])
            if 'source_url' in result:
                links.append(result['source_url'])
            
            # Get the score
            score = result.get('score', 0.0)

            # Chunk Id
            chunk_id = result['id']
        
            # Create SearchRow object
            search_row = SearchRow(
                text=result.get('text', ''),
                area=area,
                score=score,
                chunk_id=chunk_id,
                links=links,
                metadata=result.get('metadata')
            )
            search_rows.append(search_row)
        
        return SearchResults(
            rows=search_rows,
            count=len(search_rows)
        )
    
    async def _set_collection(self):
        """Set the collection to documents_queries"""
        self._original_collection = self.qdrant_service.collection_name
        self.qdrant_service.collection_name = self.collection_name
    
    async def _restore_collection(self):
        """Restore the original collection"""
        self.qdrant_service.collection_name = self._original_collection
    
    async def search_by_query_text(self, query: str, limit: int = 5) -> SearchResults:
        """
        Search in the queries collection using a semantic query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            SearchResults object with matching results
        """
        try:
            logger.info(f"QA collection search for: {query}")
            
            # Switch to queries collection
            await self._set_collection()
            
            # Generate embedding for the query
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Perform the search
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} results in QA collection")
            return self._convert_to_search_results(results)
            
        except Exception as e:
            logger.error(f"Error in QA collection search: {str(e)}")
            raise
            
        finally:
            # Restore original collection
            await self._restore_collection()
    
    async def search_in_qa_content(self, text: str, limit: int = 5) -> SearchResults:
        """
        Search for exact text matches within question-answer pairs.
        
        Args:
            text: Text string to search for within QA pairs
            limit: Maximum number of results to return
            
        Returns:
            SearchResults object with matching results
        """
        try:
            logger.info(f"Searching for text in QA pairs: {text}")
            
            # Switch to queries collection
            await self._set_collection()
            
            # First get more results than needed using semantic search
            query_embedding = self.embedding_service.generate_embedding(text)
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                limit=limit * 3
            )
            
            # Filter results to those that contain the text in their QA pairs
            filtered_results = []
            for result in results:
                qa_pairs = result.get('qa_pairs', [])
                
                for qa_pair in qa_pairs:
                    question = qa_pair.get('query', '')
                    answer = qa_pair.get('answer', '')
                    
                    if (text.lower() in question.lower() or text.lower() in answer.lower()):
                        filtered_results.append(result)
                        break
                
                if len(filtered_results) >= limit:
                    break
            
            logger.info(f"Found {len(filtered_results)} matches in QA pairs")
            return self._convert_to_search_results(filtered_results)
            
        except Exception as e:
            logger.error(f"Error in QA content search: {str(e)}")
            raise
            
        finally:
            # Restore original collection
            await self._restore_collection()
    
    async def search_questions_only(self, query: str, limit: int = 5) -> SearchResults:
        """
        Search only in the questions of QA pairs.
        
        Args:
            query: Query text to search for in questions
            limit: Maximum number of results to return
            
        Returns:
            SearchResults object with matching results
        """
        try:
            logger.info(f"Searching for questions containing: {query}")
            
            # Switch to queries collection
            await self._set_collection()
            
            # First get more results than needed using semantic search
            query_embedding = self.embedding_service.generate_embedding(query)
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                limit=limit * 3
            )
            
            # Filter results to those that contain the text in questions only
            filtered_results = []
            for result in results:
                qa_pairs = result.get('qa_pairs', [])
                
                for qa_pair in qa_pairs:
                    question = qa_pair.get('query', '')
                    
                    if query.lower() in question.lower():
                        filtered_results.append(result)
                        break
                
                if len(filtered_results) >= limit:
                    break
            
            logger.info(f"Found {len(filtered_results)} question matches")
            return self._convert_to_search_results(filtered_results)
            
        except Exception as e:
            logger.error(f"Error in question search: {str(e)}")
            raise
            
        finally:
            # Restore original collection
            await self._restore_collection()
    
    async def search_with_query_metadata(self, 
                                       query: str, 
                                       metadata_filter: Dict[str, Any],
                                       limit: int = 5) -> SearchResults:
        """
        Search in the queries collection with additional metadata filters.
        
        Args:
            query: Search query text
            metadata_filter: Dictionary of metadata filters to apply
            limit: Maximum number of results to return
            
        Returns:
            SearchResults object with matching results
        """
        try:
            logger.info(f"QA collection search with metadata filters: {metadata_filter}")
            
            # Switch to queries collection
            await self._set_collection()
            
            # Generate embedding for the query
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Perform the filtered search
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                metadata_filter=metadata_filter,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} filtered results in QA collection")
            return self._convert_to_search_results(results)
            
        except Exception as e:
            logger.error(f"Error in QA collection filtered search: {str(e)}")
            raise
            
        finally:
            # Restore original collection
            await self._restore_collection() 