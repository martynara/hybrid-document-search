import logging
from typing import List, Dict, Any, Optional

from Application.Domain.SearchResults import SearchResults
from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Infrastructure.Services.QdrantManagerService import QdrantManagerService

logger = logging.getLogger(__name__)

class SearchByKeywords:
    """
    Service for searching in the document keywords collection.
    This service focuses exclusively on searching within the documents_keywords collection.
    """
    
    def __init__(self, qdrant_service: QdrantManagerService, embedding_service: EmbeddingsService):
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        # Store original collection name
        self._original_collection = self.qdrant_service.collection_name
        # Set the specific collection for this service
        self.collection_name = "documents_keywords"
    
    def _convert_to_search_results(self, results: List[Dict[str, Any]]) -> SearchResults:
        """Convert raw search results to SearchResults object"""
        from Application.Domain.SearchRow import SearchRow
        
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
        """Set the collection to documents_keywords"""
        self._original_collection = self.qdrant_service.collection_name
        self.qdrant_service.collection_name = self.collection_name
    
    async def _restore_collection(self):
        """Restore the original collection"""
        self.qdrant_service.collection_name = self._original_collection
    
    async def search_by_keyword_query(self, query: str, limit: int = 5) -> SearchResults:
        """
        Search in the keywords collection using a semantic query.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            
        Returns:
            SearchResults object with matching results
        """
        try:
            logger.info(f"Keyword collection search for: {query}")
            
            # Switch to keywords collection
            await self._set_collection()
            
            # Generate embedding for the query
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Perform the search
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} results in keywords collection")
            return self._convert_to_search_results(results)
            
        except Exception as e:
            logger.error(f"Error in keyword collection search: {str(e)}")
            raise
            
        finally:
            # Restore original collection
            await self._restore_collection()
    
    async def search_by_exact_keywords(self, keywords: List[str], limit: int = 5) -> SearchResults:
        """
        Search for exact keyword matches in the keywords collection.
        
        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results to return
            
        Returns:
            SearchResults object with matching results
        """
        try:
            logger.info(f"Searching for exact keywords: {keywords}")
            
            # Switch to keywords collection
            await self._set_collection()
            
            # Create a filter to match documents containing any of the keywords
            results = await self.qdrant_service.search_by_keywords(
                keywords=keywords,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} exact keyword matches")
            return self._convert_to_search_results(results)
            
        except Exception as e:
            logger.error(f"Error in exact keyword search: {str(e)}")
            raise
            
        finally:
            # Restore original collection
            await self._restore_collection()
    
    async def search_with_keyword_metadata(self, 
                                         query: str, 
                                         metadata_filter: Dict[str, Any],
                                         limit: int = 5) -> SearchResults:
        """
        Search in the keywords collection with additional metadata filters.
        
        Args:
            query: Search query text
            metadata_filter: Dictionary of metadata filters to apply
            limit: Maximum number of results to return
            
        Returns:
            SearchResults object with matching results
        """
        try:
            logger.info(f"Keyword collection search with metadata filters: {metadata_filter}")
            
            # Switch to keywords collection
            await self._set_collection()
            
            # Generate embedding for the query
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Perform the filtered search
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                metadata_filter=metadata_filter,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} filtered results in keywords collection")
            return self._convert_to_search_results(results)
            
        except Exception as e:
            logger.error(f"Error in keyword collection filtered search: {str(e)}")
            raise
            
        finally:
            # Restore original collection
            await self._restore_collection() 