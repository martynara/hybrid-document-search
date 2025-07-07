import logging
import time
import asyncio
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple

from Application.Domain.SearchResults import SearchResults
from Application.Services.Search.SearchByKeywords import SearchByKeywords
from Application.Services.Search.SearchByQueries import SearchByQueries
from Application.Services.Search.SearchBySummaries import SearchBySummaries
from Application.Services.Search.AddBonus import BonusService
from Infrastructure.Services.QdrantManagerService import QdrantManagerService

logger = logging.getLogger(__name__)

class AdvancedSearchService:
    """
    Advanced search service providing multi-collection search capabilities
    with bonus scoring for matches across collections.
    """
    
    def __init__(self, qdrant_service: QdrantManagerService, search_service, embedding_service):
        """
        Initialize the advanced search service.
        
        Args:
            search_service: Service for performing searches in the main documents collection
            embedding_service: Service for creating text embeddings
        """
        self.qdrant_service = qdrant_service
        self.search_service = search_service
        self.embedding_service = embedding_service
        
        # Initialize specialized search services
        self.keywords_service = SearchByKeywords(self.qdrant_service, self.embedding_service)
        self.queries_service = SearchByQueries(self.qdrant_service, self.embedding_service)
        self.summaries_service = SearchBySummaries(self.qdrant_service, self.embedding_service)
        
        # Initialize bonus service for score calculation
        self.bonus_service = BonusService()
        
        # Map collection names to their respective search services
        self.collection_service_map = {
            "documents": self.search_service,
            "documents_keywords": self.keywords_service,
            "documents_summaries": self.summaries_service,
            "documents_queries": self.queries_service
        }
    
    async def search_in_collection(self, collection_name: str, query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[SearchResults]]:
        """
        Helper method to search in a specific collection using the appropriate specialized service.
        
        Args:
            collection_name: Name of the collection to search in
            query: User's search query
            top_k: Maximum number of results to return
            metadata_filter: Optional filter for document metadata
            
        Returns:
            Tuple of (collection_name, search_results)
        """
        try:
            logger.info(f"Searching in collection: {collection_name}")
            
            # Get the appropriate service for this collection
            if collection_name not in self.collection_service_map:
                logger.warning(f"No specialized service for collection {collection_name}, using default search service")
                # Use the default search service if no specialized service exists
                service = self.search_service
                
                # Store original collection
                original_collection = service.qdrant_service.collection_name
                # Switch to the target collection
                service.qdrant_service.collection_name = collection_name
                
                # Prepare search parameters
                search_params = {"query": query, "limit": top_k}
                if metadata_filter:
                    search_params["metadata_filter"] = metadata_filter
                
                # Execute search
                results = await service.search_semantic(**search_params)
                
                # Restore original collection
                service.qdrant_service.collection_name = original_collection
            else:
                # Use the specialized search service
                service = self.collection_service_map[collection_name]
                
                if collection_name == "documents":
                    # For main documents, use the standard search method
                    search_params = {"query": query, "limit": top_k}
                    if metadata_filter:
                        search_params["metadata_filter"] = metadata_filter
                    
                    results = await service.search_semantic(**search_params)
                elif collection_name == "documents_keywords":
                    # Use keyword specific search
                    if metadata_filter:
                        results = await service.search_with_keyword_metadata(query, metadata_filter, top_k)
                    else:
                        results = await service.search_by_keyword_query(query, top_k)
                elif collection_name == "documents_summaries":
                    # Use summary specific search
                    if metadata_filter:
                        results = await service.search_with_summary_metadata(query, metadata_filter, top_k)
                    else:
                        results = await service.search_by_summary_query(query, top_k)
                elif collection_name == "documents_queries":
                    # Use query specific search
                    if metadata_filter:
                        results = await service.search_with_query_metadata(query, metadata_filter, top_k)
                    else:
                        results = await service.search_by_query_text(query, top_k)
            
            logger.info(f"Found {results.count} results in {collection_name}")
            return collection_name, results
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {str(e)}")
            if 'st' in globals():
                st.error(f"Error searching collection {collection_name}: {str(e)}")
            return collection_name, None
    
    async def multi_collection_search(self, 
                                    query: str, 
                                    collections: List[str] = ["documents"], 
                                    limit: int = 5,
                                    top_k: int = 50,
                                    top_a: int = 5,
                                    metadata_filter: Optional[Dict[str, Any]] = None) -> SearchResults:
        """
        Performs search across multiple collections in parallel with additional scoring for repeated matches.
        
        This method implements a multi-tier search strategy:
        1. Retrieves top_k results from each collection in parallel
        2. Adds all results from the main "documents" collection to the candidate pool
        3. Analyzes only the top_a results from each additional collection
        4. Adds bonus points when the same chunk appears in the top_a results of additional collections
           (different collections contribute different bonus values)
        5. Sorts all results by final score (base_score + bonus_score)
        6. Returns only the top 'limit' results to the user
        
        The hierarchical parameters (top_k > top_a > limit) allow for precise control over:
        - The breadth of the initial search (top_k)
        - The selectivity of bonus scoring (top_a)
        - The conciseness of the final results (limit)
        
        Args:
            query: User's search query text
            collections: List of collection names to search in. The first collection in the list
                         (typically "documents") is treated as the main collection.
            limit: Maximum number of final results to return after scoring and sorting.
                  This controls how many results the user ultimately sees.
            top_k: Number of initial results to retrieve from each collection.
                  This should be larger than both top_a and limit to ensure 
                  a diverse candidate pool.
            top_a: Number of top results from additional collections to consider for 
                  bonus scoring. Only matches in these top_a results receive bonus points.
            metadata_filter: Optional filter criteria for document metadata
            
        Returns:
            SearchResults object containing the final combined and rescored results,
            ordered by final_score (descending). The metadata of each result includes
            information about matched collections, base scores, and bonus scores.
            
        Example:
            ```python
            # Search in documents and keywords collections
            results = await advanced_search.multi_collection_search(
                query="quantum computing",
                collections=["documents", "documents_keywords"],
                limit=10,  # Show 10 final results to user
                top_k=50,  # Get 50 candidates from each collection
                top_a=5    # Consider only top 5 results from documents_keywords for bonus
            )
            ```
        """
        if not query:
            return SearchResults()
        
        collection_results_map = {}
        
        try:
            # Log collection search intent
            logger.info(f"Searching in parallel across collections: {collections}")
            logger.info(f"Using parameters: limit={limit}, top_k={top_k}, top_a={top_a}")
            
            # Ensure top_a is not larger than top_k
            top_a = min(top_a, top_k)
            
            # Create search tasks for all collections in parallel
            search_tasks = [
                self.search_in_collection(collection, query, top_k, metadata_filter) 
                for collection in collections
            ]
            
            # Execute all searches in parallel and wait for all results
            collection_search_results = await asyncio.gather(*search_tasks)
            
            # Log detailed results from each collection 
            logger.info("=== Search results from each collection ===")
            for collection_name, results in collection_search_results:
                result_count = results.count if results is not None else 0
                logger.info(f"Collection '{collection_name}' returned {result_count} results")
                if results is not None and result_count > 0:
                    # Store results in the map
                    collection_results_map[collection_name] = results
                    # Log the first few chunk_ids to debug
                    sample_chunk_ids = [row.metadata.get('chunk_id', 'unknown') for row in results.rows[:3]]
                    logger.info(f"Sample chunk_ids from {collection_name}: {sample_chunk_ids}")
            
            # Check if we have results from the main collection
            if "documents" not in collection_results_map or collection_results_map["documents"].count == 0:
                logger.info("No results found in main 'documents' collection")
                return SearchResults()
            
            # Use the BonusService to apply bonuses
            main_results = collection_results_map["documents"]
            
            # Apply bonuses using the bonus service
            logger.info(f"Applying bonuses with top_a={top_a}")
            combined_results = self.bonus_service.apply_bonuses(
                main_results=main_results,
                collection_results=collection_results_map,
                top_a=top_a
            )
            
            # Get top results based on final limit
            final_results = SearchResults(
                rows=combined_results.rows[:limit],
                count=min(limit, combined_results.count)
            )
        
        except Exception as e:
            logger.error(f"Error in parallel multi-collection search: {str(e)}")
            if 'st' in globals():
                st.error(f"Error in parallel multi-collection search: {str(e)}")
            return SearchResults()
        
        # Store the collection results for debugging in st.session_state if we're in streamlit
        if 'st' in globals() and hasattr(st, 'session_state'):
            if 'collection_results' not in st.session_state:
                st.session_state.collection_results = {}
            
            st.session_state.collection_results = collection_results_map
        
        logger.info(f"Returning {final_results.count} final results with bonus scoring")
        return final_results