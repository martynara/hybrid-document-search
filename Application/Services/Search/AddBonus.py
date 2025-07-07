import logging
from typing import Dict, List, Any, Optional, Set

from Application.Domain.SearchResults import SearchResults
from Application.Domain.SearchRow import SearchRow

logger = logging.getLogger(__name__)

class BonusService:
    """
    Service for calculating and applying bonuses to search results
    based on matches across multiple collections.
    """
    
    def __init__(self):
        """Initialize the bonus service with default collection bonuses."""
        # Default collection-specific bonus values
        self.collection_bonuses = {
            "documents_keywords": 1.5,
            "documents_summaries": 0.8, 
            "documents_queries": 0.5
        }
    
    def set_collection_bonuses(self, bonuses: Dict[str, float]):
        """
        Set custom collection bonus values.
        
        Args:
            bonuses: Dictionary mapping collection names to bonus values
        """
        self.collection_bonuses = bonuses
    
    def get_collection_bonus(self, collection_name: str) -> float:
        """
        Get the bonus value for a specific collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Bonus value for the collection (default 1.0 if not specified)
        """
        return self.collection_bonuses.get(collection_name, 1.0)
    
    def apply_bonuses(self, 
                     main_results: SearchResults,
                     collection_results: Dict[str, SearchResults],
                     top_a: int = 5) -> SearchResults:
        """
        Apply bonuses to results from the main collection based on matches in other collections.
        
        Args:
            main_results: SearchResults from the main collection (usually "documents")
            collection_results: Dictionary mapping collection names to their SearchResults
            top_a: Number of top results from each collection to consider for bonus scoring
            
        Returns:
            SearchResults with updated scores including bonuses
        """
        # Create a map of all results from main collection for quick lookup
        all_results = {}
        
        # First process main collection results
        for row in main_results.rows:
            metadata = row.metadata or {}
            document_id = metadata.get('document_id', '')
            chunk_id = metadata.get('chunk_id', '')
            
            # Generate a unique key for each result
            if chunk_id:
                key = chunk_id
            else:
                # Fall back to document_id + hash if chunk_id is missing
                key = f"{document_id}_{hash(row.text)}"
            
            all_results[key] = {
                'row': row,
                'base_score': row.score,
                'bonus_score': 0,
                'collections': ["documents"],
                'document_id': document_id,
                'chunk_id': chunk_id,
                'matched_rows': {
                    "documents": row
                }
            }
        
        # Process results from other collections
        for collection_name, results in collection_results.items():
            if collection_name == "documents" or results is None:
                continue
                
            # Only consider top_a results from additional collections for bonus scoring
            top_a_rows = results.rows[:top_a] if results.rows else []
            logger.info(f"Considering top {len(top_a_rows)} results from {collection_name} for bonus scoring")
            
            # Track how many matches we found for this collection
            matches_found = 0
            
            for row in top_a_rows:
                metadata = row.metadata or {}
                document_id = metadata.get('document_id', '')
                chunk_id = metadata.get('chunk_id', '')
                
                # Generate a unique key for each result
                if chunk_id:
                    key = chunk_id
                else:
                    # Fall back to document_id + hash if chunk_id is missing
                    key = f"{document_id}_{hash(row.text)}"
                
                # Check if we've already seen this result in main collection
                if key in all_results:
                    # Get the bonus value for this collection
                    bonus_value = self.get_collection_bonus(collection_name)
                    
                    # Update existing result with bonus score
                    all_results[key]['bonus_score'] += bonus_value
                    all_results[key]['collections'].append(collection_name)
                    all_results[key]['matched_rows'][collection_name] = row
                    matches_found += 1
                    logger.info(f"Match found in top_{top_a} of {collection_name} for key={key}! New bonus: +{bonus_value}, total bonus: {all_results[key]['bonus_score']}")
            
            logger.info(f"Found {matches_found} matches in top_{top_a} results from {collection_name}")
        
        # Calculate final scores
        for key, result in all_results.items():
            result['final_score'] = result['base_score'] + result['bonus_score']
            
        # Sort results by final score
        sorted_results = sorted(all_results.values(), key=lambda x: x['final_score'], reverse=True)
        
        # Create SearchResults object with updated scores
        final_rows = []
        for result in sorted_results:
            row = result['row']
            row.score = result['final_score']
            if not row.metadata:
                row.metadata = {}
            row.metadata['matched_collections'] = result['collections']
            row.metadata['base_score'] = result['base_score']
            row.metadata['bonus_score'] = result['bonus_score']
            row.metadata['matched_rows'] = result['matched_rows']
            final_rows.append(row)
        
        return SearchResults(rows=final_rows, count=len(final_rows))
    
    def calculate_weighted_bonuses(self,
                                  main_results: SearchResults,
                                  collection_results: Dict[str, SearchResults],
                                  weights: Optional[Dict[str, float]] = None,
                                  top_a: int = 5) -> SearchResults:
        """
        Calculate weighted bonuses based on position and collection importance.
        
        Args:
            main_results: SearchResults from the main collection
            collection_results: Dictionary mapping collection names to their SearchResults
            weights: Optional dictionary of weights to apply to different collections
            top_a: Number of top results to consider from each collection
            
        Returns:
            SearchResults with weighted bonuses applied
        """
        # Use default weights if none provided
        if weights is None:
            weights = self.collection_bonuses
            
        # Create a map for all results
        all_results = {}
        
        # Process main collection results
        for i, row in enumerate(main_results.rows):
            metadata = row.metadata or {}
            document_id = metadata.get('document_id', '')
            chunk_id = metadata.get('chunk_id', '')
            
            # Generate key
            if chunk_id:
                key = chunk_id
            else:
                key = f"{document_id}_{hash(row.text)}"
                
            # Position-based weight for main collection
            position_weight = 1.0 - (i / len(main_results.rows)) * 0.5  # decreases with position
            
            all_results[key] = {
                'row': row,
                'base_score': row.score * position_weight,
                'bonus_score': 0,
                'collections': ["documents"],
                'document_id': document_id,
                'chunk_id': chunk_id,
                'matched_rows': {
                    "documents": row
                }
            }
            
        # Process additional collections with weighted bonuses
        for collection_name, results in collection_results.items():
            if collection_name == "documents" or results is None:
                continue
                
            collection_weight = weights.get(collection_name, 1.0)
            top_a_rows = results.rows[:top_a] if results.rows else []
            
            for i, row in enumerate(top_a_rows):
                metadata = row.metadata or {}
                document_id = metadata.get('document_id', '')
                chunk_id = metadata.get('chunk_id', '')
                
                if chunk_id:
                    key = chunk_id
                else:
                    key = f"{document_id}_{hash(row.text)}"
                
                # Position-based bonus decreases with position
                position_factor = 1.0 - (i / len(top_a_rows)) * 0.5
                bonus_value = collection_weight * position_factor
                
                if key in all_results:
                    all_results[key]['bonus_score'] += bonus_value
                    all_results[key]['collections'].append(collection_name)
                    all_results[key]['matched_rows'][collection_name] = row
                    
        # Calculate final scores
        for key, result in all_results.items():
            result['final_score'] = result['base_score'] + result['bonus_score']
            
        # Sort results by final score
        sorted_results = sorted(all_results.values(), key=lambda x: x['final_score'], reverse=True)
        
        # Create SearchResults object with updated scores
        final_rows = []
        for result in sorted_results:
            row = result['row']
            row.score = result['final_score']
            if not row.metadata:
                row.metadata = {}
            row.metadata['matched_collections'] = result['collections']
            row.metadata['base_score'] = result['base_score']
            row.metadata['bonus_score'] = result['bonus_score']
            row.metadata['matched_rows'] = result['matched_rows']
            final_rows.append(row)
        
        return SearchResults(rows=final_rows, count=len(final_rows)) 