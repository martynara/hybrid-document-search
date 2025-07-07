import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Report containing quality evaluation metrics."""
    
    timestamp: datetime
    total_queries: int
    successful_queries: int
    success_rate: float
    average_response_time: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    
    def __post_init__(self):
        if self.total_queries > 0:
            self.success_rate = self.successful_queries / self.total_queries
        else:
            self.success_rate = 0.0


class QualityMonitor:
    """
    Service for monitoring and evaluating search quality.
    
    Manages test queries, runs evaluations, and generates quality reports.
    """
    
    def __init__(self, search_service):
        """
        Initialize the QualityMonitor.
        
        Args:
            search_service: Service for performing searches
        """
        self.search_service = search_service
        self.test_queries: List[Tuple[str, List[str]]] = []
        self.evaluation_results: List[Dict[str, Any]] = []
        
        logger.info("QualityMonitor initialized")
    
    def add_test_query(self, query: str, expected_chunk_ids: List[str]) -> None:
        """
        Add a test query with expected results.
        
        Args:
            query: The search query text
            expected_chunk_ids: List of expected chunk IDs that should be returned
        """
        self.test_queries.append((query, expected_chunk_ids))
        logger.info(f"Added test query: '{query}' with {len(expected_chunk_ids)} expected chunks")
    
    def load_test_queries_from_json(self, filename: str) -> None:
        """
        Load test queries from a JSON file.
        
        Expected JSON format:
        [
            {"query": "example query", "chunk_id": "chunk123"},
            {"query": "another query", "chunk_id": "chunk456"}
        ]
        
        Args:
            filename: Path to the JSON file
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear existing queries
            self.test_queries.clear()
            
            # Process the JSON data
            for item in data:
                query = item.get('query', '')
                chunk_id = item.get('chunk_id', '')
                
                if query and chunk_id:
                    # Check if this query already exists, if so, add chunk_id to it
                    existing_query = None
                    for i, (existing_q, existing_chunks) in enumerate(self.test_queries):
                        if existing_q == query:
                            existing_query = i
                            break
                    
                    if existing_query is not None:
                        # Add to existing query
                        self.test_queries[existing_query][1].append(chunk_id)
                    else:
                        # Add new query
                        self.test_queries.append((query, [chunk_id]))
            
            logger.info(f"Loaded {len(self.test_queries)} test queries from {filename}")
            
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {filename}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading test queries from {filename}: {str(e)}")
            raise
    
    async def run_evaluation(self, limit: int = 10) -> QualityReport:
        """
        Run evaluation on all test queries.
        
        Args:
            limit: Maximum number of results to retrieve per query
            
        Returns:
            QualityReport with evaluation metrics
        """
        if not self.test_queries:
            logger.warning("No test queries available for evaluation")
            return QualityReport(
                timestamp=datetime.now(),
                total_queries=0,
                successful_queries=0,
                success_rate=0.0,
                average_response_time=0.0,
                precision_at_k={},
                recall_at_k={}
            )
        
        successful_queries = 0
        total_response_time = 0.0
        precision_scores = {k: [] for k in [1, 3, 5, 10]}
        recall_scores = {k: [] for k in [1, 3, 5, 10]}
        
        logger.info(f"Running evaluation on {len(self.test_queries)} test queries")
        
        for query, expected_chunk_ids in self.test_queries:
            try:
                start_time = datetime.now()
                
                # Perform search
                results = await self.search_service.search_semantic(query, limit=limit)
                
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                total_response_time += response_time
                
                # Get returned chunk IDs
                returned_chunk_ids = [row.chunk_id for row in results.rows if row.chunk_id]
                
                # Check if any expected chunk was found
                found_expected = any(chunk_id in returned_chunk_ids for chunk_id in expected_chunk_ids)
                if found_expected:
                    successful_queries += 1
                
                # Calculate precision and recall at different K values
                for k in [1, 3, 5, 10]:
                    returned_at_k = returned_chunk_ids[:k]
                    
                    # Precision@K: relevant items in top K / K
                    relevant_in_k = len([chunk_id for chunk_id in returned_at_k if chunk_id in expected_chunk_ids])
                    precision_at_k = relevant_in_k / k if k > 0 else 0.0
                    precision_scores[k].append(precision_at_k)
                    
                    # Recall@K: relevant items in top K / total relevant items
                    recall_at_k = relevant_in_k / len(expected_chunk_ids) if expected_chunk_ids else 0.0
                    recall_scores[k].append(recall_at_k)
                
                # Store evaluation result
                self.evaluation_results.append({
                    'query': query,
                    'expected_chunk_ids': expected_chunk_ids,
                    'returned_chunk_ids': returned_chunk_ids,
                    'found_expected': found_expected,
                    'response_time': response_time
                })
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {str(e)}")
                # Add default scores for failed queries
                for k in [1, 3, 5, 10]:
                    precision_scores[k].append(0.0)
                    recall_scores[k].append(0.0)
        
        # Calculate average metrics
        avg_response_time = total_response_time / len(self.test_queries) if self.test_queries else 0.0
        avg_precision_at_k = {k: sum(scores) / len(scores) if scores else 0.0 
                             for k, scores in precision_scores.items()}
        avg_recall_at_k = {k: sum(scores) / len(scores) if scores else 0.0 
                          for k, scores in recall_scores.items()}
        
        report = QualityReport(
            timestamp=datetime.now(),
            total_queries=len(self.test_queries),
            successful_queries=successful_queries,
            success_rate=successful_queries / len(self.test_queries) if self.test_queries else 0.0,
            average_response_time=avg_response_time,
            precision_at_k=avg_precision_at_k,
            recall_at_k=avg_recall_at_k
        )
        
        logger.info(f"Evaluation completed. Success rate: {report.success_rate:.2%}")
        return report
    
    def get_latest_report(self) -> Optional[QualityReport]:
        """Get the most recent quality report."""
        # This would typically retrieve from a database or file
        # For now, return None
        return None
    
    def clear_test_queries(self) -> None:
        """Clear all test queries."""
        self.test_queries.clear()
        logger.info("Cleared all test queries")
