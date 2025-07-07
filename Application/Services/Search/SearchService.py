import logging
from asyncio.log import logger
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
import os

from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Infrastructure.Services.QdrantManagerService import QdrantManagerService

from Application.Domain.SearchResults import SearchResults
from Application.Domain.SearchRow import SearchRow

from Application.Domain.ChunkedDocuments import Document, Chunk

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self, qdrant_service: QdrantManagerService, embedding_service: EmbeddingsService):
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.json_docs_dir = "Data/Processed/Documents"  # Default directory

    def set_json_docs_directory(self, directory_path: str):
        """Set the directory path for JSON documents"""
        self.json_docs_dir = directory_path

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

    def _document_chunks_to_search_results(self, chunks: List[Chunk], score: float = 0.5) -> SearchResults:
        """Convert DocumentChunk objects to SearchResults"""
        search_rows = []
        for chunk in chunks:
            search_row = SearchRow(
                text=chunk.text,
                area=chunk.metadata.get("category", "General"),
                score=score,  # Default score for document matches
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                links=[],
                metadata=chunk.metadata
            )
            search_rows.append(search_row)
        
        return SearchResults(
            rows=search_rows,
            count=len(search_rows)
        )

    async def search_semantic(self, query: str, limit: int = 5) -> SearchResults:
        """Wyszukiwanie semantyczne na podstawie podobieństwa wektorów"""
        try:
            logger.info(f"Semantic search for: {query}")
            query_embedding = self.embedding_service.generate_embedding(query)
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                limit=limit
            )
            logger.info(f"Found {len(results)} results")
            return self._convert_to_search_results(results)
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise

    async def search_with_filter(self, query: str, metadata_filter: Dict[str, Any], limit: int = 5) -> SearchResults:
        """Wyszukiwanie semantyczne z filtrem metadanych"""
        try:
            logger.info(f"Filtered search for: {query} with filter: {metadata_filter}")
            query_embedding = self.embedding_service.generate_embedding(query)
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                metadata_filter=metadata_filter,
                limit=limit
            )
            logger.info(f"Found {len(results)} results")
            return self._convert_to_search_results(results)
        except Exception as e:
            logger.error(f"Error in filtered search: {str(e)}")
            raise

    async def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 5) -> SearchResults:
        """Wyszukiwanie tylko po metadanych"""
        try:
            logger.info(f"Metadata search with filter: {metadata_filter}")
            results = await self.qdrant_service.search_by_metadata(
                metadata_filter=metadata_filter,
                limit=limit
            )
            logger.info(f"Found {len(results)} results")
            return self._convert_to_search_results(results)
        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            raise

    async def search_by_keywords(self, keywords: List[str], limit: int = 5) -> SearchResults:
        """Wyszukiwanie po słowach kluczowych"""
        try:
            logger.info(f"Keyword search for: {keywords}")
            results = await self.qdrant_service.search_by_keywords(
                keywords=keywords,
                limit=limit
            )
            logger.info(f"Found {len(results)} results")
            return self._convert_to_search_results(results)
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            raise

    async def search_in_summaries(self, query: str, limit: int = 5) -> SearchResults:
        """Wyszukiwanie w streszczeniach"""
        try:
            logger.info(f"Summary search for: {query}")
            query_embedding = self.embedding_service.generate_embedding(query)
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                limit=limit * 3
            )
            
            filtered_results = []
            for result in results:
                summary = result.get('summary', '')
                if summary and query.lower() in summary.lower():
                    filtered_results.append(result)
                    if len(filtered_results) >= limit:
                        break
            
            logger.info(f"Found {len(filtered_results)} results")
            return self._convert_to_search_results(filtered_results)
        except Exception as e:
            logger.error(f"Error in summary search: {str(e)}")
            raise

    async def search_in_qa_pairs(self, query: str, limit: int = 5) -> SearchResults:
        """Wyszukiwanie w parach pytanie-odpowiedź (QA pairs)"""
        try:
            logger.info(f"QA pairs search for: {query}")
            query_embedding = self.embedding_service.generate_embedding(query)
            results = await self.qdrant_service.search(
                query_vector=query_embedding,
                limit=limit * 3
            )
            
            filtered_results = []
            for result in results:
                qa_pairs = result.get('qa_pairs', [])
                
                for qa_pair in qa_pairs:
                    question = qa_pair.get('query', '')
                    answer = qa_pair.get('answer', '')
                    
                    if (query.lower() in question.lower() or query.lower() in answer.lower()):
                        filtered_results.append(result)
                        break
                
                if len(filtered_results) >= limit:
                    break
            
            logger.info(f"Found {len(filtered_results)} results with matching QA pairs")
            return self._convert_to_search_results(filtered_results)
        except Exception as e:
            logger.error(f"Error in QA pairs search: {str(e)}")
            raise

    async def advanced_search(self, 
                            query: Optional[str] = None, 
                            metadata: Optional[Dict[str, Any]] = None,
                            keywords: Optional[List[str]] = None,
                            summary_contains: Optional[str] = None,
                            qa_contains: Optional[str] = None,
                            limit: int = 5) -> SearchResults:
        """Zaawansowane wyszukiwanie łączące różne kryteria"""
        try:
            logger.info(f"Advanced search with query: {query}, metadata: {metadata}, keywords: {keywords}")
            
            # Get initial results based on available criteria
            if query:
                query_embedding = self.embedding_service.generate_embedding(query)
                results = await self.qdrant_service.search(
                    query_vector=query_embedding,
                    limit=limit * 5
                )
            elif metadata:
                results = await self.qdrant_service.search_by_metadata(
                    metadata_filter=metadata,
                    limit=limit * 5
                )
            elif keywords:
                results = await self.qdrant_service.search_by_keywords(
                    keywords=keywords,
                    limit=limit * 5
                )
            else:
                return SearchResults()
            
            # Apply filters and calculate match scores
            filtered_results = []
            for result in results:
                match_score = 0
                
                # Check metadata match (worth 2 points)
                if metadata:
                    metadata_match = True
                    for key, value in metadata.items():
                        result_value = result.get('metadata', {}).get(key)
                        if result_value != value:
                            metadata_match = False
                            break
                    if metadata_match:
                        match_score += 2
                
                # Check keywords match (worth 1 point)
                if keywords:
                    result_keywords = result.get('keywords', [])
                    if any(k.lower() in [rk.lower() for rk in result_keywords] for k in keywords):
                        match_score += 1
                
                # Check summary match (worth 1 point)
                if summary_contains:
                    summary = result.get('summary', '')
                    if summary and summary_contains.lower() in summary.lower():
                        match_score += 1
                        
                # Check QA pairs match (worth 1 point)
                if qa_contains:
                    qa_pairs = result.get('qa_pairs', [])
                    for qa_pair in qa_pairs:
                        question = qa_pair.get('query', '')
                        answer = qa_pair.get('answer', '')
                        if (qa_contains.lower() in question.lower() or 
                            qa_contains.lower() in answer.lower()):
                            match_score += 1
                            break
                
                # Add result if it has any matches
                if match_score > 0:
                    result['match_score'] = match_score
                    filtered_results.append(result)
            
            # Sort by match score and limit results
            filtered_results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            filtered_results = filtered_results[:limit]
            
            logger.info(f"Found {len(filtered_results)} results")
            return self._convert_to_search_results(filtered_results)
        except Exception as e:
            logger.error(f"Error in advanced search: {str(e)}")
            raise
            
    async def search_all_collections(self, 
                                   query: str, 
                                   collections: List[str],
                                   limit: int = 5) -> SearchResults:
        """
        Przeszukuje równocześnie wiele kolekcji Qdrant.
        
        Wykonuje wyszukiwanie semantyczne tego samego zapytania we wszystkich
        wskazanych kolekcjach i łączy wyniki.
        
        Args:
            query: Tekst zapytania do wyszukania
            collections: Lista nazw kolekcji do przeszukania
            limit: Maksymalna liczba wyników
            
        Returns:
            Obiekt SearchResults z połączonymi wynikami z wszystkich kolekcji
        """
        try:
            logger.info(f"Searching all collections ({collections}) for: {query}")
            
            # Generujemy embedding raz dla wszystkich kolekcji
            query_embedding = self.embedding_service.generate_embedding(query)
            
            all_results = []
            original_collection = self.qdrant_service.collection_name
            
            # Przeszukujemy każdą kolekcję
            for collection in collections:
                try:
                    # Przełączamy kolekcję
                    self.qdrant_service.collection_name = collection
                    
                    # Wyszukujemy w bieżącej kolekcji
                    collection_results = await self.qdrant_service.search(
                        query_vector=query_embedding,
                        limit=limit
                    )
                    
                    # Dodajemy informację o źródle (kolekcji)
                    for result in collection_results:
                        result['source_collection'] = collection
                    
                    all_results.extend(collection_results)
                    logger.info(f"Found {len(collection_results)} results in collection {collection}")
                except Exception as e:
                    logger.error(f"Error searching collection {collection}: {str(e)}")
            
            # Przywracamy oryginalną kolekcję
            self.qdrant_service.collection_name = original_collection
            
            # Sortujemy wszystkie wyniki według trafności
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Ograniczamy liczbę wyników
            limited_results = all_results[:limit]
            
            logger.info(f"Total results across all collections: {len(limited_results)}")
            return self._convert_to_search_results(limited_results)
                
        except Exception as e:
            logger.error(f"Error in multi-collection search: {str(e)}")
            raise
    
    async def hybrid_search(self, 
                          text_query: str, 
                          document_query: Optional[Dict[str, Any]] = None,
                          document_path: Optional[str] = None,
                          limit: int = 5) -> SearchResults:
        """
        Perform hybrid search using both vector and document-based methods.
        
        Args:
            text_query: Search query text for semantic search
            document_query: Optional query dict for document search
            document_path: Optional path to specific JSON document (if None, searches all)
            limit: Maximum number of results to return
            
        Returns:
            Combined SearchResults object
        """
        try:
            logger.info(f"Hybrid search for: {text_query}")
            
            # First do vector search
            vector_results = await self.search_semantic(text_query, limit * 2)
            
            # Prepare document query if not provided
            if document_query is None and text_query:
                document_query = {"text": {"$contains": text_query}}
            
            # If document query exists, do document search
            document_results = SearchResults(rows=[], count=0)
            if document_query:
                if document_path:
                    document_results = self.search_chunks_in_document(document_path, document_query)
                else:
                    document_results = self.search_all_documents(document_query, limit * 2)
            
            # Combine results
            combined_rows = vector_results.rows + document_results.rows
            
            # Remove duplicates based on chunk_id
            seen_ids = set()
            unique_rows = []
            for row in combined_rows:
                if row.chunk_id not in seen_ids:
                    seen_ids.add(row.chunk_id)
                    unique_rows.append(row)
            
            # Sort by score and limit results
            unique_rows.sort(key=lambda x: x.score, reverse=True)
            final_rows = unique_rows[:limit]
            
            return SearchResults(rows=final_rows, count=len(final_rows))
                
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise