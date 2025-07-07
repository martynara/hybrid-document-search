import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct
)
import logging
from sentence_transformers import SentenceTransformer
from filelock import FileLock
import asyncio
import threading
import time

logger = logging.getLogger(__name__)

class QdrantManagerService:
    """
    Service for managing Qdrant vector database collections.
    
    This class provides methods for creating, managing, and interacting with Qdrant 
    collections for vector search functionality. It includes both synchronous and
    asynchronous interfaces, with synchronous methods being preferred for most operations.
    """
    
    def __init__(self, collection_name: str = "documents", vector_size: int = 384, max_retries: int = 3):
        """
        Initialize the QdrantManagerService with improved lock handling.
        
        Args:
            collection_name: Name of the collection to use
            vector_size: Dimension of vectors to store
            max_retries: Maximum number of retries if database is locked
        """
        self._collection_name = collection_name
        self.vector_size = vector_size
        self.max_retries = max_retries
        
        # Set up Qdrant database path
        self.db_path = "Data/VectorDB"
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize lock for thread safety
        self.lock = threading.RLock()
        
        # Configure Qdrant client with retry logic
        self.client = self._create_client_with_retry()
        
        # Create or recreate collection if it doesn't exist
        if self.client:
            self.ensure_collection_exists()
            logger.info(f"QdrantManagerService initialized with collection '{collection_name}' and vector size {vector_size}")
        else:
            raise RuntimeError(f"Failed to initialize Qdrant client after {max_retries} attempts")
    
    def _create_client_with_retry(self) -> Optional[QdrantClient]:
        """
        Create Qdrant client with retry logic for handling database locks.
        
        Returns:
            QdrantClient instance or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting to connect to Qdrant database (attempt {attempt + 1}/{self.max_retries})")
                client = QdrantClient(path=self.db_path)
                logger.info("Successfully connected to Qdrant database")
                return client
                
            except RuntimeError as e:
                if "already accessed by another instance" in str(e):
                    logger.warning(f"Database is locked by another instance (attempt {attempt + 1}/{self.max_retries})")
                    
                    if attempt < self.max_retries - 1:
                        # Wait before retrying
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        
                        # Try to find and suggest killing existing processes
                        self._suggest_process_cleanup()
                    else:
                        logger.error("All retry attempts failed. Please ensure no other instances are running.")
                        logger.error("You can try:")
                        logger.error("1. Close all running Streamlit instances")
                        logger.error("2. Kill Python processes: ps aux | grep python")
                        logger.error("3. Restart your application")
                        raise e
                else:
                    logger.error(f"Unexpected error creating Qdrant client: {str(e)}")
                    raise e
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise e
                    
        return None
    
    def _suggest_process_cleanup(self):
        """
        Suggest process cleanup commands to the user.
        """
        logger.info("=" * 50)
        logger.info("SUGGESTION: Kill existing processes using:")
        logger.info("  Windows: taskkill /f /im python.exe")
        logger.info("  Linux/Mac: pkill -f python")
        logger.info("  Or find processes: ps aux | grep python")
        logger.info("=" * 50)

    @property
    def collection_name(self) -> str:
        """Get the current collection name"""
        return self._collection_name
        
    @collection_name.setter
    def collection_name(self, value: str):
        """
        Set the collection name and ensure it exists.
        
        Args:
            value: New collection name to use
        """
        if value != self._collection_name:
            self._collection_name = value
            self.ensure_collection_exists()
            logger.info(f"Collection name changed to '{value}'")
    
    def ensure_collection_exists(self):
        """Ensure the current collection exists, creating it if needed"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self._collection_name not in collection_names:
                logger.info(f"Creating collection '{self._collection_name}'")
                
                # Create the collection with the specified vector size
                self.client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                
                logger.info(f"Collection '{self._collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self._collection_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection existence: {str(e)}")
            raise
    
    def clear_collection(self):
        """Clear all points from the current collection"""
        try:
            with self.lock:
                # Check if collection exists
                collections = self.client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if self._collection_name in collection_names:
                    # Delete and recreate collection
                    self.client.delete_collection(collection_name=self._collection_name)
                    logger.info(f"Collection '{self._collection_name}' deleted")
                    
                # Create the collection (even if it didn't exist before)
                self.client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                
                logger.info(f"Collection '{self._collection_name}' recreated/cleared successfully")
                
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
    
    def insert_point_sync(self, text: str, embedding: List[float], metadata: dict, chunk_id: str) -> str:
        """
        Insert a point into the collection (synchronous version).
        
        Args:
            text: Text content to store
            embedding: Vector embedding of the text
            metadata: Additional metadata to store with the point
            chunk_id: Unique identifier for the chunk
            
        Returns:
            The chunk_id of the inserted point
            
        Raises:
            Exception: If insertion fails
        """
        try:
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": metadata
                }
            )

            with self.lock:
                self.client.upsert(
                    collection_name=self._collection_name,
                    points=[point]
                )
            return chunk_id

        except Exception as e:
            logger.error(f"Error inserting point {chunk_id}: {str(e)}")
            raise
    
    def document_exists_sync(self, file_name: str) -> bool:
        """
        Check if document already exists in collection (synchronous version).
        
        Args:
            file_name: Name of the file to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            with self.lock:
                results, _ = self.client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_name",
                                match=MatchValue(value=file_name)
                            )
                        ]
                    ),
                    limit=1  # Check only one document
                )
            return len(results) > 0
        except Exception as e:
            logger.error(f"❌ Error checking document existence: {str(e)}")
            return False
    
    def search_sync(self, query_vector: List[float], metadata_filter: Dict[str, Any] = None, limit: int = 10) -> List[Dict]:
        """
        Search points by vector and/or metadata filters (synchronous version).
        
        Args:
            query_vector: Vector to search for (semantic matching)
            metadata_filter: Optional metadata filters to apply 
            limit: Maximum number of results to return
            
        Returns:
            List of matching points with scores and payload
        """
        try:
            filter_query = None
            
            if metadata_filter:
                conditions = [
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    ) for key, value in metadata_filter.items()
                ]
                filter_query = Filter(must=conditions)
            
            with self.lock:
                results = self.client.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector if query_vector else None,
                    query_filter=filter_query if metadata_filter else None,
                    limit=limit,
                    with_payload=True,
                    with_vectors=True
                )
                
            # Format results
            formatted_results = [
                {
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "score": getattr(point, "score", None),
                    # Extract metadata directly from payload
                    "metadata": point.payload.get("metadata", {}),
                    # Include other root-level fields
                    "keywords": point.payload.get("keywords", []),
                    "summary": point.payload.get("summary", ""),
                    "qa_pairs": point.payload.get("qa_pairs", []),
                    "qa_count": point.payload.get("qa_count", 0),
                    "qa_updated_at": point.payload.get("qa_updated_at", None)
                }
                for point in results
            ]
            
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Error during search: {str(e)}")
            return []
    
    def close(self):
        """
        Explicitly close the Qdrant client connection.
        More reliable than waiting for __del__ to be called.
        """
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                logger.info(f"Qdrant database connection closed: {self.db_path}")
                return True
        except Exception as e:
            logger.error(f"Error closing Qdrant client connection: {str(e)}")
            return False
        return True

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.close()
        except Exception:
            pass
    
    # Async methods - maintained for backward compatibility
    
    async def insert_point(self, text: str, embedding: List[float], metadata: dict, chunk_id: str) -> str:
        """
        Insert a point into the collection asynchronously.
        For new code, consider using insert_point_sync instead.
        
        Args:
            text: Text content to store
            embedding: Vector embedding of the text
            metadata: Additional metadata to store with the point
            chunk_id: Unique identifier for the chunk
            
        Returns:
            The chunk_id of the inserted point
            
        Raises:
            Exception: If insertion fails
        """
        try:
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": metadata
                }
            )

            # Run the synchronous operation in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _insert():
                with self.lock:
                    self.client.upsert(
                        collection_name=self._collection_name,
                        points=[point]
                    )
                return chunk_id
                
            result = await loop.run_in_executor(None, _insert)
            return result

        except Exception as e:
            logger.error(f"Error inserting point {chunk_id}: {str(e)}")
            raise

    async def document_exists(self, file_name: str) -> bool:
        """
        Check if document already exists in collection asynchronously.
        For new code, consider using document_exists_sync instead.
        
        Args:
            file_name: Name of the file to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _check_exists():
                with self.lock:
                    results, _ = self.client.scroll(
                        collection_name=self._collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="file_name",
                                    match=MatchValue(value=file_name)
                                )
                            ]
                        ),
                        limit=1  # Check only one document
                    )
                return len(results) > 0
                
            return await loop.run_in_executor(None, _check_exists)
        except Exception as e:
            logger.error(f"❌ Error checking document existence: {str(e)}")
            return False

    async def search(self, query_vector: List[float], metadata_filter: Dict[str, Any] = None, limit: int = 10) -> List[Dict]:
        """
        Search points by vector and/or metadata filters asynchronously.
        For new code, consider using search_sync instead.
        
        Args:
            query_vector: Vector to search for (semantic matching)
            metadata_filter: Optional metadata filters to apply 
            limit: Maximum number of results to return
            
        Returns:
            List of matching points with scores and payload
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _search():
                filter_query = None
                
                if metadata_filter:
                    conditions = [
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        ) for key, value in metadata_filter.items()
                    ]
                    filter_query = Filter(must=conditions)
                
                with self.lock:
                    results = self.client.search(
                        collection_name=self._collection_name,
                        query_vector=query_vector if query_vector else None,
                        query_filter=filter_query if metadata_filter else None,
                        limit=limit,
                        with_payload=True,
                        with_vectors=True
                    )
                    
                # Format results
                formatted_results = [
                    {
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "score": getattr(point, "score", None),
                        # Extract metadata directly from payload
                        "metadata": point.payload.get("metadata", {}),
                        # Include other root-level fields
                        "keywords": point.payload.get("keywords", []),
                        "summary": point.payload.get("summary", ""),
                        "qa_pairs": point.payload.get("qa_pairs", []),
                        "qa_count": point.payload.get("qa_count", 0),
                        "qa_updated_at": point.payload.get("qa_updated_at", None)
                    }
                    for point in results
                ]
                
                return formatted_results
                
            return await loop.run_in_executor(None, _search)

        except Exception as e:
            logger.error(f"❌ Error during search: {str(e)}")
            return []

    def _calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = sum(x * x for x in vector1) ** 0.5
        norm2 = sum(x * x for x in vector2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0

    async def update_point_metadata(self, chunk_id: str, payload_update: Dict[str, Any]) -> bool:
        """Update metadata for a given chunk_id using the provided payload_update dictionary
        
        This method properly handles nested payload updates while preserving existing structure.
        """
        try:
            # Since client operations are synchronous but wrapped in a lock,
            # we'll run them in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            def _update_metadata():
                with self.lock:
                    # First check if point exists
                    existing_points = self.client.retrieve(collection_name=self._collection_name, ids=[chunk_id])
                    if not existing_points or len(existing_points) == 0:
                        logger.warning(f"⚠️ Chunk {chunk_id} not found in collection - this indicates a potential ID mismatch problem")
                        
                        # Let's try to find the total count of points in the collection for diagnostics
                        collection_info = self.client.get_collection(collection_name=self._collection_name)
                        points_count = collection_info.vectors_count
                        logger.info(f"Total points in collection: {points_count}")
                        
                        # Try to find a chunk with similar content (if it contains text fragment)
                        if "metadata" in payload_update and "qa_pairs" in payload_update["metadata"]:
                            # Extract a text sample from the first QA pair to help find the chunk
                            try:
                                first_qa = payload_update["metadata"]["qa_pairs"][0]
                                text_sample = first_qa.get("query", "") + " " + first_qa.get("answer", "")
                                if text_sample and len(text_sample) > 20:
                                    logger.info(f"Attempting to find matching chunk for: {text_sample[:100]}...")
                                    
                                    # Prevent this from being a blocking call by scheduling it separately
                                    asyncio.create_task(self.find_chunk_by_text(text_sample[:100], limit=1))
                            except Exception as e:
                                logger.error(f"Error during diagnostic search: {str(e)}")
                        
                        return False
                    
                    # Update the payload
                    self.client.set_payload(
                        collection_name=self._collection_name,
                        payload=payload_update,
                        points=[chunk_id]
                    )
                    
                    # Log successful update
                    logger.info(f"Updated payload for chunk {chunk_id}: {payload_update.keys()}")
                    return True
            
            # Run the update operation in a thread pool to avoid blocking
            result = await loop.run_in_executor(None, _update_metadata)
            return result

        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return False

    async def search_by_metadata(self, metadata_filter: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Search by metadata"""
        try:
            # Prepare filter for Qdrant
            filter_conditions = []
            for key, value in metadata_filter.items():
                filter_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            
            filter_query = Filter(must=filter_conditions) if filter_conditions else None
            
            # Perform search
            with self.lock:
                results = self.client.scroll(
                    collection_name=self._collection_name,
                    limit=limit,
                    scroll_filter=filter_query,
                    with_payload=True
                )[0]
            
            # Process results
            processed_results = []
            for point in results:
                processed_results.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in point.payload.items() 
                        if k not in ["text"]
                    },
                    "score": 1.0  # No score for metadata search
                })
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            raise

    async def search_by_keywords(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Search by keywords"""
        try:
            # Prepare filter for Qdrant
            filter_conditions = []
            for keyword in keywords:
                filter_conditions.append(
                    FieldCondition(
                        key="keywords",
                        match=MatchValue(value=keyword)
                    )
                )
            
            filter_query = Filter(should=filter_conditions) if filter_conditions else None
            
            # Perform search
            with self.lock:
                results = self.client.scroll(
                    collection_name=self._collection_name,
                    limit=limit,
                    scroll_filter=filter_query,
                    with_payload=True
                )[0]
            
            # Process results
            processed_results = []
            for point in results:
                processed_results.append({
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in point.payload.items() 
                        if k not in ["text"]
                    },
                    "score": 1.0  # No score for keywords search
                })
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in keywords search: {str(e)}")
            raise

    async def find_chunk_by_text(self, text_sample: str, limit: int = 5) -> List[Dict]:
        """Find chunks that contain the given text sample, useful for diagnostics and ID recovery.
        
        Args:
            text_sample: A fragment of text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching chunks with their IDs and full text
        """
        try:
            # Generate embedding for the text sample
            sample_embedding = self.embedding_model.encode(text_sample).tolist()
            
            # Execute a semantic search to find similar texts
            matching_chunks = await self.search(sample_embedding, limit=limit)
            
            if not matching_chunks:
                logger.info(f"No matching chunks found for text: {text_sample[:50]}...")
                return []
            
            # Log the found chunks
            for chunk in matching_chunks:
                chunk_id = chunk.get('id', 'unknown')
                chunk_text = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                score = chunk.get('score', 0)
                logger.info(f"Found potential matching chunk {chunk_id} with score {score}: {chunk_text}")
                
            return matching_chunks
            
        except Exception as e:
            logger.error(f"Error finding chunk by text: {str(e)}")
            return []

    def count_vectors(self) -> int:
        """
        Return the count of points in the current collection.
    
        Returns:
            int: Number of points in the collection
        
        Raises:
            Exception: If query fails
        """
        try:
            collection_info = self.client.get_collection(collection_name=self._collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting collection count: {str(e)}")
            return 0
        
