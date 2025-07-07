import logging
import asyncio
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any

from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Infrastructure.Services.QdrantManagerService import QdrantManagerService
from Application.Domain.TextChunk import TextChunk

logger = logging.getLogger(__name__)

class VectorStorageService:
    def __init__(self, collection_name: str, input_dir: str = "Data/Processed/Chunks", qdrant_service=None):
        """
        Initialize the vector storage service.
        
        Args:
            collection_name: Name of the vector collection
            input_dir: Directory containing processed chunk files
            qdrant_service: Optional existing QdrantManagerService
        """
        self.collection_name = collection_name
        self.input_dir = Path(input_dir)
        
        # Create embeddings service
        from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
        self.embeddings_service = EmbeddingsService()
        
        # Track whether we created the qdrant_service or it was passed in
        self._owns_qdrant_service = qdrant_service is None
        
        # Create or use provided qdrant_service
        if qdrant_service is None:
            from Infrastructure.Services.QdrantManagerService import QdrantManagerService
            self.qdrant_service = QdrantManagerService(collection_name=collection_name)
            logger.info(f"Created new QdrantManagerService for collection: {collection_name}")
        else:
            self.qdrant_service = qdrant_service
            # If the collection name is different from the QdrantManagerService's current one,
            # update it for consistency
            if self.qdrant_service.collection_name != collection_name:
                self.qdrant_service.collection_name = collection_name
                logger.info(f"Updated QdrantManagerService to use collection: {collection_name}")
            else:
                logger.info(f"Using existing QdrantManagerService for collection: {collection_name}")

        # Initialize path for chunk storage
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
    def process_chunks_file_sync(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single chunks file synchronously.
        
        Args:
            file_path: Path to the chunks file
            
        Returns:
            Dictionary with processing results
        """
        try:
            if not file_path.exists():
                return {"file": file_path.name, "status": "error", "error": "File not found"}
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Make sure the data has the expected structure
            if 'chunks' not in data:
                return {"file": file_path.name, "status": "error", "error": "Invalid chunks format"}
                
            chunks_data = data['chunks']
            stored_count = 0
            file_updated = False
            
            # Prepare chunks for embedding
            chunks = []
            for chunk_data in chunks_data:
                metadata = chunk_data.get('metadata', {})
                chunk = TextChunk(
                    text=chunk_data.get('text', ''),
                    metadata=metadata
                )
                chunks.append(chunk)
                
            if not chunks:
                return {"file": file_path.name, "status": "skipped", "reason": "No chunks found"}
                
            # Generate embeddings
            chunks_with_embeddings = self.embeddings_service.generate_embeddings(chunks)
            
            # Store in Qdrant
            for idx, chunk in enumerate(chunks_with_embeddings):
                # Use the original chunk_id from the data if available, or generate a fallback if not
                if idx < len(chunks_data) and 'chunk_id' in chunks_data[idx]:
                    chunk_id = chunks_data[idx]['chunk_id']
                    logger.info(f"Using existing chunk_id from data: {chunk_id}")
                else:
                    chunk_id = str(uuid.uuid4())
                    logger.warning(f"No chunk_id found in data for chunk {idx}, generated new UUID: {chunk_id}")
                    
                    # Update the original chunks data with the new ID
                    if idx < len(chunks_data):
                        chunks_data[idx]['chunk_id'] = chunk_id
                        file_updated = True
                
                # Use the synchronous insert_point_sync method
                self.qdrant_service.insert_point_sync(
                    text=chunk.text,
                    embedding=chunk.embedding,
                    metadata=chunk.metadata,
                    chunk_id=chunk_id
                )
                stored_count += 1
            
            # Update the chunks file with the new IDs if needed
            if file_updated:
                logger.info(f"Updating chunks file {file_path} with generated chunk IDs")
                data['chunks'] = chunks_data
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Processed {stored_count} chunks from {file_path.name}")
            return {"file": file_path.name, "status": "success", "chunk_count": stored_count}
            
        except Exception as e:
            logger.error(f"Error processing chunks file {file_path}: {str(e)}")
            return {"file": file_path.name, "status": "error", "error": str(e)}
    
    def process_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Process all chunk files in the input directory synchronously.
        
        Returns:
            List of results for each processed file
        """
        results = []
        
        try:
            chunk_files = list(self.input_dir.glob("*.json"))
            logger.info(f"Found {len(chunk_files)} chunk files to process")
            
            for file_path in chunk_files:
                result = self.process_chunks_file_sync(file_path)
                results.append(result)
                
            successful = sum(1 for r in results if r.get('status') == 'success')
            skipped = sum(1 for r in results if r.get('status') == 'skipped')
            failed = sum(1 for r in results if r.get('status') == 'error')
            
            logger.info(f"Finished processing all chunks: {successful} successful, {skipped} skipped, {failed} failed")
                
        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}")
            
        return results

    def update_point_metadata_sync(self, chunk_id: str, payload_update: Dict[str, Any]) -> bool:
        """
        Synchronous method to update point metadata in Qdrant.
        
        Args:
            chunk_id: The identifier of the chunk.
            payload_update: Dictionary with the metadata updates.
        
        Returns:
            True if metadata was updated successfully, otherwise False.
        """
        try:
            # Check if point exists
            existing_points = self.qdrant_service.client.retrieve(
                collection_name=self.qdrant_service.collection_name,
                ids=[chunk_id]
            )
            
            if not existing_points or len(existing_points) == 0:
                logger.warning(f"Chunk {chunk_id} not found in collection")
                return False
            
            # Update the payload
            self.qdrant_service.client.set_payload(
                collection_name=self.qdrant_service.collection_name,
                payload=payload_update,
                points=[chunk_id]
            )
            
            logger.info(f"Updated payload for chunk {chunk_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return False

    def close_sync(self):
        """
        Synchronously close all connections used by the service.
        """
        logger.info("Closing VectorStorageService connections synchronously...")
        
        # Close Qdrant service only if we created it (not if it was passed in)
        if hasattr(self, 'qdrant_service') and self.qdrant_service and hasattr(self, '_owns_qdrant_service') and self._owns_qdrant_service:
            try:
                self.qdrant_service.close()
                logger.info("Closed Qdrant client connection (owned by VectorStorageService)")
            except Exception as e:
                logger.error(f"Error closing Qdrant client: {str(e)}")
        elif hasattr(self, 'qdrant_service') and self.qdrant_service:
            logger.info("Skipping Qdrant client connection close (managed externally)")
        
        # Close embedding service if it has a close method
        if hasattr(self, 'embeddings_service') and self.embeddings_service:
            try:
                if hasattr(self.embeddings_service, 'close_sync'):
                    self.embeddings_service.close_sync()
                    logger.info("Closed Embeddings Service synchronously")
            except Exception as e:
                logger.error(f"Error closing Embeddings Service synchronously: {str(e)}")
        
        logger.info("VectorStorageService connections closed synchronously")

    # Legacy async methods - these are kept for backward compatibility but should be avoided in new code

    async def close(self):
        """
        Close all connections used by the service asynchronously.
        Legacy method maintained for backward compatibility.
        For new code, use close_sync() instead.
        """
        # Just call the synchronous version
        self.close_sync()

    async def process_chunks_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single chunks file, generate embeddings and store in Qdrant.
        
        Args:
            file_path: Path to the JSON file containing chunks
            
        Returns:
            Dict with processing results
        """
        try:
            # Load chunks from file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            chunks_data = data.get('chunks', [])
            file_metadata = data.get('metadata', {})
            
            # Skip if already processed
            file_name = file_metadata.get('file_name', '')
            if file_name and self.qdrant_service.document_exists_sync(file_name):
                logger.info(f"Document {file_name} already exists in Qdrant, skipping")
                return {"file": file_path.name, "status": "skipped", "chunk_count": 0}
            
            # Track if we need to update the chunks file with new IDs
            file_updated = False
            
            # Convert to TextChunk objects
            chunks = []
            for chunk_data in chunks_data:
                chunk = TextChunk(
                    text=chunk_data.get('text', ''),
                    metadata={
                        **file_metadata,
                        "chunk_index": chunk_data.get('chunk_index', 0),
                        "document_id": chunk_data.get('document_id', 0),
                        "page_number": chunk_data.get('page_number', 0)
                    }
                )
                chunks.append(chunk)
                
            # Generate embeddings
            chunks_with_embeddings = self.embeddings_service.generate_embeddings(chunks)
            
            # Store in Qdrant
            stored_count = 0
            for idx, chunk in enumerate(chunks_with_embeddings):
                # Use the original chunk_id from the data if available, or generate a fallback if not
                if idx < len(chunks_data) and 'chunk_id' in chunks_data[idx]:
                    chunk_id = chunks_data[idx]['chunk_id']
                    logger.info(f"Using existing chunk_id from data: {chunk_id}")
                else:
                    chunk_id = str(uuid.uuid4())
                    logger.warning(f"No chunk_id found in data for chunk {idx}, generated new UUID: {chunk_id}")
                    
                    # Update the original chunks data with the new ID
                    if idx < len(chunks_data):
                        chunks_data[idx]['chunk_id'] = chunk_id
                        file_updated = True
                
                # Properly call the async insert_point method
                await self.qdrant_service.insert_point(
                    text=chunk.text,
                    embedding=chunk.embedding,
                    metadata=chunk.metadata,
                    chunk_id=chunk_id
                )
                stored_count += 1
            
            # Update the chunks file with the new IDs if needed
            if file_updated:
                logger.info(f"Updating chunks file {file_path} with generated chunk IDs")
                data['chunks'] = chunks_data
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Processed {stored_count} chunks from {file_path.name}")
            return {"file": file_path.name, "status": "success", "chunk_count": stored_count}
            
        except Exception as e:
            logger.error(f"Error processing chunks file {file_path}: {str(e)}")
            return {"file": file_path.name, "status": "error", "error": str(e)}
    
    async def process_all_chunks_async(self) -> List[Dict[str, Any]]:
        """
        Process all chunk files in the input directory - async version.
        
        Returns:
            List of results for each processed file
        """
        try:
            results = []
            
            # Find all JSON files in input directory
            chunk_files = list(self.input_dir.glob('*.json'))
            if not chunk_files:
                logger.warning(f"No JSON files found in {self.input_dir}")
                return []
            
            # Process each file
            for file_path in chunk_files:
                result = await self.process_chunks_file(file_path)
                results.append(result)
                
            return results
        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}")
            return []

    async def update_point_metadata(self, chunk_id: str, payload_update: Dict[str, Any]) -> bool:
        """
        Asynchronous method to update point metadata in Qdrant.
        
        Args:
            chunk_id: The identifier of the chunk.
            payload_update: Dictionary with the metadata updates.
        
        Returns:
            True if metadata was updated successfully, otherwise False.
        """
        return await self.qdrant_service.update_point_metadata(chunk_id, payload_update)

    async def store_vector(self, chunk: TextChunk, document_id: str, document_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate embedding for a text chunk and store it in Qdrant.
        
        Args:
            chunk: TextChunk object to store
            document_id: ID of the document this chunk belongs to
            document_metadata: Optional metadata from the document to include
            
        Returns:
            Dict with operation results
        """
        try:
            # Make sure we're using the right collection
            if self.qdrant_service.collection_name != self.collection_name:
                self.qdrant_service.collection_name = self.collection_name
            
            # Combine chunk metadata with document metadata
            combined_metadata = {}
            if document_metadata:
                combined_metadata.update(document_metadata)
            if hasattr(chunk, 'metadata') and chunk.metadata:
                combined_metadata.update(chunk.metadata)
            
            # Add document_id to metadata
            combined_metadata['document_id'] = document_id
                
            # Store in Qdrant
            result = await self.qdrant_service.insert_point(
                text=chunk.text,
                embedding=chunk.embedding,
                metadata=combined_metadata,
                chunk_id=chunk.chunk_id
            )
            
            return {
                'success': True,
                'chunk_id': chunk.chunk_id,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error storing vector: {str(e)}")
            return {
                'success': False,
                'chunk_id': chunk.chunk_id,
                'error': str(e)
            }
