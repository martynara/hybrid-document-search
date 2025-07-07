from typing import List
import os
from pathlib import Path
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from Application.Domain.TextChunk import TextChunk

logger = logging.getLogger(__name__)

class EmbeddingsService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embeddings service."""
        try:
            # Ustaw cache dir w projekcie
            cache_dir = Path("Data/Models")
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
            
            # Add torch initialization configuration to avoid the __path__._path error
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            logger.info(f"Loading embeddings model {model_name}...")
            self.model = SentenceTransformer(model_name)
            # Store the vector size for reference
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model {model_name} loaded successfully with dimension {self.vector_size}")
            
        except Exception as e:
            logger.error(f"Error loading embeddings model: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text (str): The input text to generate embedding for.
            
        Returns:
            List[float]: The generated embedding as a list of floats.
        """
        if not text or not text.strip():
            return [0.0] * self.vector_size
        
        try:
            embedding = self.model.encode(text.strip())
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * self.vector_size

    def generate_embeddings(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Generate embeddings for a list of TextChunks."""
        try:
            texts = [chunk.text for chunk in chunks]
            embeddings = self.model.encode(texts)
            
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()
            
            return chunks
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            for chunk in chunks:
                chunk.embedding = [0.0] * self.vector_size
            return chunks