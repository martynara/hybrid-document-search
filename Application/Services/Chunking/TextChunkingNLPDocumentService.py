import logging
import re
import uuid
import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import spacy

from Application.Domain.RawData import RawData
from Application.Domain.ChunkedDocuments import ChunkMetadata, Chunk, Document, ChunkedDocuments

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextChunkingNLPDocumentService:
    """
    Service for chunking text from RawData files and creating ChunkedDocument output.
    Reads from Data/Input/Raw and writes to Data/Output.
    """
    
    def __init__(self, max_chunk_size: int = 600):
        """
        Initialize the text chunking service.
        
        Args:
            max_chunk_size: Maximum number of words in one chunk (default 600)
        """
        self.max_chunk_size = max_chunk_size
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("pl_core_news_lg")
            self.nlp.max_length = 5_000_000
        except OSError:
            logger.error("Failed to load spaCy model. Make sure it's installed.")
            raise ImportError("Missing spaCy model.")
    
    def process_raw_data(self, raw_data: RawData) -> ChunkedDocuments:
        """
        Process a RawData chunk its text.
        
        Args:
            raw_data: RawData object
            
        Returns:
            ChunkedDocument: object
        """
        logger.info(f"Processing raw data {raw_data.document_id}")
        
        # Create chunks from the text
        if not raw_data.text:
            logger.warning(f"Empty text in RawData")
            chunk_texts = []
        else:
            chunk_texts = self.create_chunks(raw_data.text)
        
        # Create Chunk objects
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):

            chunk_metadata = ChunkMetadata(
                document_id=raw_data.document_id,
                chunk_id = str(uuid.uuid4()),
                keywords = [],  # Empty keywords initially
                summary  = "",  # Empty summary initially
                queries  = []   # Empty queries initially
                )

            chunk = Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)
        
        # Create Document with document_id
        document = Document(
            document_id=raw_data.document_id,
            path=raw_data.path,
            file_name=raw_data.file_name,
            document_type=raw_data.document_type,
            processed_at=datetime.now().isoformat(),
            chunks=chunks
        )
        
        # Create ChunkedDocument
        chunked_documents = ChunkedDocuments()
        chunked_documents.add_document(document)
        
        return chunked_documents

    def process_raw_data_file(self, input_file_path: str, output_dir: str = "Data/Output") -> str:
        """
        Process a RawData JSON file, chunk its text, and save as ChunkedDocument.
        
        Args:
            input_file_path: Path to the RawData JSON file
            output_dir: Directory to save the output ChunkedDocument JSON file
            
        Returns:
            Path to the saved output file
        """
        logger.info(f"Processing raw data file: {input_file_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the RawData from the input file for processing
        try:
            raw_data = RawData.load_from_file(input_file_path)
        except Exception as e:
            logger.error(f"Failed to load RawData from {input_file_path}: {str(e)}")
            raise
        
        # Get original file name to use as output file name
        input_filename = os.path.basename(input_file_path)
        
        # Normalize path to handle mixed slashes
        original_file_path = raw_data.path.replace('\\', '/')
        
        # Extract file_name (name with extension)
        # Handle both forward and backslashes by splitting on both
        parts = re.split(r'[/\\]', original_file_path)
        file_name = parts[-1] if parts else ''
        
        # Extract document_type (extension without the dot)
        document_type = os.path.splitext(file_name)[1].lstrip('.').lower()
        if not document_type:
            document_type = 'unknown'
        
        # Create chunks from the text
        if not raw_data.text:
            logger.warning(f"Empty text in RawData: {input_file_path}")
            chunk_texts = []
        else:
            chunk_texts = self.create_chunks(raw_data.text)
        
        # Create Chunk objects
        chunks = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk = Chunk(
                chunk_id=f"{raw_data.document_id}_chunk_{i}",
                text=chunk_text,
                keywords=[],  # Empty keywords initially
                summary="",   # Empty summary initially
                queries=[]    # Empty queries initially
            )
            chunks.append(chunk)
        
        # Create Document with the original document_id
        document = Document(
            document_id=raw_data.document_id,  # Use document_id from RawData
            path=original_file_path,  # Use the normalized path
            document_type=document_type,
            file_name=file_name,
            processed_at=datetime.now().isoformat(),
            chunks=chunks
        )
        
        # Create ChunkedDocument
        chunked_document = ChunkedDocument()
        chunked_document.add_document(document)
        
        return chunked_document
    
    def process_raw_data_directory(self, input_dir: str = "Data/Input/Raw", output_dir: str = "Data/Output") -> List[str]:
        """
        Process all RawData JSON files in a directory.
        
        Args:
            input_dir: Directory containing RawData JSON files
            output_dir: Directory to save the output ChunkedDocument JSON files
            
        Returns:
            List of paths to the saved output files
        """
        logger.info(f"Processing raw data directory: {input_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all JSON files in the input directory
        output_files = []
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.json'):
                input_file_path = os.path.join(input_dir, file_name)
                try:
                    output_file_path = self.process_raw_data_file(input_file_path, output_dir)
                    output_files.append(output_file_path)
                except Exception as e:
                    logger.error(f"Error processing {input_file_path}: {str(e)}")
                    continue
        
        logger.info(f"Processed {len(output_files)} raw data files")
        return output_files
    
    def create_chunks(self, text: str) -> List[str]:
        """
        Create text chunks from the input text.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Split text into sentences
        sentences = self._find_sentence_boundaries(text)
        
        # Connect sentences into chunks
        chunks = self._connect_phrases(sentences)
        
        # Process very long chunks
        new_chunks = []
        for chunk in chunks:
            if self._word_count(chunk) <= self.max_chunk_size:
                new_chunks.append(chunk)
            else:
                new_chunks.extend(self._create_chunks_from_points(chunk))
        
        return new_chunks
    
    def _find_sentence_boundaries(self, text: str) -> List[str]:
        """
        Find sentence boundaries using spaCy.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
    
    def _connect_phrases(self, phrases: List[str]) -> List[str]:
        """
        Connect sentences into chunks with maximum word count (max_chunk_size),
        but never split a sentence in the middle.
        
        Args:
            phrases: List of sentences
            
        Returns:
            List of connected chunks
        """
        chunks = []
        current_chunk = []
        current_chunk_len = 0

        for phrase in phrases:
            words_in_phrase = self._word_count(phrase)
            
            # If a single phrase exceeds max size and we don't have anything in the current chunk,
            # add it as is (we'll split it later if needed)
            if words_in_phrase > self.max_chunk_size and not current_chunk:
                chunks.append(phrase)
                continue
            
            # If adding this phrase doesn't exceed max size, add it to current chunk
            if current_chunk_len + words_in_phrase <= self.max_chunk_size:
                current_chunk.append(phrase)
                current_chunk_len += words_in_phrase
            else:
                # Current chunk is full, save it and start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [phrase]
                current_chunk_len = words_in_phrase

        # Add any remaining content
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    @staticmethod
    def _word_count(text: str) -> int:
        """
        Count words in text.
        
        Args:
            text: Text to count words in
            
        Returns:
            Number of words
        """
        return len(text.split())

    def _create_chunks_from_points(self, text: str) -> List[str]:
        """
        Create chunks from text containing points (e.g., numbered lists),
        preserving the structure of points.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunks
        """
        intro, points = self._split_text_into_points(text)
        chunks = []
        
        intro_word_count = self._word_count(intro)
        if intro_word_count > self.max_chunk_size:
            chunks.append(intro.strip())
            current_chunk = ""
            current_word_count = 0
        else:
            current_chunk = intro
            current_word_count = intro_word_count

        for point in points:
            point_word_count = self._word_count(point)
            
            # If a single point is too large, add it as a separate chunk
            if point_word_count > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                chunks.append(point.strip())
                current_chunk = ""
                current_word_count = 0
                continue
            
            # If adding this point would exceed max size, finish the current chunk and start a new one
            if current_word_count + point_word_count > self.max_chunk_size:
                chunks.append(current_chunk.strip())
                if intro_word_count <= self.max_chunk_size:
                    current_chunk = intro
                    current_word_count = intro_word_count
                else:
                    current_chunk = ""
                    current_word_count = 0

            current_chunk += " " + point
            current_word_count += point_word_count

        # Add any remaining content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    @staticmethod
    def _split_text_into_points(text: str) -> tuple[str, List[str]]:
        """
        Split text into introduction and points.
        
        Args:
            text: Text to split
            
        Returns:
            Tuple of (introduction, list of points)
        """
        pattern = r"(\d+)\s+(?=[A-Za-z])"
        matches = list(re.finditer(pattern, text))

        if not matches:
            return text, []

        intro = text[:matches[0].start()].strip()
        points = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            points.append(text[start:end].strip())

        return intro, points