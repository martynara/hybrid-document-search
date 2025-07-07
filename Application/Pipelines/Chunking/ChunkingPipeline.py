import logging
import os
from typing import Any, Dict, List

from Application.Domain.RawData import RawData
from Application.Domain.ChunkedDocuments import ChunkedDocuments
from Application.Services.Chunking.TextChunkingNLPDocumentService import TextChunkingNLPDocumentService

logger = logging.getLogger(__name__)

class ChunkingPipeline:
    """
    Pipeline that processes RawData files to ChunkedDocument objects
    """

    def __init__(self, input_dir="Data/Input/Raw", output_dir="Data/Output"):
        self.logger = logging.getLogger(__name__)

        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.chunking_service = TextChunkingNLPDocumentService()
       
    def process_raw_data(self, raw_data: RawData) -> ChunkedDocuments:
        """
        Process a RawData through the chunking pipeline.
        
        Args:
            raw_data: RawData to process
            
        Returns:
            ChunkedDocument object
        """
        self.logger.info(f"Processing {raw_data.document_id} through chunking pipeline...")
        
        try:
            # Process the raw data file using the chunking service
            chunked_documents = self.chunking_service.process_raw_data(raw_data=raw_data)
            
            return chunked_documents
            
        except Exception as e:
            self.logger.error(f"Failed to process file: {raw_data.document_id}. Error: {str(e)}")
            raise


    def process_single_file(self, raw_data_filename: str) -> str:
        """
        Process a single RawData JSON file through the chunking pipeline.
        
        Args:
            raw_data_filename: Name of the JSON file to process
            
        Returns:
            Path to the output ChunkedDocument JSON file
        """
        self.logger.info(f"Processing file {raw_data_filename} through chunking pipeline...")
        
        try:
            input_file_path = os.path.join(self.input_dir, raw_data_filename)
            
            # Process the raw data file using the chunking service
            output_file_path = self.chunking_service.process_raw_data_file(
                input_file_path=input_file_path,
                output_dir=self.output_dir
            )
            
            self.logger.info(f"Successfully processed {raw_data_filename} to {output_file_path}")
            return output_file_path
            
        except Exception as e:
            self.logger.error(f"Failed to process file: {raw_data_filename}. Error: {str(e)}")
            raise
            
    def process_directory(self) -> List[str]:
        """
        Process all RawData JSON files in the input directory.
        
        Returns:
            List of paths to output ChunkedDocument JSON files
        """
        self.logger.info(f"Processing all files in {self.input_dir} through chunking pipeline...")
        
        try:
            # Process all files in the directory using the chunking service
            output_files = self.chunking_service.process_raw_data_directory(
                input_dir=self.input_dir,
                output_dir=self.output_dir
            )
            
            self.logger.info(f"Successfully processed {len(output_files)} files")
            return output_files
            
        except Exception as e:
            self.logger.error(f"Failed to process directory. Error: {str(e)}")
            raise
