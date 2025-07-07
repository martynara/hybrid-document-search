import logging
import os
from typing import Any, Dict

from Application.Domain.RawData import RawData
from Application.Services.PDF.TextFromPDFService import TextFromPDFService

# Configure logging to display in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LoadPDFPipeline:
    """
    Pipeline that processes PDF files to Document objects
    """

    def __init__(self, input_dir="Data/Input/Pdf", output_dir="Data/Input/Raw"):
        self.logger = logging.getLogger(__name__)

        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.text_from_pdf_service = TextFromPDFService(input_dir=input_dir)
       
    def process_single_file(self, pdf_filename: str) -> str:
        """
        Process a single PDF document through the entire pipeline.
        
        Args:
            pdf_filename: Name of the PDF file to process
            
        Returns:
            Raw data object
        """

        base_name, file_ext = os.path.splitext(pdf_filename)
        
        # Construct and normalize the full path to prevent mixed slashes
        pdf_full_path = os.path.join(self.input_dir, pdf_filename)
        pdf_full_path = pdf_full_path.replace('\\', '/') # Normalize slashes to forward slashes
        
        raw_data = RawData(file_path=pdf_full_path, document_type="pdf")

        self.logger.info(f"Step 1: Extracting text from PDF file {pdf_filename}...")
        try:
            pdf_path = os.path.join(self.input_dir, pdf_filename)

            extracted_text = self.text_from_pdf_service.convert_pdf(pdf_path)

        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {pdf_filename}. Error: {str(e)}")
            raise

        self.logger.info(f"Step 1: Extracting text from PDF file {pdf_filename}...")
        try:
            raw_output_path = os.path.join(self.output_dir, f"{base_name}.json")

            raw_data.set_text(extracted_text)
            extracted_text = raw_data.save_to_file(raw_output_path)

        except Exception as e:
            self.logger.error(f"Failed to extract text from PDF: {pdf_filename}. Error: {str(e)}")
            raise
        
        return extracted_text