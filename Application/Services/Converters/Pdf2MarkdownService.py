# PdfLoaderService.py
from typing import Optional, Dict
import fitz  # PyMuPDF
import re

class Pdf2MarkdownService:
    def __init__(self):
        pass
        
    def load_pdf(self, file_path: str, start_page: int = 0, end_page: Optional[int] = None) -> str:
        """
        Load PDF file and convert it to markdown text.
        
        Args:
            file_path: Path to the PDF file
            start_page: Starting page number (0-based)
            end_page: Ending page number (None for all pages)
            
        Returns:
            str: Markdown formatted text from PDF
        """
        markdown_text = ""
        
        try:
            # Open the PDF
            doc = fitz.open(file_path)
            
            # Set end page if not specified
            if end_page is None:
                end_page = len(doc)
            
            # Process each page
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                
                # Extract text blocks
                blocks = page.get_text("blocks")
                
                # Process each block
                for block in blocks:
                    text = block[4]  # block[4] contains the text
                    
                    # Clean the text
                    text = self._clean_text(text)
                    
                    # Detect and format headers
                    if self._is_header(text):
                        markdown_text += f"# {text}\n\n"
                    else:
                        markdown_text += f"{text}\n\n"
                
                # Add page separator
                markdown_text += f"---\nPage {page_num + 1}\n---\n\n"
            
            return markdown_text
            
        except Exception as e:
            raise Exception(f"Error loading PDF: {str(e)}")
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters (customize as needed)
        text = re.sub(r'[^\w\s.,!?;:()\-\"\'§]', '', text)
        return text
        
    def _is_header(self, text: str) -> bool:
        """
        Detect if text block is likely a header.
        Basic implementation - can be enhanced with more sophisticated rules.
        """
        # Check for common header patterns
        header_patterns = [
            r'^\d+\.\s',  # Numbered sections
            r'^[A-Z\s]{10,}$',  # All caps text
            r'^(Section|Chapter|SECTION|CHAPTER)',  # Explicit header words
            r'^§\s*\d+',  # Section symbol
        ]
        
        return any(re.match(pattern, text) for pattern in header_patterns)