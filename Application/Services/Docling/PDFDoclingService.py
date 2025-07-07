import json
import uuid
import logging
import re
from datetime import datetime
from typing import Dict, List, Literal, Optional, Any
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker import HierarchicalChunker
from docling_core.types.doc.base import ImageRefMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentHierarchyExtractorService:
    """
    A service class for extracting hierarchical structure from markdown documents.
    """

    def __init__(self):
        # Regular expression to match headers (# Header) with optional $ marker
        self.header_pattern = re.compile(r'^(#{1,6})\s+(\$)?\s*(.+)$')

    def extract_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract hierarchy from a markdown document and generate paths.
        
        Args:
            text (str): The markdown text to parse
            
        Returns:
            list: A list of dictionaries containing hierarchy info with paths
        """
        # Split the document into lines
        lines = text.split('\n')
        
        hierarchy = []
        current_path = []
        current_levels = []
        
        for line in lines:
            match = self.header_pattern.match(line.strip())
            if match:
                level = len(match.group(1))  # Number of # symbols
                is_paragraph = bool(match.group(2))  # Check if $ is present
                title = match.group(3).strip()
                
                # Adjust the current path based on the header level
                while current_levels and current_levels[-1] >= level:
                    current_path.pop()
                    current_levels.pop()
                
                # Add the new title to the path
                current_path.append(title)
                current_levels.append(level)
                
                # Create path string
                path_str = '/'.join(current_path)
                
                # Create hierarchy item
                hierarchy_item = {
                    'level': level,
                    'title': title,
                    'is_paragraph': is_paragraph,
                    'path': path_str,
                    'original_line': line.strip()
                }
                
                hierarchy.append(hierarchy_item)
        
        return hierarchy

    def get_full_paths(self, text: str) -> List[Dict[str, str]]:
        """
        Get a simplified list of titles and their full paths.
        
        Args:
            text (str): The markdown text to parse
            
        Returns:
            list: A list of dictionaries with title and path
        """
        hierarchy = self.extract_hierarchy(text)
        return [{'title': item['title'], 'path': item['path']} for item in hierarchy]
    
    def find_path_for_text(self, text: str, content: str) -> str:
        """
        Find the hierarchical path for a given text chunk based on its content and headers.
        
        Args:
            text (str): The text chunk to find a path for
            content (str): The full markdown content
            
        Returns:
            str: The hierarchical path for the text
        """
        # Get all hierarchical paths
        hierarchy = self.extract_hierarchy(content)
        
        # Default path is empty
        current_path = ""
        
        # Extract the start of the text (first 100 chars to identify it)
        text_start = text[:100] if len(text) > 100 else text
        text_start = text_start.strip()
        
        # Find the section in the content that contains this text
        content_lines = content.split('\n')
        
        # Look for the text in the content and track the previous headers
        for i, line in enumerate(content_lines):
            # If we find the start of our text
            if text_start in line:
                # Look backward to find the most recent header
                for j in range(i, -1, -1):
                    previous_line = content_lines[j]
                    # Check if it's a header
                    if previous_line.strip().startswith('#'):
                        # Find this header in our hierarchy
                        for item in hierarchy:
                            if item['original_line'] in previous_line:
                                return item['path']
                        break
        
        return current_path


class PDFDoclingService:
    """
    Service for converting PDF files to markdown using Docling.
    The document is saved in markdown format, preserving its structure.
    """

    def __init__(self, 
                 input_path: str = "Data/Input/PDF", 
                 output_dir: str = "Data/Processed/Chunks",
                 tokenizer: str = "sdadas/st-polish-paraphrase-from-distilroberta",
                 chunker_type: Literal["hybrid", "hierarchical"] = "hybrid",
                 processing_mode: Literal["accurate", "fast"] = "accurate"):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.chunker_type = chunker_type
        self.processing_mode = processing_mode
        self.hierarchy_extractor = DocumentHierarchyExtractorService()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.input_path.is_file() and self.input_path.suffix.lower() == '.pdf':
            self.is_single_file = True
            logger.info(f"Processing single PDF file: {self.input_path}")
        elif self.input_path.is_dir():
            self.is_single_file = False
            logger.info(f"Processing all PDF files in directory: {self.input_path}")
        else:
            if not self.input_path.exists():
                logger.warning(f"Input path does not exist: {self.input_path}")
            elif self.input_path.is_file() and self.input_path.suffix.lower() != '.pdf':
                logger.warning(f"Input file is not a PDF: {self.input_path}")
            else:
                logger.warning(f"Invalid input path: {self.input_path}")

    def process(self) -> List[str]:
        if self.is_single_file:
            result = self.process_pdf(self.input_path)
            return [result] if result else []
        else:
            return self.process_all_pdfs()

    def process_all_pdfs(self) -> List[str]:
        processed_files = []
        if not self.input_path.is_dir():
            logger.error(f"Input path is not a directory: {self.input_path}")
            return processed_files
        for pdf_file in self.input_path.glob("*.pdf"):
            try:
                output_file = self.process_pdf(pdf_file)
                processed_files.append(output_file)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
        return processed_files

    def process_pdf(self, pdf_path: Path) -> Optional[str]:
        try:
            if not pdf_path.exists():
                logger.error(f"PDF file does not exist: {pdf_path}")
                return None
            if pdf_path.suffix.lower() != '.pdf':
                logger.error(f"File is not a PDF: {pdf_path}")
                return None
            base_name = pdf_path.stem
            output_file = self.output_dir / f"{base_name}-docling.md"
            converter = DocumentConverter()
            conv_res = converter.convert(str(pdf_path))
            doc = conv_res.document
            doc.save_as_markdown(output_file, image_mode=ImageRefMode.PLACEHOLDER)
            
            # Read the markdown content for hierarchy extraction
            with open(output_file, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            
            # Extract document hierarchy
            document_hierarchy = self.hierarchy_extractor.extract_hierarchy(markdown_content)
            
            if self.chunker_type == "hybrid":
                self._process_with_hybrid_chunker(doc, pdf_path, output_file, markdown_content)
            else:
                self._process_with_hierarchical_chunker(doc, pdf_path, output_file, markdown_content)
            logger.info(f"Successfully processed {pdf_path} to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _process_with_hybrid_chunker(self, doc, pdf_path: Path, output_file: Path, markdown_content: str) -> None:
        chunker = HybridChunker(tokenizer=self.tokenizer)
        chunk_iter = chunker.chunk(doc)
        structure_file = self.output_dir / f"{pdf_path.stem}-structure.json"
        chunks = []
        
        # Get file name for structure field
        file_name = pdf_path.stem
        
        for chunk in chunk_iter:
            text = chunker.serialize(chunk)
            
            # Default structure is just the filename
            structure = file_name
            
            # First try to extract path using document hierarchy
            hierarchical_path = self.hierarchy_extractor.find_path_for_text(text, markdown_content)
            if hierarchical_path:
                structure = f"{file_name}/{hierarchical_path}"
            # Fallback to meta headings if hierarchy extraction failed
            elif hasattr(chunk, 'meta'):
                meta_data = chunk.meta.model_dump() if hasattr(chunk.meta, 'model_dump') else vars(chunk.meta)
                headings = meta_data.get('headings', [])
                if headings and len(headings) > 0:
                    structure = f"{file_name}/{headings[0]}"
            
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": text,
                "structure": structure
            })
        
        # Save chunks in the JSON file
        with open(structure_file, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
        
        self._add_structure_annotations_to_markdown(output_file, chunks)

    def _process_with_hierarchical_chunker(self, doc, pdf_path: Path, output_file: Path, markdown_content: str) -> None:
        chunker = HierarchicalChunker()
        chunk_iter = chunker.chunk(doc)
        structure_file = self.output_dir / f"{pdf_path.stem}-structure.json"
        chunks = []
        
        # Get file name for structure field
        file_name = pdf_path.stem
        
        for chunk in chunk_iter:
            text = chunker.serialize(chunk)
            
            # Default structure is just the filename
            structure = file_name
            
            # First try to extract path using document hierarchy
            hierarchical_path = self.hierarchy_extractor.find_path_for_text(text, markdown_content)
            if hierarchical_path:
                structure = f"{file_name}/{hierarchical_path}"
            # Fallback to path attribute if hierarchy extraction failed
            elif hasattr(chunk, 'path') and chunk.path:
                # Use the document path from the chunk for better structure representation
                structure = f"{file_name}/{chunk.path}"
            
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": text,
                "structure": structure
            })
        
        # Save chunks in the JSON file
        with open(structure_file, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
        
        self._add_structure_annotations_to_markdown(output_file, chunks)

    def _add_structure_annotations_to_markdown(self, markdown_file: Path, chunks: List[Dict]) -> None:
        with open(markdown_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        
        # Extract document hierarchy for table of contents
        hierarchy = self.hierarchy_extractor.extract_hierarchy(markdown_content)
        
        # Create a table of contents section
        toc_lines = ["# Document Structure"]
        for item in hierarchy:
            indent = "  " * (item['level'] - 1)
            toc_lines.append(f"{indent}- {item['title']}")
        
        toc_content = "\n".join(toc_lines)
        
        # Add metadata and TOC
        metadata_section = f"""
# Document Metadata
- Total chunks: {len(chunks)}
- Chunker type: {self.chunker_type}
- Processing mode: {self.processing_mode}
- Tokenizer: {self.tokenizer}

{toc_content}

---

"""
        markdown_content = metadata_section + markdown_content
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

    def _detect_language(self) -> str:
        if "polish" in self.tokenizer.lower():
            return "pl"
        return "pl"
