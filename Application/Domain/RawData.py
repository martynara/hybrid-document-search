from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import json
import os

@dataclass
class RawData:
    """
    Raw data class that contains document text.
    """
    path: str
    file_name: str
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_type: str = "unknown"
    area: str = ""
    text: Optional[str] = None
    
    def __init__(self, file_path: str, document_type: str = "unknown", text: Optional[str] = None, area: str = ""):
        """
        Initialize a Document instance.
        
        Args:
            path: Relative path to the document (e.g., "Data/Input/PDF").
            text: Text data of document.
        """
        self.document_id = str(uuid.uuid4())
        self.document_type = document_type
        self.text = text

        # Split the path and filename
        path = os.path.dirname(file_path)  # Gets the directory path
        file_name = os.path.basename(file_path)  # Gets just the filename

        self.path = path
        self.file_name = file_name
        
    def set_text(self, text: str) -> None:
        """
        Set text of document.
        
        Args:
            text: The text to set
            
        Returns:
            None
        """
        self.text = text
    
    def set_path(self, path: str) -> None:
        """
        Set path of document.
        
        Args:
            path: The path to set
            
        Returns:
            None
        """
        self.path = path
    
    def to_json(self) -> str:
        """
        Convert the RawData instance to a JSON string.
        
        Returns:
            A JSON string representation of the RawData instance
        """
        return json.dumps(asdict(self), indent=4, sort_keys=True)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RawData':
        """
        Create a RawData instance from a JSON string.
        
        Args:
            json_str: JSON string representation of a RawData instance
            
        Returns:
            A RawData instance
        """
        data = json.loads(json_str)
        # Create a raw_data instance
        raw_data = cls(file_path=data['path']+data['file_name'], document_type=data["document_type"], text=data.get('text'), area=data.get('area'))
        
        # Make sure to preserve the original document_id if it exists
        if 'document_id' in data:
            raw_data.document_id = data['document_id']
            
        return raw_data
    
    def save_to_file(self, filename: str) -> None:
        """
        Save the RawData instance to a JSON file.
        
        Args:
            filename: Name of the file to save to
            
        Returns:
            None
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'RawData':
        """
        Load a RawData instance from a JSON file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            A RawData instance
        
        Raises:
            FileNotFoundError: If the specified file does not exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        
        with open(filename, 'r', encoding='utf-8') as f:
            json_str = f.read()
            
        return cls.from_json(json_str)