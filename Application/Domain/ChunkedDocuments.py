from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import json
import os

@dataclass
class ChunkMetadata:
    document_id: str = ""
    chunk_id: str = ""
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    queries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id or str(uuid.uuid4()),
            "keywords": self.keywords,
            "summary": self.summary,
            "queries": self.queries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        return cls(
            document_id=data.get('document_id', ''),
            chunk_id=data.get('chunk_id', str(uuid.uuid4())),
            keywords=data.get('keywords', []),
            summary=data.get('summary', ''),
            queries=data.get('queries', [])
        )

@dataclass
class Chunk:
    text: str = ""
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        return cls(
            text=data.get('text', ''),
            metadata=ChunkMetadata.from_dict(data.get('metadata', {})),
        )

@dataclass
class Document:
    document_id: str = ""
    path: str = ""
    document_type: str = ""
    file_name: str = ""
    processed_at: str = ""
    total_chunks: int = 0
    chunks: List[Chunk] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.document_id:
            self.document_id = str(uuid.uuid4())
        if not self.processed_at:
            self.processed_at = datetime.now().isoformat()
        self.total_chunks = len(self.chunks)
        
    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)
        self.total_chunks = len(self.chunks)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "path": self.path,
            "document_type": self.document_type,
            "file_name": self.file_name,
            "processed_at": self.processed_at,
            "total_chunks": self.total_chunks,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        return cls(
            document_id=data.get('document_id', ''),
            path=data.get('path', ''),
            document_type=data.get('document_type', ''),
            file_name=data.get('file_name', ''),
            processed_at=data.get('processed_at', ''),
            chunks=[Chunk.from_dict(chunk) for chunk in data.get('chunks', [])]
        )

@dataclass
class ChunkedDocuments:
    documents: List[Document] = field(default_factory=list)
    
    def add_document(self, document: Document) -> None:
        self.documents.append(document)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "documents": [doc.to_dict() for doc in self.documents]
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkedDocuments':
        return cls(
            documents=[Document.from_dict(doc) for doc in data.get('documents', [])]
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ChunkedDocuments':
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, output_file_path: str) -> bool:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return True
            
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ChunkedDocuments':
        with open(file_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return cls.from_json(json_str)