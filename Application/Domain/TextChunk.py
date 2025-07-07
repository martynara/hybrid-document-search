import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class TextChunk:
    text: str
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __init__(self, text: str, chunk_id: str = None, metadata: Dict[str, Any] = None, embedding: Optional[List[float]] = None):
        self.text = text
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.metadata = metadata or {}
        self.embedding = embedding
    
    def __post_init__(self):
        self.chunk_id = self.chunk_id or str(uuid.uuid4())
        
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextChunk':
        return cls(
            text=data.get('text', ''),
            chunk_id=data.get('chunk_id', str(uuid.uuid4())),
            metadata=data.get('metadata', {}),
            embedding=data.get('embedding'),
        )