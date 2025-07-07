import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class SearchRow:
    text: str
    area: str
    score: float = 0.0
    chunk_id: str = ""
    links: List[str] = field(default_factory=list)
    metadata: Optional[Dict] = None
    
    # The __post_init__ method is redundant when using field(default_factory=list)
    # and we're manually defining __init__, so it can be removed
    
    def __init__(self, text: str, area: str, score: float = 0.0, chunk_id: str = "", links: List[str] = None,
                 metadata: Optional[Dict] = None):
        self.text = text
        self.area = area
        self.score = score
        self.chunk_id = chunk_id
        self.links = links if links is not None else []
        self.metadata = metadata