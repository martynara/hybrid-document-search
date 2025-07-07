from dataclasses import dataclass, field
from typing import List
from typing import Optional
from Application.Domain.SearchRow import SearchRow

@dataclass
class SearchResults:
    rows: List[SearchRow] = field(default_factory=list)
    count: int = 0