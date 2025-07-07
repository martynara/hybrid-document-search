from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class QAPair(BaseModel):
    """Model representing a question-answer pair"""
    query: str = Field(..., description="The question text")
    answer: str = Field(..., description="The answer text")


class DocumentMetadata(BaseModel):
    """Model representing document metadata"""
    # Common metadata fields
    source: Optional[str] = Field(None, description="Source of the document")
    language: Optional[str] = Field(None, description="Language of the document")
    document_type: Optional[str] = Field(None, description="Type of document")
    section: Optional[str] = Field(None, description="Section within the document")
    file_name: Optional[str] = Field(None, description="Original file name")
    
    # Additional metadata fields
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    
    # Extensible field for any other metadata
    additional_metadata: Dict[str, Any] = Field(default_factory=dict, description="Any additional metadata fields")


class QdrantDocument(BaseModel):
    """Model representing a document in Qdrant"""
    # Main content field
    text: str = Field(..., description="The main text content of the document")
    
    # Metadata fields
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")
    
    # Enrichment fields at root level
    keywords: List[str] = Field(default_factory=list, description="Keywords extracted from the document")
    summary: Optional[str] = Field(None, description="Summary of the document content")
    qa_pairs: List[QAPair] = Field(default_factory=list, description="Question-answer pairs generated from the document")
    qa_count: Optional[int] = Field(None, description="Number of QA pairs")
    qa_updated_at: Optional[str] = Field(None, description="When QA pairs were last updated")
    
    # Qdrant metadata fields
    chunk_id: Optional[str] = Field(None, description="Unique identifier for the document chunk")
    vector_id: Optional[str] = Field(None, description="ID of the vector in Qdrant")
    score: Optional[float] = Field(None, description="Relevance score (used in search results)")


class QdrantDocumentInput(BaseModel):
    """Model for inserting a new document into Qdrant"""
    text: str = Field(..., description="The main text content of the document")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")
    chunk_id: Optional[str] = Field(None, description="Optional chunk ID (will be generated if not provided)")


class QdrantDocumentUpdate(BaseModel):
    """Model for updating an existing document in Qdrant"""
    chunk_id: str = Field(..., description="ID of the document to update")
    keywords: Optional[List[str]] = Field(None, description="Updated keywords")
    summary: Optional[str] = Field(None, description="Updated summary")
    qa_pairs: Optional[List[QAPair]] = Field(None, description="Updated QA pairs")
    metadata_updates: Optional[Dict[str, Any]] = Field(None, description="Updates to metadata fields")


# Utility functions
def convert_to_qdrant_payload(document: QdrantDocumentInput) -> Dict[str, Any]:
    """Convert a Pydantic model to the format expected by Qdrant insertion"""
    return {
        "text": document.text,
        "metadata": document.metadata.dict(exclude_none=True),
    }


def update_payload_from_model(update: QdrantDocumentUpdate) -> Dict[str, Any]:
    """Create an update payload from a QdrantDocumentUpdate model"""
    payload = {}
    
    if update.keywords is not None:
        payload["keywords"] = update.keywords
    
    if update.summary is not None:
        payload["summary"] = update.summary
        
    if update.qa_pairs is not None:
        payload["qa_pairs"] = [qa.dict() for qa in update.qa_pairs]
        payload["qa_count"] = len(update.qa_pairs)
        payload["qa_updated_at"] = datetime.now().isoformat()
    
    if update.metadata_updates:
        # For nested metadata updates
        payload["metadata"] = update.metadata_updates
    
    return payload 