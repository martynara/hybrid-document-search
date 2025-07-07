# Hybrid Document Search

A hybrid document search and retrieval system that combines multiple search strategies to provide highly relevant results. Built with NLP techniques, multi-collection vector databases, and AI-powered answer generation, optimized for Polish language processing.

## Features

- **Multi-Collection Search Architecture**: Four specialized vector collections for different search perspectives
- **AI-Powered Document Enhancement**: Automatic keyword extraction, summarization, and Q&A generation  
- **Intelligent Text Chunking**: spaCy-based semantic chunking with sentence boundary preservation
- **Scoring System**: Cross-collection bonus scoring for improved relevance
- **Quality Monitoring**: Built-in evaluation system with comprehensive metrics
- **Neural Reranking**: Polish-specific RoBERTa model for result refinement


## Architecture Overview

```
PDF Documents → Processing Pipeline → Vector Collections → Search Engine → User Interface
     ↓               ↓                    ↓               ↓            ↓
   Docling      Chunking+Metadata    4 Specialized    Advanced     Streamlit
  Extraction    (Keywords,Summary,QA)  Collections   Multi-Search   Frontend
```

### Multi-Collection Strategy

1. **`documents`** - Primary collection with original text chunks
2. **`documents_keywords`** - Enhanced with extracted keywords (+1.5 bonus)
3. **`documents_summaries`** - Contains summaries for concept search (+0.8 bonus)
4. **`documents_queries`** - Q&A pairs for natural questions (+0.5 bonus)

## Quick Start

```

### 🎯 Quick Test Run

```bash
# Start the application (recommended)
python start_app.py

# Alternative: Direct Streamlit
streamlit run home.py

# If you get database lock errors
python fix_database_lock.py
```

## 📊 Usage

### 1. Start the Application

**Option A: Smart Startup (Recommended)**
```bash
# Automatically handles dependencies, cleanup, and startup
python start_app.py

# Custom port
python start_app.py --port 8502

# Skip checks (if you're sure everything is installed)
python start_app.py --skip-checks
```

**Option B: Direct Streamlit**
```bash
streamlit run home.py
```

**Access Points:**
- **Main Application**: http://localhost:8501
- **Search Interface**: http://localhost:8501/Wyszukiwarka  
- **Quality Monitor**: http://localhost:8501/Quality_monitor

### 2. Document Processing Pipeline (Optional)

To process your own documents, place PDF files in `Data/Input/PDF/` and run:

```bash
# Step 1: Extract text from PDFs
uv run python Application/Process/01_load_pdf_nlp.py

# Step 2: Create intelligent text chunks  
uv run python Application/Process/02_chunk_json.py

# Step 3: Generate Q&A metadata
uv run python Application/Process/03_metadata_query.py

# Step 4: Generate summaries
uv run python Application/Process/04_metadata_summary.py

# Step 5: Extract keywords
uv run python Application/Process/05_metadata_keywords.py

# Step 6: Index to main collection
uv run python Application/Process/09_index_to_documents.py

# Step 7: Index keywords collection
uv run python Application/Process/10_index_to_documents_keywords.py

# Step 8: Index queries collection
uv run python Application/Process/11_index_to_documents_queries.py
```

### 3. Testing Without Documents

The application works even without processed documents:

- UI and search interface will load
- Empty search results demonstrate functionality
- Quality monitoring tools are available
- All features can be explored

### 4. API Usage

```python
from Application.Services.Search.AdvancedSearchService import AdvancedSearchService
from Application.Services.Search.SearchService import SearchService
from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Infrastructure.Services.QdrantManagerService import QdrantManagerService

# Initialize services
qdrant_service = QdrantManagerService(collection_name="documents", vector_size=384)
embedding_service = EmbeddingsService(model_name="all-MiniLM-L6-v2")
search_service = SearchService(
    qdrant_service=qdrant_service, 
    embedding_service=embedding_service
)
advanced_search_service = AdvancedSearchService(
    qdrant_service=qdrant_service,
    search_service=search_service,
    embedding_service=embedding_service
)

# Perform search
results = await advanced_search_service.multi_collection_search(
    query="Gdzie mogą być składane oświadczenia?",
    collections=["documents", "documents_keywords", "documents_summaries"],
    limit=10,
    top_k=50,
    top_a=5
)
```

### 5. Utility Scripts

The project includes several utility scripts for easier management:

```bash
# Fix database lock issues (if you get "Storage folder already accessed" error)
python fix_database_lock.py

# Comprehensive process cleanup and startup
python start_app.py

# Full cleanup (processes + database locks + temp files)
python cleanup_processes.py
```

## Configuration

### Search Parameters

- **`top_k`**: Initial results retrieved from each collection (default: 50)
- **`top_a`**: Top results considered for bonus scoring (default: 5)
- **`limit`**: Final results returned to user (default: 10)

### Collection Bonuses

```python
collection_bonuses = {
    "documents_keywords": 1.5,    # High bonus for keyword matches
    "documents_summaries": 0.8,   # Medium bonus for summary matches
    "documents_queries": 0.5      # Lower bonus for Q&A matches
}
```

### Model Configuration

```python
# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# NLP model
SPACY_MODEL = "pl_core_news_lg"

# Reranker model
RERANKER_MODEL = "sdadas/polish-reranker-roberta-v2"

# LLM model
LLM_MODEL = "gpt-3.5-turbo"
```

## Project Structure

```
HybridDocumentSearch/
├── Application/
│   ├── Domain/                    # Data models and entities
│   ├── Services/
│   │   ├── Search/               # Search services (fixed constructors)
│   │   ├── Embeddings/           # Vector embedding services
│   │   ├── QualityMonitor/       # Quality evaluation (✅ added)
│   │   └── ...                   # Other business logic services
│   ├── Pipelines/                # Document processing pipelines
│   ├── Process/                  # Batch processing scripts
│   └── InternalServices/         # Internal service integrations
├── Infrastructure/
│   └── Services/                 # External service integrations (Qdrant, etc.)
├── Admin/
│   ├── Quality/                  # Quality evaluation tools
│   └── Qdrant/                   # Database administration
├── Data/
│   ├── Input/
│   │   ├── PDF/                  # Input PDF documents
│   │   ├── Raw/                  # Extracted raw text
│   │   └── TXT/                  # Plain text files
│   ├── Output/                   # Processed chunked documents
│   ├── VectorDB/                 # Qdrant vector database
│   └── Models/                   # Cached ML models
├── pages/
│   ├── 01_Wyszukiwarka.py       # Main search interface (✅ fixed)
│   ├── 02_Quality_monitor.py    # Quality monitoring (✅ fixed)
│   └── img/                     # UI assets
├── Utility Scripts (✅ new):
│   ├── start_app.py             # Smart startup with dependency checks
│   ├── cleanup_processes.py     # Process and database cleanup
│   └── fix_database_lock.py     # Quick database lock fix
├── Configuration:
│   ├── pyproject.toml           # UV dependency management
│   ├── .env                     # Environment variables (create manually)
│   └── README.md                # This documentation
└── Entry Points:
    ├── home.py                  # Main Streamlit app (✅ correct case)
    └── st_app.py                # Alternative entry point
```

## Quality Monitoring

### Running Evaluations

```bash
# Install quality monitoring dependencies first
uv sync --extra quality

# Evaluate search quality with advanced metrics
uv run python Admin/Quality/matrix_advanced.py

# Evaluate individual chunk performance
uv run python Admin/Quality/matrix_only_chunks.py

# View evaluation results
cat query_evaluation_results.txt
```

### Metrics Calculated

- **Chunk-level success rate**: Direct chunk ID matches
- **Document-level success rate**: Document-level relevance
- **Precision@K**: Precision at different K values
- **Recall@K**: Recall at different K values
- **Average search latency**: Performance timing

### Quality Monitor Interface

Access the quality monitoring interface at `/Quality_monitor` in the Streamlit app to:

- Add test queries manually
- Upload query-answer JSON files
- Run batch evaluations
- Compare search strategies
- Monitor performance metrics

## 🔧 Development

### Adding New Collections

1. Create specialized search service:
   
```python
class SearchByNewCollection:
    def __init__(self, qdrant_service, embedding_service):
        self.collection_name = "documents_new_collection"
        # Implementation
```

1. Update AdvancedSearchService:

```python
self.collection_service_map = {
    "documents": self.search_service,
    "documents_keywords": self.keywords_service,
    "documents_summaries": self.summaries_service,
    "documents_queries": self.queries_service,
    "documents_new_collection": self.new_collection_service  # Add here
}
```

1. Configure bonus scoring:

```python
self.collection_bonuses = {
    "documents_keywords": 1.5,
    "documents_summaries": 0.8,
    "documents_queries": 0.5,
    "documents_new_collection": 1.0  # Add bonus value
}
```

### Custom Metadata Extraction

Create new metadata services in `Application/Services/Metadata/`:

```python
class CustomMetadataService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def extract_metadata(self, text: str) -> Dict[str, Any]:
        # Custom metadata extraction logic
        pass
```

### Testing

```bash
# Run search examples
uv run python Application/Process/12_search_examples.py

# Test individual components
uv run pytest tests/  # (if tests directory exists)
```

## 📚 API Reference

### Core Services

#### AdvancedSearchService

```python
async def multi_collection_search(
    query: str,
    collections: List[str] = ["documents"],
    limit: int = 5,
    top_k: int = 50,
    top_a: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> SearchResults
```

#### EmbeddingsService

```python
def generate_embedding(self, text: str) -> List[float]
def generate_embeddings(self, chunks: List[TextChunk]) -> List[TextChunk]
```

#### QdrantManagerService

```python
async def search(
    query_vector: List[float],
    limit: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]

async def insert_point(
    text: str,
    embedding: List[float],
    metadata: Dict[str, Any],
    chunk_id: str
) -> Dict[str, Any]
```

## Search Query Examples

### Basic Search
```
"Gdzie mogą być składane oświadczenia związane z likwidacją szkody?"
```

### Complex Queries
```
"Jakie są warunki ubezpieczenia komunikacyjnego dla pojazdów?"
"Procedury składania reklamacji w przypadku szkody"
"Definicja pojazdu mechanicznego w przepisach"
```

### Advanced Filtering
```python
# Search with metadata filters
results = await search_service.advanced_search(
    query="ubezpieczenie",
    metadata={"document_type": "pdf"},
    keywords=["szkoda", "likwidacja"],
    limit=10
)
```

### Startup Options

```bash
# Recommended: Smart startup with all checks
python start_app.py

# Fast startup (skip dependency checks)
python start_app.py --skip-checks

# No cleanup (if processes are clean)
python start_app.py --skip-cleanup

# Custom port (avoid conflicts)
python start_app.py --port 8502

# Direct Streamlit (minimal)
streamlit run home.py --server.port 8503
```

## License

This project is licensed under the MIT License.
