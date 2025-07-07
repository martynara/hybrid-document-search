import asyncio
import logging
from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Application.Services.Reranker.RerankerService import RerankerService
from Infrastructure.Services.QdrantManagerService import QdrantManagerService
from Application.Services.Search.SearchService import SearchService
from Application.Services.Search.AdvancedSearchService import AdvancedSearchService
from Application.Domain.SearchResults import SearchResults
from Application.Domain.SearchRow import SearchRow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_search_examples():
    try:
        qdrant_service = QdrantManagerService(collection_name="documents", vector_size=384)
        embedding_service = EmbeddingsService(model_name="all-MiniLM-L6-v2")

        search_service = SearchService(qdrant_service, embedding_service)
        search_advanced_service = AdvancedSearchService(qdrant_service, search_service, embedding_service)

        reranker_service = RerankerService(model_name="sdadas/polish-reranker-roberta-v2")

        query="Co to jest DZIPU?"

        print("\n=== Wyszukiwanie semantyczne ===")
        results = await search_service.search_semantic(
            query=query,
            limit=5
        )

        display_search_results(results)

        print("\n=== Reranked results ===")
        reranked_results = reranker_service.rerank(query, results, use_probabilities=True)
        display_search_results(reranked_results)

    except Exception as e:
        logger.error(f"Error in search examples: {str(e)}")

def display_search_results(results):
    if not results or results.count == 0:
        print("No results found")
        return

    print(f"\nFound {results.count} results:")
    for i, row in enumerate(results.rows):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {row.score:.4f}")
        print(f"Text: {row.text[:200]}...")
        if row.metadata:
            print(f"Document ID: {row.metadata.get('document_id', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(run_search_examples())