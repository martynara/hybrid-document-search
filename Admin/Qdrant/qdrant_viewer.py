#!/usr/bin/env python3
import asyncio
import logging
from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantViewer:
    """
    A simple viewer for Qdrant vector database that displays the top 10 records
    """

    def __init__(self, collection_name):
        """
        Initialize the QdrantViewer with the database path and collection name
        
        Args:
            db_path (str): Path to the Qdrant database
            collection_name (str): Name of the collection to query
        """
        self.db_path = "Data/VectorDB"
        self.collection_name = collection_name
        self.client = QdrantClient(path=self.db_path)
        
    def get_top_records(self, limit=10):
        """
        Retrieve and display the top records from the Qdrant collection
        
        Args:
            limit (int): Maximum number of records to retrieve
            
        Returns:
            list: List of records from the database
        """
        # Get the top records from the collection
        # Since we're not doing a vector search, we'll just get the points by ID
        records = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False  # Usually vectors are large, so skip unless needed
        )
        
        return records[0]  # Scroll returns a tuple (points, next_page_offset)
    
    def display_records(self, records):
        """
        Print the records in a readable format
        
        Args:
            records (list): List of records to display
        """
        print(f"\nDisplaying top {len(records)} records from collection '{self.collection_name}':\n")
        print("-" * 80)
        
        for i, record in enumerate(records, 1):
            print(f"Record #{i} (ID: {record.id}):")
            print(f"Text: {record.payload.get('text', 'N/A')[:200]}...")
            
            # Display metadata if available
            if 'metadata' in record.payload:
                print("Metadata:")
                for key, value in record.payload['metadata'].items():
                    print(f"  {key}: {value}")
            
            print("-" * 80)


def main():
    try:
        viewer = QdrantViewer("documents")
        records = viewer.get_top_records(10)
        viewer.display_records(records)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()