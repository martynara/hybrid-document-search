import os
import tempfile
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime
import time
import shutil
import logging
import asyncio

from Application.Domain.ChunkedDocuments import ChunkedDocuments
from Application.Services.Embeddings.EmbeddingsService import EmbeddingsService
from Infrastructure.Services.QdrantManagerService import QdrantManagerService
from Application.Services.Search.SearchService import SearchService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryEvaluator:
    """
    Evaluates query performance using document query metadata
    """
    
    def __init__(self, search_service: SearchService):
        """Initialize with search service"""
        self.search_service = search_service
        self.query_results = []
    
    def extract_query_chunk_pairs(self, document_json_path: str) -> List[Dict[str, Any]]:
        """
        Extract query-chunk pairs from a document using JsonDocumentService
        
        Args:
            document_json_path: Path to the JSON document file
            
        Returns:
            List of dictionaries with query and chunk_id pairs
        """
        logger.info(f"Processing document: {document_json_path}")
        query_chunk_pairs = []
        
        try:
            # Use JsonDocumentService to read the document
            chunked_documents = ChunkedDocuments.load_from_file(document_json_path)
            
            if not chunked_documents:
                logger.error(f"Failed to load documents from {document_json_path}")
                return query_chunk_pairs
            
            # Process each document
            for document in chunked_documents.documents:
                # Process each chunk
                for chunk in document.chunks:
                    # if 'queries' in chunk.metadata:
                    for query in chunk.metadata.queries:
                        query_chunk_pairs.append({
                            'query': query,
                            'chunk_id': chunk.metadata.chunk_id,
                            'document_id': document.document_id
                        })
                    
                        logger.info(f"Extracted {len(chunk.metadata.queries)} queries from chunk {chunk.metadata.chunk_id}")
            
            logger.info(f"Total extracted query-chunk pairs: {len(query_chunk_pairs)}")

            return query_chunk_pairs
            
        except Exception as e:
            logger.error(f"Error processing document {document_json_path}: {str(e)}")
            return query_chunk_pairs
    
    def extract_query_chunk_pairs_from_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Extract query-chunk pairs from all JSON documents in a folder
        
        Args:
            folder_path: Path to folder containing JSON documents
            
        Returns:
            List of dictionaries with query and chunk_id pairs
        """
        logger.info(f"Processing documents from folder: {folder_path}")
        all_query_chunk_pairs = []
        
        try:
            # Get all JSON files in the folder
            json_files = [str(f) for f in Path(folder_path).glob("*.json")]
            logger.info(f"Found {len(json_files)} JSON files in folder")
            
            # Process each file
            for file_path in json_files:
                logger.info(f"Processing file: {file_path}")
                
                # Extract query-chunk pairs from this file
                file_pairs = self.extract_query_chunk_pairs(file_path)
                
                # Add file path information for reference
                for pair in file_pairs:
                    pair['source_file'] = os.path.basename(file_path)
                
                all_query_chunk_pairs.extend(file_pairs)
                logger.info(f"Extracted {len(file_pairs)} query-chunk pairs from {file_path}")
            
            logger.info(f"Total extracted query-chunk pairs from all files: {len(all_query_chunk_pairs)}")
            return all_query_chunk_pairs
            
        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {str(e)}")
            return all_query_chunk_pairs
    
    async def evaluate_queries(self, 
                             query_chunk_pairs: List[Dict[str, Any]], 
                             limit: int = 50,
                             top_k: int = 10,
                             top_a: int = 3) -> List[Dict[str, Any]]:
        """
        Evaluate a set of queries against the search service
    
        Args:
            query_chunk_pairs: List of dictionaries with query and chunk_id pairs
            top_k: Number of results to retrieve from search service
            top_a: Number of top results to use for metrics calculation
        
        Returns:
            List of evaluation results
        """
        results = []
    
        for pair in query_chunk_pairs:
            query = pair['query']
            expected_chunk_id = pair['chunk_id']
            expected_document_id = pair['document_id']
        
            logger.info(f"Evaluating query: {query}")
        
            # Search for results - get top_k results
            search_results = await self.search_service.search_semantic(query=query, limit=limit)
        
            # Extract chunk IDs, document IDs and texts from results
            retrieved_chunks_data = []
            retrieved_chunk_ids = []
            retrieved_document_ids = []
        
            for row in search_results.rows:
                retrieved_chunk_ids.append(row.chunk_id)
            
                document_id = row.metadata['document_id']
                retrieved_document_ids.append(document_id)
            
                # Store additional data about retrieved chunks for reporting
                retrieved_chunks_data.append({
                    'chunk_id': row.chunk_id,
                    'document_id': document_id,
                    'text': row.text[:100] + "..." if len(row.text) > 100 else row.text,
                    'score': row.score
                })
        
            # Get top_a chunk IDs and document IDs for metrics calculation
            top_a_chunk_ids = retrieved_chunk_ids[:top_a] if len(retrieved_chunk_ids) >= top_a else retrieved_chunk_ids
            top_a_document_ids = retrieved_document_ids[:top_a] if len(retrieved_document_ids) >= top_a else retrieved_document_ids
        
            # Debug document IDs
            logger.info(f"Expected document ID: {expected_document_id}")
            logger.info(f"Retrieved document IDs (top {top_a}): {top_a_document_ids}")
        
            # Standardize document IDs for comparison
            expected_document_id_std = str(expected_document_id).strip().lower()
            top_a_document_ids_std = [str(doc_id).strip().lower() for doc_id in top_a_document_ids if doc_id is not None]
        
            # Calculate chunk-level metrics
            is_chunk_found = expected_chunk_id in top_a_chunk_ids
            chunk_position = retrieved_chunk_ids.index(expected_chunk_id) + 1 if expected_chunk_id in retrieved_chunk_ids else 0
            chunk_mrr = 1.0 / chunk_position if chunk_position > 0 else 0.0
        
            # Calculate document-level metrics with standardized IDs
            is_document_found = expected_document_id_std in top_a_document_ids_std
        
            # Find position with standardized comparison
            document_position = 0
            for i, doc_id in enumerate(retrieved_document_ids):
                if doc_id is not None and str(doc_id).strip().lower() == expected_document_id_std:
                    document_position = i + 1
                    break
        
            document_mrr = 1.0 / document_position if document_position > 0 else 0.0
        
            # Store result
            result = {
                'query': query,
                'expected_chunk_id': expected_chunk_id,
                'expected_document_id': expected_document_id,
                'retrieved_chunks': retrieved_chunk_ids,
                'retrieved_documents': retrieved_document_ids,
                'top_a_chunks': top_a_chunk_ids,
                'top_a_documents': top_a_document_ids,
                'retrieved_chunks_data': retrieved_chunks_data,
                'is_chunk_found': is_chunk_found,
                'chunk_position': chunk_position,
                'chunk_mrr': chunk_mrr,
                'is_document_found': is_document_found,
                'document_position': document_position,
                'document_mrr': document_mrr
            }
        
            results.append(result)
            logger.info(f"Result: Chunk found in top {top_a}={is_chunk_found}, Position={chunk_position}, MRR={chunk_mrr:.4f}")
            logger.info(f"Result: Document found in top {top_a}={is_document_found}, Position={document_position}, MRR={document_mrr:.4f}")
    
        self.query_results = results
        return results

    def calculate_precision_recall_f1(self, query_result, level='chunk'):
        """
        Calculate precision, recall and F1 score for a single query result
        
        Args:
            query_result: Dictionary containing query result data
            level: 'chunk' or 'document' to calculate metrics at different levels
            
        Returns:
            Dictionary with precision, recall, and F1 metrics
        """
        # Determine relevant and retrieved items based on level
        if level == 'chunk':
            relevant = [query_result['expected_chunk_id']]
            retrieved = query_result['top_a_chunks']  # Use top_a chunks for metrics
        elif level == 'document':
            relevant = [query_result['expected_document_id']]
            retrieved = query_result['top_a_documents']  # Use top_a documents for metrics
        else:
            raise ValueError(f"Invalid level: {level}. Must be 'chunk' or 'document'")
            
        # Calculate true positives (correctly retrieved items)
        true_positives = set(retrieved).intersection(set(relevant))
        
        # Calculate precision, recall, F1
        precision = len(true_positives) / len(retrieved) if retrieved else 0
        recall = len(true_positives) / len(relevant) if relevant else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            f'{level}_precision': precision,
            f'{level}_recall': recall,
            f'{level}_f1': f1
        }

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate overall evaluation metrics for both chunk and document levels
        
        Returns:
            Dictionary with metrics
        """
        if not self.query_results:
            return {
                'chunk_success_rate': 0.0,
                'chunk_mrr': 0.0,
                'chunk_avg_position': 0.0,
                'chunk_precision': 0.0,
                'chunk_recall': 0.0,
                'chunk_f1': 0.0,
                'document_success_rate': 0.0,
                'document_mrr': 0.0,
                'document_avg_position': 0.0,
                'document_precision': 0.0,
                'document_recall': 0.0,
                'document_f1': 0.0
            }
        
        # Calculate individual precision, recall, F1 for each query at chunk level
        for result in self.query_results:
            if 'chunk_precision' not in result:  # Only calculate if not already done
                chunk_metrics = self.calculate_precision_recall_f1(result, level='chunk')
                result.update(chunk_metrics)
            
            if 'document_precision' not in result:  # Only calculate if not already done
                document_metrics = self.calculate_precision_recall_f1(result, level='document')
                result.update(document_metrics)
        
        total_queries = len(self.query_results)
        
        # Chunk-level metrics
        chunk_successful_queries = sum(1 for r in self.query_results if r['is_chunk_found'])
        chunk_mrr_sum = sum(r['chunk_mrr'] for r in self.query_results)
        chunk_precision_sum = sum(r['chunk_precision'] for r in self.query_results)
        chunk_recall_sum = sum(r['chunk_recall'] for r in self.query_results)
        chunk_f1_sum = sum(r['chunk_f1'] for r in self.query_results)
        
        # Calculate average position for successful chunk queries only
        chunk_positions = [r['chunk_position'] for r in self.query_results if r['is_chunk_found']]
        chunk_avg_position = sum(chunk_positions) / len(chunk_positions) if chunk_positions else 0
        
        # Document-level metrics
        document_successful_queries = sum(1 for r in self.query_results if r['is_document_found'])
        document_mrr_sum = sum(r['document_mrr'] for r in self.query_results)
        document_precision_sum = sum(r['document_precision'] for r in self.query_results)
        document_recall_sum = sum(r['document_recall'] for r in self.query_results)
        document_f1_sum = sum(r['document_f1'] for r in self.query_results)
        
        # Calculate average position for successful document queries only
        document_positions = [r['document_position'] for r in self.query_results if r['is_document_found']]
        document_avg_position = sum(document_positions) / len(document_positions) if document_positions else 0
        
        metrics = {
            # Chunk-level metrics
            'chunk_success_rate': chunk_successful_queries / total_queries if total_queries > 0 else 0,
            'chunk_mrr': chunk_mrr_sum / total_queries if total_queries > 0 else 0,
            'chunk_avg_position': chunk_avg_position,
            'chunk_precision': chunk_precision_sum / total_queries if total_queries > 0 else 0,
            'chunk_recall': chunk_recall_sum / total_queries if total_queries > 0 else 0,
            'chunk_f1': chunk_f1_sum / total_queries if total_queries > 0 else 0,
            
            # Document-level metrics
            'document_success_rate': document_successful_queries / total_queries if total_queries > 0 else 0,
            'document_mrr': document_mrr_sum / total_queries if total_queries > 0 else 0,
            'document_avg_position': document_avg_position,
            'document_precision': document_precision_sum / total_queries if total_queries > 0 else 0,
            'document_recall': document_recall_sum / total_queries if total_queries > 0 else 0,
            'document_f1': document_f1_sum / total_queries if total_queries > 0 else 0
        }
        
        return metrics

    def generate_report(self, top_k: int = 10, top_a: int = 3) -> str:
        """
        Generate a detailed evaluation report with both chunk and document level metrics
        
        Args:
            top_k: Number of results retrieved from search service
            top_a: Number of top results used for metrics calculation

        Returns:
            Report as a string
        """
        if not self.query_results:
            return "No evaluation results available"

        # First, calculate precision, recall, F1 for each query at both levels
        for i, result in enumerate(self.query_results):
            chunk_metrics = self.calculate_precision_recall_f1(result, level='chunk')
            document_metrics = self.calculate_precision_recall_f1(result, level='document')
            self.query_results[i].update(chunk_metrics)
            self.query_results[i].update(document_metrics)
        
            # Add query_id for each query
            self.query_results[i]['query_id'] = f"q{i+1}"

        # Calculate average metrics at chunk level
        chunk_avg_precision = np.mean([r['chunk_precision'] for r in self.query_results])
        chunk_avg_recall = np.mean([r['chunk_recall'] for r in self.query_results])
        chunk_avg_f1 = np.mean([r['chunk_f1'] for r in self.query_results])
        chunk_avg_mrr = np.mean([r['chunk_mrr'] for r in self.query_results])

        # Calculate average metrics at document level
        document_avg_precision = np.mean([r['document_precision'] for r in self.query_results])
        document_avg_recall = np.mean([r['document_recall'] for r in self.query_results])
        document_avg_f1 = np.mean([r['document_f1'] for r in self.query_results])
        document_avg_mrr = np.mean([r['document_mrr'] for r in self.query_results])

        # Calculate success rates
        total_queries = len(self.query_results)
        
        # Chunk level success rate
        chunk_successful_queries = sum(1 for r in self.query_results if r['is_chunk_found'])
        chunk_success_rate = chunk_successful_queries / total_queries if total_queries > 0 else 0
        
        # Document level success rate
        document_successful_queries = sum(1 for r in self.query_results if r['is_document_found'])
        document_success_rate = document_successful_queries / total_queries if total_queries > 0 else 0

        # Calculate average positions for successful queries only
        chunk_positions = [r['chunk_position'] for r in self.query_results if r['is_chunk_found']]
        chunk_avg_position = sum(chunk_positions) / len(chunk_positions) if chunk_positions else 0
        
        document_positions = [r['document_position'] for r in self.query_results if r['is_document_found']]
        document_avg_position = sum(document_positions) / len(document_positions) if document_positions else 0

        # Calculate total unique chunks and documents retrieved across all queries
        all_chunks = set()
        all_documents = set()
        for result in self.query_results:
            all_chunks.update(result.get('retrieved_chunks', []))
            all_documents.update(result.get('retrieved_documents', []))
        
        total_chunks = len(all_chunks)
        total_documents = len(all_documents)

        # Build the report using UTF-8 characters
        report = "\nSUMMARY METRICS\n"
        report += "--------------\n"
        report += f"Queries evaluated: {len(self.query_results)}\n"
        report += f"Total chunks in collection: {total_chunks}\n"
        report += f"Total documents in collection: {total_documents}\n"
        report += f"Search Results Retrieved: {top_k}\n"
        report += f"Top Results Used for Metrics: {top_a}\n\n"
        
        # Chunk-level metrics
        report += "CHUNK-LEVEL METRICS\n"
        report += "-------------------\n"
        report += f"Success Rate: {chunk_success_rate:.2%}\n"
        report += f"Average Position: {chunk_avg_position:.2f}\n"
        report += f"Average Precision: {chunk_avg_precision:.4f}\n"
        report += f"Average Recall: {chunk_avg_recall:.4f}\n"
        report += f"Average F1 Score: {chunk_avg_f1:.4f}\n"
        report += f"Mean Reciprocal Rank (MRR): {chunk_avg_mrr:.4f}\n\n"
        
        # Document-level metrics
        report += "DOCUMENT-LEVEL METRICS\n"
        report += "----------------------\n"
        report += f"Success Rate: {document_success_rate:.2%}\n"
        report += f"Average Position: {document_avg_position:.2f}\n"
        report += f"Average Precision: {document_avg_precision:.4f}\n"
        report += f"Average Recall: {document_avg_recall:.4f}\n"
        report += f"Average F1 Score: {document_avg_f1:.4f}\n"
        report += f"Mean Reciprocal Rank (MRR): {document_avg_mrr:.4f}\n\n"

        # Per-query metrics table (chunk level)
        report += "PER-QUERY METRICS (CHUNK LEVEL)\n"
        report += "-------------------------------\n"
        report += f"{'Query ID':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'MRR':<12}\n"
        report += "-" * 58 + "\n"

        for r in self.query_results:
            report += f"{r['query_id']:<10} {r['chunk_precision']:<12.4f} {r['chunk_recall']:<12.4f} {r['chunk_f1']:<12.4f} {r['chunk_mrr']:<12.4f}\n"
        report += "\n"
        
        # Per-query metrics table (document level)
        report += "PER-QUERY METRICS (DOCUMENT LEVEL)\n"
        report += "----------------------------------\n"
        report += f"{'Query ID':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'MRR':<12}\n"
        report += "-" * 58 + "\n"

        for r in self.query_results:
            report += f"{r['query_id']:<10} {r['document_precision']:<12.4f} {r['document_recall']:<12.4f} {r['document_f1']:<12.4f} {r['document_mrr']:<12.4f}\n"
        report += "\n"

        # Detailed retrieval analysis
        report += "DETAILED RETRIEVAL ANALYSIS\n"
        report += "--------------------------\n"

        for r in self.query_results:
            report += f"Query {r['query_id']}: {r['query']}\n"
            report += f"Expected chunk: {r['expected_chunk_id']}\n"
            report += f"Expected document: {r['expected_document_id']}\n"
            report += f"Retrieved chunks (showing all {top_k}, metrics calculated on top {top_a}):\n"
        
            for i, chunk_data in enumerate(r['retrieved_chunks_data']):
                chunk_id = chunk_data['chunk_id']
                document_id = chunk_data['document_id']
                
                # Check if this chunk is the expected chunk (correct retrieval)
                is_correct_chunk = chunk_id == r['expected_chunk_id']
                is_correct_document = document_id == r['expected_document_id']
                is_in_top_a = i < top_a
                
                # Create markers for correct retrieval and for top_a
                chunk_marker = "✓" if is_correct_chunk else "✗"
                doc_marker = "✓" if is_correct_document else "✗"
                top_a_marker = "*" if is_in_top_a else " "
            
                # Get score and text from the actual search results
                score = chunk_data['score']
                chunk_text = chunk_data['text']
            
                report += f"  {i+1}. {top_a_marker} [Chunk: {chunk_marker}] [Doc: {doc_marker}] {chunk_id} (Doc: {document_id}, Score: {score:.4f})\n"
                report += f"     Text: {chunk_text}\n"
        
            report += f"Chunk Precision: {r['chunk_precision']:.4f}, Recall: {r['chunk_recall']:.4f}, F1: {r['chunk_f1']:.4f}, MRR: {r['chunk_mrr']:.4f}\n"
            report += f"Doc Precision: {r['document_precision']:.4f}, Recall: {r['document_recall']:.4f}, F1: {r['document_f1']:.4f}, MRR: {r['document_mrr']:.4f}\n"
            report += "-" * 80 + "\n"

        return report

    def plot_results(self, save_path: Optional[str] = None, top_k: int = 10, top_a: int = 3) -> plt.Figure:
        """
        Create visualization of evaluation results for both chunk and document levels
        
        Args:
            save_path: Optional path to save the plot
            top_k: Number of results retrieved
            top_a: Number of top results used for metrics
            
        Returns:
            Matplotlib Figure
        """
        if not self.query_results:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No evaluation results available", ha='center', va='center')
            return fig
        
        # Create DataFrame from results
        df = pd.DataFrame([{
            'query_id': r.get('query_id', f"q{i+1}"),
            'chunk_precision': r['chunk_precision'],
            'chunk_recall': r['chunk_recall'],
            'chunk_f1': r['chunk_f1'],
            'chunk_mrr': r['chunk_mrr'],
            'is_chunk_found': r['is_chunk_found'],
            'chunk_position': r['chunk_position'] if r['chunk_position'] > 0 else np.nan,
            'document_precision': r['document_precision'],
            'document_recall': r['document_recall'],
            'document_f1': r['document_f1'],
            'document_mrr': r['document_mrr'],
            'is_document_found': r['is_document_found'],
            'document_position': r['document_position'] if r['document_position'] > 0 else np.nan
        } for i, r in enumerate(self.query_results)])
        
        # Plot metrics
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Success rate plot - Chunk Level
        df['chunk_result'] = df['is_chunk_found'].map({True: f'Found in top {top_a}', False: f'Not found in top {top_a}'})
        sns.countplot(x='chunk_result', data=df, ax=axes[0, 0])
        axes[0, 0].set_title(f'Chunk Success Rate (top {top_a} results)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xlabel('')
        
        # Add percentages to bars
        for p in axes[0, 0].patches:
            height = p.get_height()
            axes[0, 0].text(p.get_x() + p.get_width()/2.,
                        height + 0.5,
                        f'{height/len(df):.1%}',
                        ha="center")
        
        # Success rate plot - Document Level
        df['document_result'] = df['is_document_found'].map({True: f'Found in top {top_a}', False: f'Not found in top {top_a}'})
        sns.countplot(x='document_result', data=df, ax=axes[0, 1])
        axes[0, 1].set_title(f'Document Success Rate (top {top_a} results)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xlabel('')
        
        # Add percentages to bars
        for p in axes[0, 1].patches:
            height = p.get_height()
            axes[0, 1].text(p.get_x() + p.get_width()/2.,
                        height + 0.5,
                        f'{height/len(df):.1%}',
                        ha="center")
        
        # Metrics comparison - Chunk Level
        chunk_metrics = df[['chunk_precision', 'chunk_recall', 'chunk_f1', 'chunk_mrr']].melt()
        chunk_metrics['variable'] = chunk_metrics['variable'].str.replace('chunk_', '')
        sns.boxplot(x='variable', y='value', data=chunk_metrics, ax=axes[1, 0])
        axes[1, 0].set_title(f'Distribution of Chunk Metrics (top {top_a} results)')
        axes[1, 0].set_xlabel('')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_xticklabels(['Precision', 'Recall', 'F1', 'MRR'])
        
        # Metrics comparison - Document Level
        doc_metrics = df[['document_precision', 'document_recall', 'document_f1', 'document_mrr']].melt()
        doc_metrics['variable'] = doc_metrics['variable'].str.replace('document_', '')
        sns.boxplot(x='variable', y='value', data=doc_metrics, ax=axes[1, 1])
        axes[1, 1].set_title(f'Distribution of Document Metrics (top {top_a} results)')
        axes[1, 1].set_xlabel('')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticklabels(['Precision', 'Recall', 'F1', 'MRR'])
        
        # Position distribution for successful chunk queries
        chunk_successful = df[df['chunk_position'].notna()]
        if not chunk_successful.empty:
            sns.countplot(x='chunk_position', data=chunk_successful, ax=axes[2, 0])
            axes[2, 0].set_title('Chunk Position Distribution (When Found)')
            axes[2, 0].set_xlabel('Position')
            axes[2, 0].set_ylabel('Count')
            # Set x-axis limits to show only positions 1 to top_k
            axes[2, 0].set_xlim(-0.5, min(top_k + 0.5, chunk_successful['chunk_position'].max() + 0.5))
        else:
            axes[2, 0].text(0.5, 0.5, "No successful chunk queries to display", ha='center', va='center')
        
        # Position distribution for successful document queries
        doc_successful = df[df['document_position'].notna()]
        if not doc_successful.empty:
            sns.countplot(x='document_position', data=doc_successful, ax=axes[2, 1])
            axes[2, 1].set_title('Document Position Distribution (When Found)')
            axes[2, 1].set_xlabel('Position')
            axes[2, 1].set_ylabel('Count')
            # Set x-axis limits to show only positions 1 to top_k
            axes[2, 1].set_xlim(-0.5, min(top_k + 0.5, doc_successful['document_position'].max() + 0.5))
        else:
            axes[2, 1].text(0.5, 0.5, "No successful document queries to display", ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        return fig

async def main():
    """Main function to run the script"""
    logger.info("=== Starting Query Evaluation Process ===")
    
    # Define input and output parameters
    documents_folder = "Data/Output"  # Folder containing JSON documents
    output_base_name = "query_evaluation_results"  # Base name for output files
    limit = 50
    top_k = 50  # Number of results to retrieve for each query
    top_a = 15   # Number of top results to use for metric calculation
    
    # Initialize services
    qdrant_service = QdrantManagerService(collection_name="documents", vector_size=384)
    embedding_service = EmbeddingsService(model_name="all-MiniLM-L6-v2")

    search_service = SearchService(qdrant_service, embedding_service)
    
    # Create evaluator
    evaluator = QueryEvaluator(search_service)
    
    # Process all documents in the folder
    query_chunk_pairs = evaluator.extract_query_chunk_pairs_from_folder(documents_folder)
    
    if not query_chunk_pairs:
        logger.error("No query-chunk pairs found. Exiting.")
        return
    
    logger.info(f"Found {len(query_chunk_pairs)} query-chunk pairs for evaluation")
    
    # Evaluate queries
    results = await evaluator.evaluate_queries(query_chunk_pairs, limit=limit, top_k=top_k, top_a=top_a)
    
    # Generate and print report
    report = evaluator.generate_report(top_k=top_k, top_a=top_a)
    print(report)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    print("\nMetrics Summary:")
    
    # Print chunk-level metrics
    print("\nChunk-Level Metrics:")
    for key, value in metrics.items():
        if key.startswith('chunk_'):
            if key == 'chunk_success_rate':
                print(f"- {key.replace('chunk_', '')}: {value:.2%}")
            else:
                print(f"- {key.replace('chunk_', '')}: {value:.4f}")
    
    # Print document-level metrics
    print("\nDocument-Level Metrics:")
    for key, value in metrics.items():
        if key.startswith('document_'):
            if key == 'document_success_rate':
                print(f"- {key.replace('document_', '')}: {value:.2%}")
            else:
                print(f"- {key.replace('document_', '')}: {value:.4f}")
    
    # Create visualization
    fig = evaluator.plot_results(f"{output_base_name}.png", top_k=top_k, top_a=top_a)
    
    # Save report to text file
    with open(f"{output_base_name}.txt", "w", encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to {output_base_name}.txt")
    logger.info("=== Query Evaluation Process Completed ===")

if __name__ == "__main__":
    asyncio.run(main())