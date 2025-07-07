import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Optional, Dict, Union
from Application.Domain.SearchRow import SearchRow
from Application.Domain.SearchResults import SearchResults

class RerankerService:
    """
    A class for reranking search results using a neural reranker model.
    """
    def __init__(self, model_name: str = "sdadas/polish-reranker-roberta-v2"):
        """
        Initialize the reranker with a specified model.
        
        Args:
            model_name: The HuggingFace model name to use for reranking
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="cuda" if torch.cuda.is_available() else "cpu"
            )
        self.device = next(self.model.parameters()).device

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax to convert scores to probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
        
    def rerank(self, query: str, results: SearchResults, use_probabilities: bool = False) -> SearchResults:
        if not results or not results.rows:
            return results
        
        try:
            texts = [row.text for row in results.rows]
            
            pairs = []
            for text in texts:
                pairs.append([query, text])
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                if use_probabilities:
                    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1]
                else:
                    scores = outputs.logits[:, 1]
                
                scores = scores.cpu().numpy()
            
            for row, score in zip(results.rows, scores):
                row.score = float(score)
            
            results.rows.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            return results
        
    def rerank_simple(self, query: str, search_results: SearchResults,
                      use_probabilities: bool = True) -> SearchResults:
        """
        Simple reranking function that takes SearchResults and returns SearchResults.
        Almost identical to rerank() but with 'simple' in the name for compatibility.
        
        Args:
            query: The search query
            search_results: The original search results to rerank
            use_probabilities: Whether to convert scores to probabilities
            
        Returns:
            A new SearchResults object with reranked results and updated scores
        """
        if not search_results.rows:
            return search_results
            
        # Extract answers from search rows
        answers = [row.text for row in search_results.rows]
        
        # Prepare inputs for the model
        texts = [f"{query}</s></s>{answer}" for answer in answers]
        tokens = self.tokenizer(
            texts, 
            padding="longest", 
            max_length=512, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            output = self.model(**tokens)
            
        # Process scores
        results = output.logits.detach().cpu().float().numpy()
        results = np.squeeze(results)
        
        # Apply softmax if requested
        if use_probabilities:
            scores = self.softmax(results).tolist()
        else:
            scores = results.tolist()
            
        # Create new search rows with updated scores
        new_rows = []
        for i, row in enumerate(search_results.rows):
            # Create a new SearchRow with the updated score
            new_row = SearchRow(
                text=row.text,
                area=row.area,
                score=float(scores[i]),
                chunk_id=row.chunk_id,
                links=row.links,
                metadata=row.metadata
            )
            new_rows.append(new_row)
            
        # Sort new rows by score in descending order
        new_rows.sort(key=lambda x: x.score, reverse=True)
        
        # Create and return a new SearchResults object
        return SearchResults(rows=new_rows, count=len(new_rows))
