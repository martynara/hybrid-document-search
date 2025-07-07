import logging
import json
import os
import asyncio
from typing import List, Optional
from datetime import datetime
from Application.InternalServices.LLMService import LLMService
import openai

logger = logging.getLogger(__name__)

class SummaryService:
    """Service for extracting summaries from text using LLM."""
    
    def __init__(self, llm_service=None):
        """
        Initialize Summary Service with LLM.
        
        Args:
            llm_service: LLMService to use for summary generation
        """
        self.llm_service = llm_service or LLMService.create_openai()
        self.logger = logging.getLogger(__name__)

    async def generate_summary(self, text: str) -> str:
        """
        Generate a summary for a text passage.
        
        Args:
            text: Text to summarize
            
        Returns:
            Generated summary
        """
        try:
            if not text or len(text.strip()) < 50:
                return "Text too short to summarize."
            
            # Generate summary using LLM
            prompt = f"""Please provide a short summary of the following text (maximum 2 sentences):

Text:
---
{text[:3000]}  # Limit text to first 3000 chars to avoid context limits
---

Summary:"""
            
            response = await self.llm_service.generate_text(prompt)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return "Failed to generate summary."
    
    def generate_summary_sync(self, text: str) -> str:
        """
        Generate a summary for a text passage (synchronous version).
        
        Args:
            text: Text to summarize
            
        Returns:
            Generated summary
        """
        try:
            if not text or len(text.strip()) < 50:
                return "Tekst zbyt krótki do podsumowania."
            
            # Direct synchronous OpenAI API call
            prompt = f"""Wygeneruj zwięzłe podsumowanie poniższego tekstu (3 do 10 zdań):

Tekst:
---
{text}  # Limit text to first 3000 chars to avoid context limits
---

Podsumowanie:"""
            
            # Use OpenAI's synchronous API directly
            openai_client = openai.OpenAI()
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Jesteś pomocnym asystentem, który tworzy zwięzłe podsumowania."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.5
            )
            
            # Extract and return the summary text
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in synchronous summary generation: {str(e)}")
            return "Nie udało się wygenerować podsumowania."

    async def process_file(self, input_path: str, output_path: str, chunk_limit: Optional[int] = None):
        """Przetwarza plik z chunkami i generuje streszczenia."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            chunks_to_process = data['chunks'][:chunk_limit] if chunk_limit else data['chunks']
            chunks_with_summaries = []
            
            for i, chunk in enumerate(chunks_to_process, 1):
                chunk_id = chunk['chunk_id']
                logger.info(f"Processing chunk {i}/{len(chunks_to_process)} (ID: {chunk_id})")
                
                try:
                    summary = await self.generate_summary(chunk['text'])
                except Exception as e:
                    logger.error(f"Error generating summary for chunk {chunk_id}: {str(e)}")
                    summary = ""

                chunks_with_summaries.append({
                    "chunk_id": chunk_id,
                    "text": chunk['text'],
                    "summary": summary
                })
                
                logger.info(f"Processed chunk {chunk_id}")

            # Przygotuj wyniki
            output_data = {
                "metadata": data['metadata'],
                "chunks": chunks_with_summaries
            }

            # Zapisz do pliku
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Summaries generation completed. Results saved to {output_path}")
            return output_data

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
