import logging
import json
from typing import List, Optional
from Application.InternalServices.LLMService import LLMService

logger = logging.getLogger(__name__)

class KeywordsService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using LLM"""
        try:
            prompt = f"""
            Wydobądź od 5 do 20 najważniejszych słów kluczowych lub fraz kluczowych z tego tekstu.  
            Tekst jest w języku polskim, a słowa kluczowe również powinny być w języku polskim.

            Zasady:
            - Wydobywaj **tylko konkretne terminy**, które mają jednoznaczne znaczenie, takie jak nazwy usług, produktów, organizacji, regulacji, wydarzeń, technologii, miejsc lub ważnych pojęć związanych z tekstem.
            - **NIE** uwzględniaj ogólnych słów, takich jak "tekst", "pojęcia", "słowa kluczowe".
            - **NIE** uwzględniaj terminów abstrakcyjnych ani emocjonalnych (np. "uczciwość", "jakość", "zaufanie").
            - **NIE** uwzględniaj czasowników ani ogólnych przymiotników.
            - Zwróć tylko **listę słów kluczowych oddzielonych przecinkami**, bez dodatkowego formatowania.

            Tekst do analizy:
            {text}

            Słowa kluczowe:
            """

            response = await self.llm_service.complete(prompt)
            
            keywords = [k.strip() for k in response.split(',')]
            return keywords

        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    async def process_file(self, input_path: str, output_path: str, chunk_limit: Optional[int] = None):
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            chunks_to_process = data['chunks'][:chunk_limit] if chunk_limit else data['chunks']
            chunks_with_keywords = []
            
            for i, chunk in enumerate(chunks_to_process, 1):
                chunk_id = chunk['chunk_id']
                logger.info(f"Processing chunk {i}/{len(chunks_to_process)} (ID: {chunk_id})")
                
                try:
                    keywords = await self.extract_keywords(chunk['text'])
                except Exception as e:
                    logger.error(f"Error extracting keywords for chunk {chunk_id}: {str(e)}")
                    keywords = []

                chunks_with_keywords.append({
                    "chunk_id": chunk_id,
                    "text": chunk['text'],
                    "keywords": keywords
                })
                
                logger.info(f"Processed chunk {chunk_id} with {len(keywords)} keywords")

            output_data = {
                "metadata": data['metadata'],
                "chunks": chunks_with_keywords
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Keywords extraction completed. Results saved to {output_path}")
            return output_data

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
