import logging
import json
import os
from typing import List, Optional
from datetime import datetime
from Application.InternalServices.LLMService import LLMService
import uuid

logger = logging.getLogger(__name__)

QA_PROMPT = """
Przeanalizuj poniższy tekst i utwórz 3-5 kluczowych pytań wraz z odpowiedziami dotyczących jego głównych tematów i zawartości:
{text}

Zwróć wynik wyłącznie w poniższym formacie JSON, bez dodatkowego tekstu:
[
    {
        "query": "pytanie_1",
        "answer": "odpowiedź_1"
    },
    {
        "query": "pytanie_2",
        "answer": "odpowiedź_2"
    }
]
"""

class QueryAnswerService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def _parse_query_answears(self, response: str) -> List[str]:
        """Parse LLM response into a list of q&a."""
        try:
            # Clean up the response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1]
            if response.startswith("```"):
                response = response.split("```")[1]
            if response.endswith("```"):
                response = response[:-3]

            response = response.strip()

            # Parse JSON
            qas = json.loads(response)
            if not isinstance(qas, list):
                logger.warning(f"Expected list but got {type(qas).__name__}")
                return []

            return qas

        except Exception as e:
            logger.error(f"Error parsing qas response: {str(e)}")
            logger.error(f"Raw response: {response}")
            return []

    async def process_file(self, input_path: str, output_path: str, chunk_limit: Optional[int] = None):
        """
        Process chunks file and generate keywords.
        
        Args:
            input_path: Path to input chunks JSON file
            output_path: Path to output keywords JSON file
            chunk_limit: Optional limit on number of chunks to process (for testing)
        """
        try:
            # Load input file
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Update metadata
            metadata = data['metadata']
            metadata.update({
                "qa_generated_at": datetime.now().isoformat()
            })

            # Get chunks to process
            chunks_to_process = data['chunks'][:chunk_limit] if chunk_limit else data['chunks']
            total_chunks = len(chunks_to_process)
            
            # Verify all chunks have chunk_ids
            self._ensure_chunk_ids(chunks_to_process, input_path)
            
            # Process selected chunks
            chunks_with_qa = []
            
            for i, chunk in enumerate(chunks_to_process, 1):
                print(chunk['text'])
                chunk_text = chunk['text']
                chunk_id = chunk['chunk_id']
                
                logger.info(f"Processing qa for chunk {i}/{total_chunks} (ID: {chunk_id})")
                
                # Get qa from LLM
                prompt = QA_PROMPT.replace('{text}', chunk_text)
                response = self.llm_service.complete(prompt)

                print(response)

                qa = self._parse_query_answears(response)

                # Add qa to chunk
                chunk_with_qa = {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    # "keywords": keywords,
                    "qa_pairs": qa

                }
                chunks_with_qa.append(chunk_with_qa)
                
                logger.info(f"Generated {len(qa)} qa_pairs from chunk {chunk_id}")

            # Prepare output data
            output_data = {
                "metadata": metadata,
                "chunks": chunks_with_qa
            }

            # Save output file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            logger.info(f"QA pairs generation completed. Results saved to {output_path}")
            logger.info(f"Processed {len(chunks_with_qa)} out of {len(data['chunks'])} total chunks")

        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise
            
    def _ensure_chunk_ids(self, chunks, input_path):
        """
        Ensures all chunks have valid chunk_ids, generating them if needed.
        
        Args:
            chunks: List of chunk objects
            input_path: Path to the original chunks file, for updating if needed
        """
        updated = False
        
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk or not chunk['chunk_id']:
                # Generate a new ID
                chunk['chunk_id'] = str(uuid.uuid4())
                logger.warning(f"Generated missing chunk_id for chunk {i}: {chunk['chunk_id']}")
                updated = True
                
        # If we updated any chunk IDs, save the changes back to the input file
        if updated:
            try:
                # Load the full file
                with open(input_path, 'r', encoding='utf-8') as f:
                    full_data = json.load(f)
                
                # Update only the chunks we've modified
                chunk_index_offset = 0
                for i, chunk in enumerate(chunks):
                    # Find the actual index in the full data
                    if i < len(full_data['chunks']):
                        # Check if this is the same chunk by comparing text content
                        if full_data['chunks'][i+chunk_index_offset].get('text') == chunk.get('text'):
                            full_data['chunks'][i+chunk_index_offset]['chunk_id'] = chunk['chunk_id']
                        else:
                            # The chunks don't align as expected, so search for matching text
                            found = False
                            for j, full_chunk in enumerate(full_data['chunks']):
                                if full_chunk.get('text') == chunk.get('text'):
                                    full_data['chunks'][j]['chunk_id'] = chunk['chunk_id']
                                    chunk_index_offset = j - i
                                    found = True
                                    break
                            
                            if not found:
                                logger.warning(f"Could not find matching chunk for ID {chunk.get('chunk_id')}")
                
                # Save the updated file
                with open(input_path, 'w', encoding='utf-8') as f:
                    json.dump(full_data, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"Updated {input_path} with generated chunk IDs")
                
            except Exception as e:
                logger.error(f"Error updating chunk IDs in original file: {str(e)}") 