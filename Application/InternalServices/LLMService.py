import logging
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, client):
        self.client = client

    @classmethod
    def create_openai(cls):
        load_dotenv()
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return cls(client)

    async def complete(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in LLM completion: {str(e)}")
            raise
