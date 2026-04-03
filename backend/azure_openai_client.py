"""
Azure OpenAI Client Wrapper
Handles embeddings and chat completions for FinSight AI
"""

import os
import time
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AzureOpenAIClient:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

    def get_embedding(self, text: str, retries: int = 3) -> list[float]:
        """Generate embedding vector for a given text. Retries on rate limit."""
        text = text.replace("\n", " ").strip()
        for attempt in range(retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_deployment
                )
                return response.data[0].embedding
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit hit. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"Embedding failed: {e}")
                    raise

    def get_embeddings_batch(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Generate embeddings for a list of texts in batches (one API call per batch)."""
        all_embeddings = []
        total_batches = (len(texts) - 1) // batch_size + 1
        for i in range(0, len(texts), batch_size):
            batch = [t.replace("\n", " ").strip() for t in texts[i: i + batch_size]]
            logger.info(f"Embedding batch {i // batch_size + 1} / {total_batches}")
            for attempt in range(3):
                try:
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.embedding_deployment
                    )
                    batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
                    all_embeddings.extend(batch_embeddings)
                    time.sleep(0.1)
                    break
                except Exception as e:
                    if "rate_limit" in str(e).lower() and attempt < 2:
                        wait = 2 ** attempt * 10
                        logger.warning(f"Rate limit hit. Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"Batch embedding failed: {e}")
                        raise
        return all_embeddings

    def chat_completion(
        self,
        messages: list[dict],
        stream: bool = True,
        max_tokens: int = 800,
        temperature: float = 0.3,
    ):
        """
        Call GPT-4o with the given messages.
        If stream=True, returns a generator yielding text chunks.
        If stream=False, returns the full response string.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if stream:
                return response  # Caller iterates over chunks
            else:
                return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
