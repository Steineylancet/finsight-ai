"""
FinSight AI — shared model clients (two-model strategy).

mini  → gpt-5.4-mini: query understanding, parameter extraction, tool selection.
full  → GPT-4o: every piece of prose the user reads.

Calls tagged FINAL_ANSWER_TAG are the only ones whose tokens are streamed to the
user; routing/extraction calls stay invisible.
"""

import os
import warnings
from functools import lru_cache

from langchain_openai import AzureChatOpenAI

FINAL_ANSWER_TAG = "final_answer"

# langchain-openai's structured-output path trips a harmless pydantic serializer warning
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def _client(deployment_env: str, default: str, temperature: float) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "PLACEHOLDER"),
        azure_deployment=os.getenv(deployment_env, default),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "PLACEHOLDER"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=temperature,
        max_retries=2,
        timeout=60,
    )


@lru_cache(maxsize=1)
def mini_llm() -> AzureChatOpenAI:
    return _client("AZURE_OPENAI_MINI_DEPLOYMENT", "gpt-5.4-mini", temperature=0)


@lru_cache(maxsize=4)
def full_llm(temperature: float = 0.3) -> AzureChatOpenAI:
    return _client("AZURE_OPENAI_FULL_DEPLOYMENT", "gpt-4o", temperature=temperature)


def final_llm(temperature: float = 0.3):
    """Full model, tagged so its tokens stream to the user."""
    return full_llm(temperature).with_config(tags=[FINAL_ANSWER_TAG])
