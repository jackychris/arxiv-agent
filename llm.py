# llm.py
import asyncio

from openai import APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME, TEMPERATURE

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=120.0,
)

_RETRIES = 3
_RETRY_DELAY = 5.0


async def _create(messages: list[dict], **kwargs) -> str:
    for attempt in range(_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,  # type: ignore
                temperature=TEMPERATURE,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except RateLimitError:
            if attempt == _RETRIES - 1:
                raise
            await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
        except APITimeoutError:
            if attempt == _RETRIES - 1:
                raise
            await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < _RETRIES - 1:
                await asyncio.sleep(_RETRY_DELAY)
            else:
                raise
    raise RuntimeError("LLM retries exhausted without returning or raising")


async def chat(messages: list[dict]) -> str:
    return await _create(messages, response_format={"type": "json_object"})


async def complete(messages: list[dict]) -> str:
    return await _create(messages)
