# llm.py
from openai import APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_NAME, TEMPERATURE
from utils import retry_async

client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    timeout=120.0,
)

_RETRY_DELAYS = [10.0, 30.0]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


async def _create(messages: list[dict], **kwargs) -> str:
    async def _call():
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,  # type: ignore
            temperature=TEMPERATURE,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    return await retry_async(
        _call,
        retry_delays=_RETRY_DELAYS,
        is_retryable_exception=_is_retryable,
        exhausted_exception=lambda exc, _: exc or RuntimeError("LLM retries exhausted"),
    )


async def chat(messages: list[dict]) -> str:
    return await _create(messages, response_format={"type": "json_object"})


async def complete(messages: list[dict]) -> str:
    return await _create(messages)
