from dataclasses import dataclass

from config import MCP_TOOL_RETRY_ATTEMPTS, MCP_TOOL_RETRY_BACKOFF_SECONDS


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = MCP_TOOL_RETRY_ATTEMPTS
    backoff_seconds: float = MCP_TOOL_RETRY_BACKOFF_SECONDS
    retryable_codes: frozenset[str] = frozenset(
        {
            "ARXIV_RATE_LIMIT",
            "MCP_TOOL_ERROR",
            "MCP_TOOL_TIMEOUT",
            "RATE_LIMIT",
            "TOOL_ERROR",
            "TOOL_TIMEOUT",
            "TOO_MANY_REQUESTS",
        }
    )


DEFAULT_TOOL_RETRY_POLICY = RetryPolicy()


def error_code(result: dict) -> str | None:
    error = result.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return str(code) if code else None


def should_retry(
    result: dict, attempt: int, policy: RetryPolicy = DEFAULT_TOOL_RETRY_POLICY
) -> bool:
    if result.get("ok") is True:
        return False
    code = error_code(result)
    return (
        code in policy.retryable_codes
        and attempt < policy.max_attempts
        and bool((result.get("error") or {}).get("recoverable", True))
    )


def backoff_seconds(attempt: int, policy: RetryPolicy = DEFAULT_TOOL_RETRY_POLICY) -> float:
    return policy.backoff_seconds * attempt
