from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    retryable_codes: frozenset[str] = frozenset({
        "MCP_TOOL_ERROR",
        "MCP_TOOL_TIMEOUT",
        "TOOL_ERROR",
        "TOOL_TIMEOUT",
    })


DEFAULT_TOOL_RETRY_POLICY = RetryPolicy()


def error_code(result: dict) -> str | None:
    error = result.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return str(code) if code else None


def should_retry(result: dict, attempt: int, policy: RetryPolicy = DEFAULT_TOOL_RETRY_POLICY) -> bool:
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
