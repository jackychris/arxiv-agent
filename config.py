# config.py
import os
from dotenv import load_dotenv

from utils import parse_float_list

load_dotenv()


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-v4-flash")
MAX_STEPS = _int_env("MAX_STEPS", 12)
TEMPERATURE = _float_env("TEMPERATURE", 0.7)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def validate_required() -> None:
    """Raise at startup if required environment variables are missing."""
    missing = [name for name, val in [("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)] if not val]
    if missing:
        raise EnvironmentError(
            f"Required environment variable(s) not set: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in the values."
        )

PORT_MAIN = _int_env("PORT_MAIN", 8000)
PORT_MCP = _int_env("PORT_MCP", 8765)
PORT_GITHUB_MCP = _int_env("PORT_GITHUB_MCP", 8767)
PORT_WEB_MCP = _int_env("PORT_WEB_MCP", 8768)
PORT_RESEARCH_AGENT = _int_env("PORT_RESEARCH_AGENT", 8766)
PORT_DISCOVERY_AGENT = _int_env("PORT_DISCOVERY_AGENT", 8769)
PORT_ANALYSIS_AGENT = _int_env("PORT_ANALYSIS_AGENT", 8770)
PORT_SYNTHESIS_AGENT = _int_env("PORT_SYNTHESIS_AGENT", 8771)

SERVER_START_TIMEOUT = _float_env("SERVER_START_TIMEOUT", 60.0)
SSE_HEARTBEAT_INTERVAL = _float_env("SSE_HEARTBEAT_INTERVAL", 20.0)
A2A_CLIENT_TIMEOUT = _float_env("A2A_CLIENT_TIMEOUT", 300.0)
MCP_TOOL_TIMEOUT = _float_env("MCP_TOOL_TIMEOUT", 90.0)
WEB_FETCH_TIMEOUT = _float_env("WEB_FETCH_TIMEOUT", 30.0)
ARXIV_FETCH_TIMEOUT = _float_env("ARXIV_FETCH_TIMEOUT", 30.0)
ARXIV_429_RETRY_DELAYS = parse_float_list(os.getenv("ARXIV_429_RETRY_DELAYS"), [30.0, 60.0, 120.0])
GITHUB_HTTP_TIMEOUT = _float_env("GITHUB_HTTP_TIMEOUT", 15.0)

ENABLE_RUN_CONTEXT = _bool_env("ENABLE_RUN_CONTEXT", True)
RUN_CONTEXT_ENTRY_MAX_CHARS = _int_env("RUN_CONTEXT_ENTRY_MAX_CHARS", 500)
ENABLE_TRACE = _bool_env("ENABLE_TRACE", True)
TRACE_DIR = os.getenv("TRACE_DIR", "runs")
