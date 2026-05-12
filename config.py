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
EFFORT_STEPS: dict[str, int] = {
    "low": _int_env("EFFORT_STEPS_LOW", 5),
    "medium": _int_env("EFFORT_STEPS_MEDIUM", 12),
    "high": _int_env("EFFORT_STEPS_HIGH", 20),
}
TEMPERATURE = _float_env("TEMPERATURE", 0.7)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

_pg_user = os.getenv("POSTGRES_USER", "arxiv")
_pg_pass = os.getenv("POSTGRES_PASSWORD", "")
_pg_host = os.getenv("POSTGRES_HOST", "postgres")
_pg_db = os.getenv("POSTGRES_DB", "arxiv_agent")
DATABASE_URL = (
    os.getenv("DATABASE_URL") or f"postgresql://{_pg_user}:{_pg_pass}@{_pg_host}:5432/{_pg_db}"
)
PAPER_CONTENT_DIR = os.getenv("PAPER_CONTENT_DIR", "./paper_content")

MEMORY_MAX_ACTIVE = _int_env("MEMORY_MAX_ACTIVE", 8)
MEMORY_MAX_STORED = _int_env("MEMORY_MAX_STORED", 30)

SYNTHESIZER_MAP_THRESHOLD = _int_env("SYNTHESIZER_MAP_THRESHOLD", 40000)
SYNTHESIZER_MAP_OUTPUT_LIMIT = _int_env("SYNTHESIZER_MAP_OUTPUT_LIMIT", 4000)


def validate_required() -> None:
    """Raise at startup if required environment variables are missing."""
    missing = [name for name, val in [("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)] if not val]
    if missing:
        raise OSError(
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

ARXIV_MCP_HOST = os.getenv("ARXIV_MCP_HOST", "127.0.0.1")
GITHUB_MCP_HOST = os.getenv("GITHUB_MCP_HOST", "127.0.0.1")
WEB_MCP_HOST = os.getenv("WEB_MCP_HOST", "127.0.0.1")
RESEARCH_AGENT_HOST = os.getenv("RESEARCH_AGENT_HOST", "127.0.0.1")

ARXIV_MCP_URL = os.getenv("ARXIV_MCP_URL", f"http://{ARXIV_MCP_HOST}:{PORT_MCP}/mcp/")
GITHUB_MCP_URL = os.getenv(
    "GITHUB_MCP_URL", f"http://{GITHUB_MCP_HOST}:{PORT_GITHUB_MCP}/mcp/"
)
WEB_MCP_URL = os.getenv("WEB_MCP_URL", f"http://{WEB_MCP_HOST}:{PORT_WEB_MCP}/mcp/")
RESEARCH_AGENT_URL = os.getenv(
    "RESEARCH_AGENT_URL", f"http://{RESEARCH_AGENT_HOST}:{PORT_RESEARCH_AGENT}"
)

SERVER_START_TIMEOUT = _float_env("SERVER_START_TIMEOUT", 60.0)
SSE_HEARTBEAT_INTERVAL = _float_env("SSE_HEARTBEAT_INTERVAL", 10.0)
SSE_REVIEW_HEARTBEAT_INTERVAL = _float_env("SSE_REVIEW_HEARTBEAT_INTERVAL", 10.0)
SSE_DISCONNECT_GRACE = _float_env("SSE_DISCONNECT_GRACE", 5.0)
A2A_CLIENT_TIMEOUT = _float_env("A2A_CLIENT_TIMEOUT", 300.0)
MCP_TOOL_TIMEOUT = _float_env("MCP_TOOL_TIMEOUT", 90.0)
WEB_FETCH_TIMEOUT = _float_env("WEB_FETCH_TIMEOUT", 30.0)
ARXIV_FETCH_TIMEOUT = _float_env("ARXIV_FETCH_TIMEOUT", 30.0)
ARXIV_429_RETRY_DELAYS = parse_float_list(os.getenv("ARXIV_429_RETRY_DELAYS"), [10.0])
GITHUB_HTTP_TIMEOUT = _float_env("GITHUB_HTTP_TIMEOUT", 15.0)

DISABLE_DB = _bool_env("DISABLE_DB", False)
ENABLE_RUN_CONTEXT = _bool_env("ENABLE_RUN_CONTEXT", True)
MAX_CRITIC_ROUNDS = _int_env("MAX_CRITIC_ROUNDS", 1)
CRITIC_MAX_MISSIONS = _int_env("CRITIC_MAX_MISSIONS", 2)
ENABLE_TRACE = _bool_env("ENABLE_TRACE", True)
TRACE_DIR = os.getenv("TRACE_DIR", "runs")
