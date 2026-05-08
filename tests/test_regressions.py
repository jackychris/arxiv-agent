# tests/test_regressions.py
# Lightweight regression tests for bugs that should stay fixed.
import asyncio
import json
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ENABLE_RUN_CONTEXT, _bool_env
from mcp_client import MCPClient, _infer_error_code, wrap_tool_output
from memory import long_term as lt
from memory.short_term import ShortTermMemory
from orchestrator import OrchestratorAgent
from orchestrator import events as evt
from prompts import TOOL_REFLECT_PROMPT, build_memory_hint
from researcher import events as research_events
from researcher.reflection import build_reflection_history
from runtime import run_context, trace
from runtime.context import RunContext
from runtime.policy import RetryPolicy, should_retry
from runtime.storage import InMemoryRunContextStore, JsonlTraceStore, compact_run_context_content
from schemas import MemoryEntry, SSEEvent, ToolResultEnvelope, TraceEvent
from tools.arxiv import _rate_limit as arxiv_rate_limit
from tools.web.web_tools import _html_to_text, fetch_url
from utils import (
    AsyncRateLimiter,
    SyncRateLimiter,
    parse_float_list,
    rate_limit_wait,
    retry_schedule,
    retry_sync,
)


class EmptyStreamClient:
    def __init__(self):
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def close(self):
        self.closed = True

    async def send_message(self, _req):
        if False:
            yield None


class FakeFactory:
    def __init__(self, client):
        self.client = client

    async def create_from_url(self, _url):
        return self.client


class FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeToolResult:
    def __init__(self, text, *, is_error=False):
        self.content = [FakeTextBlock(text)]
        self.isError = is_error


class FakeToolSession:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    async def call_tool(self, _name, _kwargs):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


async def async_noop_sleep(_seconds):
    return None


async def test_fetch_url_blocks_arxiv():
    print("\n--- test_fetch_url_blocks_arxiv ---")
    result = await fetch_url("https://arxiv.org/abs/2106.09685")
    assert "error" in result, result
    assert "summarize_paper" in result["error"], result
    assert "2106.09685" in result["error"], result
    print(f"  {result['error']}")
    print("  PASSED")


async def test_fetch_url_blocks_arxiv_pdf():
    print("\n--- test_fetch_url_blocks_arxiv_pdf ---")
    result = await fetch_url("https://arxiv.org/pdf/2106.09685.pdf")
    assert "error" in result, result
    assert 'paper_id": "2106.09685"' in result["error"], result
    print(f"  {result['error']}")
    print("  PASSED")


async def test_fetch_html_to_text_skips_scripts():
    print("\n--- test_fetch_html_to_text_skips_scripts ---")
    text = _html_to_text(
        "<html><script>bad()</script><h1>Title</h1><p>Hello <b>world</b>.</p></html>"
    )
    assert "bad" not in text
    assert "Title" in text
    assert "Hello" in text
    assert "world" in text
    print("  HTML text extraction works")
    print("  PASSED")


async def test_run_task_always_emits_task_result():
    print("\n--- test_run_task_always_emits_task_result ---")
    queue = asyncio.Queue()
    client = EmptyStreamClient()
    old_store = trace.get_store()
    with tempfile.TemporaryDirectory() as tmp:
        trace.set_store(JsonlTraceStore(tmp))
        try:
            result = await OrchestratorAgent()._run_task(
                FakeFactory(client),
                mission="test mission",
                context_id="ctx",
                task_label="1",
                queue=queue,
            )
        finally:
            trace.set_store(old_store)

    first = await queue.get()
    second = await queue.get()

    assert result == "[no result]"
    assert first["type"] == "task_result"
    assert first["task_id"] == "1"
    assert first["content"] == "[no result]"
    assert first["error"]["code"] == "NO_RESULT"
    assert first["run_id"] == "ctx"
    assert second["type"] == "task_done"
    assert second["task_id"] == "1"
    assert second["run_id"] == "ctx"
    print("  emitted task_result before task_done")
    print("  PASSED")


async def test_run_task_does_not_close_shared_client():
    print("\n--- test_run_task_does_not_close_shared_client ---")
    queue = asyncio.Queue()
    client = EmptyStreamClient()
    old_store = trace.get_store()
    with tempfile.TemporaryDirectory() as tmp:
        trace.set_store(JsonlTraceStore(tmp))
        try:
            await OrchestratorAgent()._run_task(
                FakeFactory(client),
                mission="test mission",
                context_id="ctx",
                task_label="1",
                queue=queue,
            )
        finally:
            trace.set_store(old_store)

    assert client.closed is False, "orchestrator closed a client it does not own"
    print("  client remained open for the shared httpx owner")
    print("  PASSED")


async def test_sse_event_envelope():
    print("\n--- test_sse_event_envelope ---")
    payload = evt.final("done", run_id="run-1")
    assert payload["type"] == "final"
    assert payload["content"] == "done"
    assert payload["run_id"] == "run-1"
    assert payload["data"] == {}
    assert payload["error"] is None
    assert "ts" in payload
    assert evt.encode_sse(payload).startswith("data: ")
    print("  event envelope is stable")
    print("  PASSED")


async def test_sse_schema_rejects_unknown_event_type():
    print("\n--- test_sse_schema_rejects_unknown_event_type ---")
    try:
        SSEEvent(type="unknown_event")
    except Exception:
        pass
    else:
        raise AssertionError("SSEEvent should reject unknown event types")
    print("  SSE schema rejects unknown event types")
    print("  PASSED")


async def test_json_status_text_extracts_step_data():
    print("\n--- test_json_status_text_extracts_step_data ---")
    status = research_events.step_status(
        3,
        "Search arxiv with structured status",
        "search_arxiv",
        {"query": "LoRA", "max_results": 5},
    )
    payload = evt.status_text("1", status, run_id="run-1")
    assert payload["type"] == "step"
    assert payload["content"] == "Step 3: search_arxiv"
    assert payload["data"]["step"] == 3
    assert payload["data"]["tool"] == "search_arxiv"
    assert payload["data"]["tool_input"] == {"query": "LoRA", "max_results": 5}
    print("  JSON A2A step status is converted into structured SSE data")
    print("  PASSED")


async def test_json_status_text_extracts_observation_data():
    print("\n--- test_json_status_text_extracts_observation_data ---")
    status = research_events.observation_status("abcdef", step=3, tool="search_arxiv", max_chars=3)
    payload = evt.status_text("1", status, run_id="run-1")
    assert payload["type"] == "observation"
    assert payload["content"] == "abc..."
    assert payload["data"]["step"] == 3
    assert payload["data"]["tool"] == "search_arxiv"
    assert payload["data"]["preview"] == "abc..."
    assert payload["data"]["truncated"] is True
    print("  JSON A2A observation status is converted into structured SSE data")
    print("  PASSED")


async def test_status_text_rejects_non_json():
    print("\n--- test_status_text_rejects_non_json ---")
    try:
        evt.status_text("1", "[Step 1] old format", run_id="run-1")
    except ValueError as e:
        assert "must be JSON" in str(e)
    else:
        raise AssertionError("legacy A2A status text should be rejected")
    print("  non-JSON A2A status is rejected")
    print("  PASSED")


async def test_run_context_config_defaults_enabled():
    print("\n--- test_run_context_config_defaults_enabled ---")
    assert ENABLE_RUN_CONTEXT is True
    assert _bool_env("MISSING_BOOL_FOR_TEST", True) is True
    assert _bool_env("MISSING_BOOL_FOR_TEST", False) is False
    print("  ENABLE_RUN_CONTEXT defaults to enabled")
    print("  PASSED")


async def test_run_context_store_shares_other_task_entries():
    print("\n--- test_run_context_store_shares_other_task_entries ---")
    ctx = RunContext(InMemoryRunContextStore())
    ctx.publish(
        "run-1",
        "task-1",
        1,
        '{"ok": true, "tool": "web_search", "data": [{"title": "task one finding"}], "error": null, "meta": {}}',
    )
    ctx.publish(
        "run-1",
        "task-2",
        2,
        '{"ok": true, "tool": "web_search", "data": [{"title": "task two finding"}], "error": null, "meta": {}}',
    )
    others = ctx.get_others("run-1", "task-1")
    assert "task two finding" in others
    assert "task one finding" not in others
    assert "tool=web_search" in others
    ctx.clear("run-1")
    assert ctx.get_others("run-1", "task-1") == ""
    print("  run context store preserves existing cross-task semantics")
    print("  PASSED")


async def test_global_run_context_store_can_be_replaced():
    print("\n--- test_global_run_context_store_can_be_replaced ---")
    old_store = run_context.get_store()
    try:
        run_context.set_store(InMemoryRunContextStore())
        run_context.publish("run-2", "task-1", 1, "finding")
        assert "finding" in run_context.get_others("run-2", "task-2")
    finally:
        run_context.set_store(old_store)
    print("  global run context accepts pluggable stores")
    print("  PASSED")


async def test_run_context_compacts_tool_error_envelope():
    print("\n--- test_run_context_compacts_tool_error_envelope ---")
    summary = compact_run_context_content(
        '{"ok": false, "tool": "fetch_url", "data": null, "error": {"code": "TOOL_TIMEOUT", "message": "fetch_url failed after 30s"}, "meta": {}}'
    )
    assert (
        summary
        == "tool=fetch_url ok=false error_code=TOOL_TIMEOUT error=fetch_url failed after 30s"
    )
    print("  run context compacts tool error envelopes")
    print("  PASSED")


async def test_run_context_compacts_and_limits_large_success_envelope():
    print("\n--- test_run_context_compacts_and_limits_large_success_envelope ---")
    content = "A" * 1000
    summary = compact_run_context_content(
        json.dumps(
            {
                "ok": True,
                "tool": "fetch_url",
                "data": {"content": content},
                "error": None,
                "meta": {},
            }
        ),
        max_chars=80,
    )
    assert summary.startswith("tool=fetch_url ok=true preview=")
    assert len(summary) <= 80
    assert summary.endswith("...")
    print("  run context limits large success previews")
    print("  PASSED")


async def test_mcp_tool_success_envelope():
    print("\n--- test_mcp_tool_success_envelope ---")
    wrapped = wrap_tool_output("search_arxiv", '[{"title": "RAG"}]')
    assert wrapped["ok"] is True
    assert wrapped["tool"] == "search_arxiv"
    assert wrapped["data"] == [{"title": "RAG"}]
    assert wrapped["error"] is None
    json.dumps(wrapped)
    print("  successful MCP output is wrapped")
    print("  PASSED")


async def test_mcp_tool_error_envelope():
    print("\n--- test_mcp_tool_error_envelope ---")
    wrapped = wrap_tool_output(
        "fetch_url",
        '{"url": "https://arxiv.org/abs/2106.09685", "error": "fetch_url does not support arXiv URLs. Use summarize_paper instead."}',
    )
    assert wrapped["ok"] is False
    assert wrapped["tool"] == "fetch_url"
    assert wrapped["data"] is None
    assert wrapped["error"]["code"] == "ARXIV_URL_NOT_ALLOWED"
    assert wrapped["error"]["recoverable"] is True
    assert wrapped["error"]["details"]["url"] == "https://arxiv.org/abs/2106.09685"
    print("  error MCP output is wrapped")
    print("  PASSED")


async def test_tool_result_schema_requires_error_for_failure():
    print("\n--- test_tool_result_schema_requires_error_for_failure ---")
    try:
        ToolResultEnvelope(ok=False, tool="fetch_url")
    except Exception:
        pass
    else:
        raise AssertionError("ToolResultEnvelope should require error when ok=false")
    success = ToolResultEnvelope(ok=True, tool="web_search", data=[])
    assert success.to_dict()["error"] is None
    print("  tool result schema enforces ok/error consistency")
    print("  PASSED")


async def test_retry_policy_retries_retryable_tool_errors():
    print("\n--- test_retry_policy_retries_retryable_tool_errors ---")
    client = MCPClient()
    session = FakeToolSession(
        [
            FakeToolResult('{"error": "fetch_url timed out"}'),
            FakeToolResult('{"content": "ok"}'),
        ]
    )
    client._tool_to_session["fetch_url"] = session
    result = json.loads(
        await client.execute_tool(
            "fetch_url",
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0),
            url="https://example.com",
        )
    )
    assert session.calls == 2
    assert result["ok"] is True
    assert result["meta"]["attempt"] == 2
    print("  retryable tool error retried and succeeded")
    print("  PASSED")


async def test_retry_policy_does_not_retry_non_retryable_tool_errors():
    print("\n--- test_retry_policy_does_not_retry_non_retryable_tool_errors ---")
    client = MCPClient()
    session = FakeToolSession(
        [
            FakeToolResult(
                '{"error": "fetch_url does not support arXiv URLs. Use summarize_paper instead."}'
            ),
            FakeToolResult('{"content": "should not happen"}'),
        ]
    )
    client._tool_to_session["fetch_url"] = session
    result = json.loads(
        await client.execute_tool(
            "fetch_url",
            retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0),
            url="https://arxiv.org/abs/2106.09685",
        )
    )
    assert session.calls == 1
    assert result["ok"] is False
    assert result["error"]["code"] == "ARXIV_URL_NOT_ALLOWED"
    assert result["meta"]["attempt"] == 1
    print("  non-retryable tool error returned immediately")
    print("  PASSED")


async def test_should_retry_respects_recoverable_flag():
    print("\n--- test_should_retry_respects_recoverable_flag ---")
    result = {
        "ok": False,
        "error": {"code": "TOOL_TIMEOUT", "recoverable": False},
    }
    assert should_retry(result, 1, RetryPolicy(max_attempts=3, backoff_seconds=0)) is False
    print("  retry policy respects recoverable=false")
    print("  PASSED")


async def test_arxiv_rate_limit_retries_429_then_succeeds():
    print("\n--- test_arxiv_rate_limit_retries_429_then_succeeds ---")
    old_delays = arxiv_rate_limit.ARXIV_429_RETRY_DELAYS
    old_sleep = arxiv_rate_limit._sleep
    calls = {"count": 0}
    arxiv_rate_limit.ARXIV_429_RETRY_DELAYS = [0.01]
    arxiv_rate_limit._sleep = async_noop_sleep
    try:

        def flaky():
            calls["count"] += 1
            if calls["count"] == 1:
                raise urllib.error.HTTPError("url", 429, "Too Many Requests", None, None)
            return "ok"

        assert await arxiv_rate_limit.arxiv_api_call(flaky) == "ok"
        assert calls["count"] == 2
    finally:
        arxiv_rate_limit.ARXIV_429_RETRY_DELAYS = old_delays
        arxiv_rate_limit._sleep = old_sleep
    print("  arxiv provider layer retries 429 before returning")
    print("  PASSED")


async def test_arxiv_rate_limit_exhaustion_maps_to_error_code():
    print("\n--- test_arxiv_rate_limit_exhaustion_maps_to_error_code ---")
    old_delays = arxiv_rate_limit.ARXIV_429_RETRY_DELAYS
    old_sleep = arxiv_rate_limit._sleep
    arxiv_rate_limit.ARXIV_429_RETRY_DELAYS = [0.01]
    arxiv_rate_limit._sleep = async_noop_sleep
    try:

        def always_429():
            raise urllib.error.HTTPError("url", 429, "Too Many Requests", None, None)

        try:
            await arxiv_rate_limit.arxiv_api_call(always_429)
        except arxiv_rate_limit.ArxivRateLimitError as e:
            assert "ARXIV_RATE_LIMIT" in str(e)
            assert _infer_error_code("search_arxiv", str(e)) == "ARXIV_RATE_LIMIT"
        else:
            raise AssertionError("arxiv_api_call should raise after 429 retries are exhausted")
    finally:
        arxiv_rate_limit.ARXIV_429_RETRY_DELAYS = old_delays
        arxiv_rate_limit._sleep = old_sleep
    print("  exhausted arxiv 429 retries map to ARXIV_RATE_LIMIT")
    print("  PASSED")


async def test_rate_limit_utils_build_shared_schedules():
    print("\n--- test_rate_limit_utils_build_shared_schedules ---")
    assert parse_float_list("1, 2.5,3", []) == [1.0, 2.5, 3.0]
    assert parse_float_list(None, [30.0]) == [30.0]
    assert retry_schedule([30.0, 60.0]) == [0.0, 30.0, 60.0]
    assert retry_schedule([30.0], initial_immediate=False) == [30.0]
    assert rate_limit_wait(elapsed=1.5, min_interval=3.0) == 1.5
    assert rate_limit_wait(elapsed=4.0, min_interval=3.0) == 0.0
    print("  shared rate-limit utilities normalize delays")
    print("  PASSED")


async def test_sync_rate_limiter_and_retry_sync_are_generic():
    print("\n--- test_sync_rate_limiter_and_retry_sync_are_generic ---")
    now = {"value": 0.0}
    sleeps = []

    def clock():
        return now["value"]

    def sleep(seconds):
        sleeps.append(seconds)
        now["value"] += seconds

    limiter = SyncRateLimiter(3.0, clock=clock, sleep=sleep)
    assert limiter.call(lambda: "first") == "first"
    now["value"] += 1.0
    assert limiter.call(lambda: "second") == "second"
    assert sleeps == [2.0]

    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("retry me")
        return "ok"

    assert (
        retry_sync(
            flaky,
            retry_delays=[5.0],
            is_retryable_exception=lambda exc: isinstance(exc, ValueError),
            exhausted_exception=lambda exc, attempts: RuntimeError(f"exhausted {attempts}: {exc}"),
            sleep=sleep,
        )
        == "ok"
    )
    assert 5.0 in sleeps
    print("  sync limiter and retry helper are reusable")
    print("  PASSED")


async def test_async_rate_limiter_is_generic():
    print("\n--- test_async_rate_limiter_is_generic ---")
    now = {"value": 0.0}
    sleeps = []

    def clock():
        return now["value"]

    async def sleep(seconds):
        sleeps.append(seconds)
        now["value"] += seconds

    limiter = AsyncRateLimiter(2.0, clock=clock, sleep=sleep)

    async def value(text):
        return text

    assert await limiter.call_awaitable(value("first")) == "first"
    now["value"] += 0.5
    assert await limiter.call_awaitable(value("second")) == "second"
    assert sleeps == [1.5]
    print("  async limiter is reusable")
    print("  PASSED")


async def test_long_term_memory_filters_stale_fetch_url_lessons():
    print("\n--- test_long_term_memory_filters_stale_fetch_url_lessons ---")
    memories = lt.get_all()
    fetch_lessons = "\n".join(entry["lesson"] for entry in memories.get("fetch_url", [])).lower()
    summarize_lessons = "\n".join(
        entry["lesson"] for entry in memories.get("summarize_paper", [])
    ).lower()
    assert "arxiv" not in fetch_lessons
    assert "fetch the abstract directly via fetch_url" not in summarize_lessons
    assert "fetch the abstract page directly" not in summarize_lessons
    print("  stale fetch_url/arXiv lessons are hidden from prompts")
    print("  PASSED")


async def test_long_term_memory_structured_entries_are_deduped_and_capped():
    print("\n--- test_long_term_memory_structured_entries_are_deduped_and_capped ---")
    old_path = lt._PATH
    with tempfile.TemporaryDirectory() as tmp:
        lt._PATH = str(Path(tmp) / "long_term.json")
        try:
            await lt.add(
                "web_search",
                {"lesson": "Use targeted query terms.", "outcome": "success", "tags": ["query"]},
            )
            await lt.add(
                "web_search",
                {"lesson": "Use targeted query terms.", "outcome": "success", "tags": ["search"]},
            )
            for i in range(10):
                await lt.add(
                    "web_search",
                    {"lesson": f"Specific web lesson {i}", "outcome": "mixed", "tags": [f"t{i}"]},
                )
            entries = lt.get("web_search")
            hint = build_memory_hint({"web_search": entries})
            assert len(entries) == lt._MAX_ACTIVE_PER_TOOL
            assert all(isinstance(entry, dict) for entry in entries)
            assert any(
                entry["lesson"] == "Use targeted query terms." and entry["seen"] == 2
                for entry in entries
            )
            assert "outcome=" in hint
            assert "seen=" in hint
        finally:
            lt._PATH = old_path
    print("  structured memory dedupes repeated lessons and caps prompt volume")
    print("  PASSED")


async def test_memory_entry_schema_compacts_tags():
    print("\n--- test_memory_entry_schema_compacts_tags ---")
    entry = MemoryEntry(
        tool="web_search", lesson="Use targeted queries.", tags=["query", "query", "  web  "]
    )
    assert entry.to_dict()["tags"] == ["query", "web"]
    print("  memory schema compacts repeated tags")
    print("  PASSED")


async def test_trace_event_schema_and_jsonl_writer():
    print("\n--- test_trace_event_schema_and_jsonl_writer ---")
    old_store = trace.get_store()
    with tempfile.TemporaryDirectory() as tmp:
        trace.set_store(JsonlTraceStore(tmp))
        try:
            event = trace.record(
                "tool_result",
                run_id="run/test",
                task_id="1",
                step=2,
                tool="fetch_url",
                tool_call_id="call-1",
                latency_ms=123,
                ok=False,
                error_code="TOOL_TIMEOUT",
            )
            path = trace.trace_path("run/test")
            lines = path.read_text().splitlines()
            payload = json.loads(lines[0])
            assert payload == event
            assert payload["kind"] == "tool_result"
            assert payload["latency_ms"] == 123
            assert "/" not in path.name
        finally:
            trace.set_store(old_store)
    print("  trace writer stores structured JSONL events")
    print("  PASSED")


async def test_trace_writer_uses_lock_file():
    print("\n--- test_trace_writer_uses_lock_file ---")
    old_store = trace.get_store()
    with tempfile.TemporaryDirectory() as tmp:
        trace.set_store(JsonlTraceStore(tmp))
        try:
            trace.record("run_start", run_id="locked-run", content="test")
            assert trace.trace_path("locked-run").exists()
            assert trace.trace_path("locked-run").with_suffix(".jsonl.lock").exists()
        finally:
            trace.set_store(old_store)
    print("  trace writer creates per-run lock file")
    print("  PASSED")


async def test_trace_schema_rejects_unknown_kind():
    print("\n--- test_trace_schema_rejects_unknown_kind ---")
    try:
        TraceEvent(kind="unknown", run_id="run-1")
    except Exception:
        pass
    else:
        raise AssertionError("TraceEvent should reject unknown kinds")
    print("  trace schema rejects unknown event kinds")
    print("  PASSED")


async def test_trace_summarizes_tool_result():
    print("\n--- test_trace_summarizes_tool_result ---")
    ok, error_code = trace.summarize_tool_result(
        '{"ok": false, "tool": "fetch_url", "data": null, "error": {"code": "TOOL_TIMEOUT"}, "meta": {}}'
    )
    assert ok is False
    assert error_code == "TOOL_TIMEOUT"
    print("  trace extracts ok/error_code from tool envelopes")
    print("  PASSED")


async def test_tool_reflect_prompt_formats_structured_schema():
    print("\n--- test_tool_reflect_prompt_formats_structured_schema ---")
    prompt = TOOL_REFLECT_PROMPT.format(
        mission="test", history="OBSERVATION: tool=fetch_url ok=false error_code=TOOL_TIMEOUT"
    )
    assert '"lesson": "specific durable lesson"' in prompt
    assert "Current hard rule: fetch_url is only for non-arXiv web pages" in prompt
    print("  reflection prompt exposes structured schema")
    print("  PASSED")


async def test_reflection_history_includes_tool_envelope_outcomes():
    print("\n--- test_reflection_history_includes_tool_envelope_outcomes ---")
    memory = ShortTermMemory()
    memory.add("user", "Find LoRA papers")
    memory.add(
        "assistant",
        '{"thought": "fetch paper", "action": "fetch_url", "action_input": {"url": "https://arxiv.org/abs/2106.09685"}}',
    )
    memory.add(
        "user",
        'Observation: {"ok": false, "tool": "fetch_url", "data": null, "error": {"code": "ARXIV_URL_NOT_ALLOWED", "message": "Use summarize_paper instead."}, "meta": {}}',
    )
    history = build_reflection_history(memory)
    assert "tool=fetch_url" in history
    assert "ok=false" in history
    assert "error_code=ARXIV_URL_NOT_ALLOWED" in history
    print("  reflection sees structured tool failures")
    print("  PASSED")


async def main():
    tests = [
        test_fetch_url_blocks_arxiv,
        test_fetch_url_blocks_arxiv_pdf,
        test_fetch_html_to_text_skips_scripts,
        test_run_task_always_emits_task_result,
        test_run_task_does_not_close_shared_client,
        test_sse_event_envelope,
        test_sse_schema_rejects_unknown_event_type,
        test_json_status_text_extracts_step_data,
        test_json_status_text_extracts_observation_data,
        test_status_text_rejects_non_json,
        test_run_context_config_defaults_enabled,
        test_run_context_store_shares_other_task_entries,
        test_global_run_context_store_can_be_replaced,
        test_run_context_compacts_tool_error_envelope,
        test_run_context_compacts_and_limits_large_success_envelope,
        test_mcp_tool_success_envelope,
        test_mcp_tool_error_envelope,
        test_tool_result_schema_requires_error_for_failure,
        test_retry_policy_retries_retryable_tool_errors,
        test_retry_policy_does_not_retry_non_retryable_tool_errors,
        test_should_retry_respects_recoverable_flag,
        test_arxiv_rate_limit_retries_429_then_succeeds,
        test_arxiv_rate_limit_exhaustion_maps_to_error_code,
        test_rate_limit_utils_build_shared_schedules,
        test_sync_rate_limiter_and_retry_sync_are_generic,
        test_async_rate_limiter_is_generic,
        test_long_term_memory_filters_stale_fetch_url_lessons,
        test_long_term_memory_structured_entries_are_deduped_and_capped,
        test_memory_entry_schema_compacts_tags,
        test_trace_event_schema_and_jsonl_writer,
        test_trace_writer_uses_lock_file,
        test_trace_schema_rejects_unknown_kind,
        test_trace_summarizes_tool_result,
        test_tool_reflect_prompt_formats_structured_schema,
        test_reflection_history_includes_tool_envelope_outcomes,
    ]
    for test in tests:
        await test()
    print(f"\n=== {len(tests)}/{len(tests)} passed ===")


if __name__ == "__main__":
    asyncio.run(main())
