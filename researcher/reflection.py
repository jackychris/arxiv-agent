import json

from memory.short_term import ShortTermMemory

_OBS_PREFIX = "Observation:"


def _compact_observation(content: str) -> str:
    raw = content[len(_OBS_PREFIX) :].strip()
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        return f"OBSERVATION: {raw[:300]}"

    if not isinstance(envelope, dict):
        return f"OBSERVATION: {raw[:300]}"

    tool = envelope.get("tool") or "unknown"
    ok = envelope.get("ok")
    if ok is False:
        error = envelope.get("error") or {}
        if not isinstance(error, dict):
            error = {"message": str(error)}
        code = error.get("code") or "TOOL_ERROR"
        message = error.get("message") or ""
        return f"OBSERVATION: tool={tool} ok=false error_code={code} error={message[:220]}"

    data = envelope.get("data")
    preview = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return f"OBSERVATION: tool={tool} ok={str(ok).lower()} data={preview[:260]}"


def build_reflection_history(memory: ShortTermMemory) -> str:
    lines = []
    for message in memory.get():
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            lines.append(f"ASSISTANT: {content[:500]}")
        elif role == "user" and content.startswith(_OBS_PREFIX):
            lines.append(_compact_observation(content))
        elif role == "user":
            lines.append(f"USER: {content[:300]}")
    return "\n".join(lines)
