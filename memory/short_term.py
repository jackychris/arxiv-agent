import logging

logger = logging.getLogger(__name__)

# ~20K tokens at 4 chars/token — well within DeepSeek's 128K context window.
# Trimming kicks in when tool outputs (HTML, PDFs) push observations near the
# 6000-char ceiling across many steps.
_MAX_CHARS = 80_000


def _total_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


def _trim(messages: list[dict]) -> list[dict]:
    if _total_chars(messages) <= _MAX_CHARS:
        return messages

    # Anchors: all leading system messages + the first user message (the mission).
    # These are never dropped — the model needs them to stay on task.
    anchor_end = 0
    for i, msg in enumerate(messages):
        anchor_end = i + 1
        if msg["role"] == "user":
            break

    anchors = messages[:anchor_end]
    tail = list(messages[anchor_end:])

    dropped = 0
    while tail and _total_chars(anchors) + _total_chars(tail) > _MAX_CHARS:
        tail.pop(0)
        dropped += 1

    logger.warning(
        "Context window trimmed: dropped %d messages (%d chars total)",
        dropped,
        _total_chars(messages),
    )
    marker = {
        "role": "system",
        "content": f"[{dropped} earlier step(s) omitted to stay within context window]",
    }
    return anchors + [marker] + tail


class ShortTermMemory:
    def __init__(self):
        self.messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get(self) -> list[dict]:
        return _trim(self.messages)
