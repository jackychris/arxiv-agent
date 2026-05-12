# researcher/events.py
import json


def step_status(step: int, thought: str, action: str, action_input: dict) -> str:
    return json.dumps(
        {
            "kind": "step",
            "step": step,
            "thought": thought[:120],
            "tool": action,
            "tool_input": action_input,
            "content": f"Step {step}: {action}",
        },
        ensure_ascii=False,
    )


def observation_status(
    observation: str, step: int | None = None, tool: str | None = None, max_chars: int = 500
) -> str:
    preview = observation[:max_chars] + ("..." if len(observation) > max_chars else "")
    return json.dumps(
        {
            "kind": "observation",
            "step": step,
            "tool": tool,
            "preview": preview,
            "truncated": len(observation) > max_chars,
            "content": observation,
            "raw": observation,
        },
        ensure_ascii=False,
    )
