# researcher/errors.py


def llm_error(exc: BaseException) -> str:
    return f"LLM error: {exc}. Please try again."


def invalid_json() -> str:
    return "Your response was not valid JSON. Respond with a JSON object containing either 'action' and 'action_input', or 'final_answer'."


def missing_action() -> str:
    return "Error: your response must include either 'action' or 'final_answer'. Please try again."


def tool_error(exc: BaseException) -> str:
    return f"Tool error: {exc}"


def final_generation_error(exc: BaseException) -> str:
    return f"[Error generating final answer: {exc}]"


def task_interrupted(exc: BaseException) -> str:
    return f"[Task interrupted: {type(exc).__name__}: {exc}]"


def no_answer() -> str:
    return "[No answer produced]"


def is_failure_text(text: str) -> bool:
    return text.startswith("[") and (
        "error" in text.lower() or "interrupted" in text.lower() or "no answer" in text.lower()
    )
