from runtime.storage import InMemoryRunContextStore, RunContextStore


class RunContext:
    def __init__(self, store: RunContextStore | None = None):
        self._store: RunContextStore = store or InMemoryRunContextStore()

    def set_store(self, store: RunContextStore) -> None:
        self._store = store

    def get_store(self) -> RunContextStore:
        return self._store

    def publish_citations(self, context_id: str, task_id: str, citations: list[dict]) -> None:
        self._store.publish_citations(context_id, task_id, citations)

    def publish_history(self, context_id: str, task_id: str, history: str) -> None:
        self._store.publish_history(context_id, task_id, history)

    def get_others(self, context_id: str, task_id: str) -> str:
        citations = self._store.get_citations_from_others(context_id, task_id)
        if not citations:
            return ""
        lines = []
        for c in citations:
            key = c.get("key")
            title = c.get("title") or key
            cited = "cited" if c.get("cited") else "uncited"
            if key:
                ref_num = c.get("ref_num")
                label = f"#{ref_num} [{key}]" if ref_num is not None else f"[{key}]"
                lines.append(f"- {label} {title} ({cited})")
        return "Citations already found by other tasks:\n" + "\n".join(lines)

    def get_citations(self, context_id: str) -> list[dict]:
        return self._store.get_citations(context_id)

    def get_histories(self, context_id: str) -> list[dict]:
        return self._store.get_histories(context_id)

    def clear(self, context_id: str) -> None:
        self._store.clear(context_id)


run_context = RunContext()
