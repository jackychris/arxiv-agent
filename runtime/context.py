from runtime.storage import InMemoryRunContextStore, RunContextStore


class RunContext:
    def __init__(self, store: RunContextStore | None = None):
        self._store: RunContextStore = store or InMemoryRunContextStore()

    def set_store(self, store: RunContextStore) -> None:
        self._store = store

    def get_store(self) -> RunContextStore:
        return self._store

    def publish(self, context_id: str, task_id: str, step: int, content: str) -> None:
        self._store.publish(context_id, task_id, step, content)

    def get_others(self, context_id: str, task_id: str) -> str:
        entries = self._store.get_entries(context_id)
        lines = [
            f"[Step {step}] {content}"
            for tid, step, content in entries
            if tid != task_id
        ]
        if not lines:
            return ""
        return "Other tasks found so far:\n" + "\n".join(lines[-10:])

    def clear(self, context_id: str) -> None:
        self._store.clear(context_id)


run_context = RunContext()
