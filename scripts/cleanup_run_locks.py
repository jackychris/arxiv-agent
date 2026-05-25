from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

TERMINAL_TYPES = {"final", "error"}
TERMINAL_KINDS = {"run_final", "error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove stale runs/*.jsonl.lock files.")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        type=Path,
        help="Directory containing run trace JSONL files.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete stale lock files. Without this flag, only report what would be deleted.",
    )
    return parser.parse_args()


def load_last_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            position = f.tell()
            buffer = bytearray()
            while position > 0:
                position -= 1
                f.seek(position)
                char = f.read(1)
                if char == b"\n" and buffer:
                    break
                buffer.extend(char)
    except FileNotFoundError:
        return None

    if not buffer:
        return None

    line = bytes(reversed(buffer)).strip()
    if not line:
        return None

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def has_terminal_event(path: Path) -> bool:
    payload = load_last_json(path)
    if payload is None:
        return False
    event_type = payload.get("type")
    kind = payload.get("kind")
    return event_type in TERMINAL_TYPES or kind in TERMINAL_KINDS


def lock_is_available(path: Path) -> bool:
    if os.name != "posix":
        return True

    import fcntl

    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return False
    locked = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except BlockingIOError:
            return False
        finally:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return True


def classify_lock(lock_path: Path) -> str | None:
    if lock_path.suffix != ".lock":
        return None
    jsonl_path = lock_path.with_suffix("")
    if not jsonl_path.exists():
        return "missing-jsonl"
    if has_terminal_event(jsonl_path):
        return "terminal-jsonl"
    return None


def main() -> int:
    args = parse_args()
    locks = sorted(args.runs_dir.glob("*.jsonl.lock"))
    stale: list[tuple[Path, str]] = []
    busy: list[Path] = []

    for lock_path in locks:
        reason = classify_lock(lock_path)
        if reason is None:
            continue
        if not lock_is_available(lock_path):
            busy.append(lock_path)
            continue
        stale.append((lock_path, reason))

    for lock_path, reason in stale:
        action = "delete" if args.delete else "would delete"
        print(f"{action}: {lock_path} ({reason})")
        if args.delete:
            lock_path.unlink()

    for lock_path in busy:
        print(f"skip busy: {lock_path}")

    mode = "deleted" if args.delete else "dry-run"
    print(
        f"{mode}: scanned={len(locks)} stale={len(stale)} busy={len(busy)} "
        f"remaining={len(locks) - (len(stale) if args.delete else 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
