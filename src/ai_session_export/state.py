from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_STATE = {
    "second_mind": {"last_export_count": 0},
    "opencode": {"last_session_time": 0},
    "claude_code": {"last_timestamp": 0},
    "codex": {"sessions": {}},
    "chatgpt": {"sessions": {}, "official_seed": {"status": "unused"}},
    "antigravity": {"last_timestamp": 0, "legacy_cursor_migrated": False, "surfaces": {}},
    "cursor": {"sessions": {}},
    "dsh": {"sessions": {}},
}

RELOCATIONS_KEY = "_source_relocations"


def assert_chatgpt_state_local(state: dict[str, Any]) -> None:
    """Reject a retired shared checkpoint before a ChatGPT writer touches archives."""
    relocations = state.get(RELOCATIONS_KEY, {})
    if not isinstance(relocations, dict):
        raise ValueError("invalid source relocation metadata")
    if "chatgpt" in relocations:
        raise ValueError("ChatGPT state was relocated; use the dedicated ChatGPT state file")


def load_state(
    state_file: Path, *, sources: tuple[str, ...] | None = None
) -> dict[str, Any]:
    if not state_file.exists():
        return {key: deepcopy(DEFAULT_STATE[key]) for key in sources or DEFAULT_STATE}
    with state_file.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if not isinstance(state, dict):
        raise ValueError("exporter state must be an object")
    relocations = state.get(RELOCATIONS_KEY, {})
    if not isinstance(relocations, dict):
        raise ValueError("invalid source relocation metadata")
    for source in sources or DEFAULT_STATE:
        if source in relocations:
            if source in state:
                raise ValueError("relocated source remains in shared exporter state")
            continue
        defaults = DEFAULT_STATE[source]
        state.setdefault(source, {})
        for key, value in defaults.items():
            state[source].setdefault(key, deepcopy(value))
    return state


def _atomic_write_state(state: dict[str, Any], state_file: Path) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{state_file.name}.", dir=state_file.parent)
    temp_file = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_file.replace(state_file)
        directory = os.open(state_file.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp_file.unlink(missing_ok=True)


def save_state(state: dict[str, Any], state_file: Path) -> None:
    """Keep a stale shared-state writer from undoing a source relocation.

    Callers serialize with the state-file lock, including the migration. The
    disk check additionally catches a stale in-memory state saved after a move.
    """
    if state_file.exists():
        current = json.loads(state_file.read_text(encoding="utf-8"))
        relocations = current.get(RELOCATIONS_KEY, {})
        if not isinstance(relocations, dict):
            raise ValueError("invalid source relocation metadata")
        if relocations:
            if state.get(RELOCATIONS_KEY) != relocations:
                raise ValueError("cannot overwrite source relocation metadata")
            if any(source in state for source in relocations):
                raise ValueError("cannot restore a relocated source to shared state")
    _atomic_write_state(state, state_file)
