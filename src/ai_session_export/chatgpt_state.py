"""Isolated ChatGPT checkpoint I/O and a dry-run-first, restartable state split.

The split does not read or rewrite Markdown, reset a checkpoint, or emit an
Observer handoff. The old shared checkpoint retains only a relocation marker
for ChatGPT; all other source values remain unchanged.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

from .state import RELOCATIONS_KEY, _atomic_write_state


def _read_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("state document must be an object")
    return value


def _validate_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or set(state) != {"chatgpt"}:
        raise ValueError("dedicated ChatGPT state must contain only the chatgpt source")
    source = state["chatgpt"]
    if not isinstance(source, dict) or not isinstance(source.get("sessions"), dict):
        raise ValueError("dedicated ChatGPT state requires its existing sessions map")
    generation = source.get("archive_generation", 0)
    if type(generation) is not int or generation < 0:
        raise ValueError("invalid ChatGPT archive generation")
    if not isinstance(source.get("observer_handoffs", []), list):
        raise ValueError("invalid ChatGPT Observer handoffs")


def load_chatgpt_state(path: Path) -> dict[str, Any]:
    """Load the exact checkpoint; a missing file is never an implicit baseline."""
    state = _read_document(path)
    _validate_state(state)
    return state


def save_chatgpt_state(state: dict[str, Any], path: Path) -> None:
    """Save only ChatGPT. Callers hold the dedicated state-file writer lock."""
    _validate_state(state)
    if path.exists():
        load_chatgpt_state(path)
    _atomic_write_state(state, path)


def _digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _paths(archive_root: Path) -> tuple[Path, Path, Path]:
    root = archive_root.expanduser().resolve()
    shared = root / ".export_state.json"
    dedicated = root / "chatgpt" / ".pipeline" / "export_state.json"
    backup = dedicated.with_name("export_state.before-split.json")
    # A symlink must not silently expand the one-time migration's write scope.
    for path in (root / "chatgpt", dedicated.parent, shared, dedicated, backup):
        if path.is_symlink():
            raise ValueError("state split does not follow symlinked targets")
    return shared, dedicated, backup


def _prepare(archive_root: Path) -> dict[str, Any]:
    shared, dedicated, backup = _paths(archive_root)
    original = _read_document(shared)
    relocations = original.get(RELOCATIONS_KEY, {})
    if not isinstance(relocations, dict):
        raise ValueError("invalid source relocation metadata")
    marker = relocations.get("chatgpt")
    if marker is not None:
        if "chatgpt" in original:
            raise ValueError("shared state has both ChatGPT data and a relocation marker")
        if not isinstance(marker, dict) or marker.get("schema_version") != 1:
            raise ValueError("unsupported ChatGPT relocation marker")
        if marker.get("state_file") != str(dedicated) or marker.get("backup_file") != str(backup):
            raise ValueError("ChatGPT relocation points to a different target")
        if marker.get("phase") not in {"pending", "complete"}:
            raise ValueError("invalid ChatGPT relocation phase")
        snapshot = load_chatgpt_state(backup)
        if _digest(snapshot) != marker.get("snapshot_sha256"):
            raise ValueError("ChatGPT migration backup does not match its recorded hash")
        if dedicated.exists():
            current = load_chatgpt_state(dedicated)
            # The destination may legitimately advance after migration. A retry
            # verifies it but never copies the older backup over a newer cursor.
            if current["chatgpt"].get("archive_generation", 0) < snapshot["chatgpt"].get("archive_generation", 0):
                raise ValueError("dedicated ChatGPT checkpoint regressed after migration")
            status = "already_migrated" if marker["phase"] == "complete" else "finalize_required"
        else:
            if marker["phase"] == "complete":
                raise ValueError("completed migration is missing its dedicated checkpoint; refusing to restore an old backup")
            current = snapshot
            status = "resume_required"
        shared_after = original
    else:
        if "chatgpt" not in original:
            raise ValueError("shared checkpoint has no ChatGPT state to migrate")
        snapshot = {"chatgpt": deepcopy(original["chatgpt"])}
        _validate_state(snapshot)
        if dedicated.exists():
            raise ValueError("dedicated ChatGPT checkpoint already exists without a relocation marker")
        if backup.exists() and load_chatgpt_state(backup) != snapshot:
            raise ValueError("migration backup conflicts with the current shared checkpoint")
        marker = {
            "schema_version": 1,
            "state_file": str(dedicated),
            "backup_file": str(backup),
            "snapshot_sha256": _digest(snapshot),
            "phase": "pending",
        }
        shared_after = deepcopy(original)
        del shared_after["chatgpt"]
        shared_after.setdefault(RELOCATIONS_KEY, {})["chatgpt"] = marker
        current = snapshot
        status = "migration_required"
    return {
        "shared": shared,
        "dedicated": dedicated,
        "backup": backup,
        "original": original,
        "shared_after": shared_after,
        "snapshot": snapshot,
        "current": current,
        "status": status,
    }


@contextmanager
def _migration_locks(shared: Path, dedicated: Path):
    handles = []
    try:
        for state_file in (shared, dedicated):
            lock = state_file.with_name(f"{state_file.name}.chatgpt.lock")
            if lock.is_symlink():
                raise ValueError("state split does not follow symlinked locks")
            lock.parent.mkdir(parents=True, exist_ok=True)
            handle = lock.open("a+", encoding="utf-8")
            handles.append(handle)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _report(plan: dict[str, Any], *, applied: bool) -> dict[str, Any]:
    source = plan["current"]["chatgpt"]
    others = {key: value for key, value in plan["original"].items() if key not in {"chatgpt", RELOCATIONS_KEY}}
    return {
        "schema_version": 1,
        "operation": "split_chatgpt_state",
        "applied": applied,
        "status": "migrated" if applied and plan["status"] != "already_migrated" else plan["status"],
        "shared_state_file": str(plan["shared"]),
        "chatgpt_state_file": str(plan["dedicated"]),
        "backup_file": str(plan["backup"]),
        "archive_generation": source.get("archive_generation", 0),
        "session_count": len(source["sessions"]),
        "observer_handoff_count": len(source.get("observer_handoffs", [])),
        "chatgpt_snapshot_sha256": _digest(plan["snapshot"]),
        "other_sources_sha256": _digest(others),
        "markdown_modified": False,
    }


def split_chatgpt_state(archive_root: Path, *, apply: bool = False) -> dict[str, Any]:
    """Preserve the entire ChatGPT checkpoint under its own archive directory.

    The order is backup -> retire legacy source -> publish dedicated source.
    A crash during the split therefore fails closed and can resume from the
    hashed ChatGPT-only backup. No moment allows two live checkpoint writers.
    """
    if not apply:
        return _report(_prepare(archive_root), applied=False)
    shared, dedicated, _ = _paths(archive_root)
    # Reject invalid input before creating lock files. Recheck under both locks.
    _prepare(archive_root)
    with _migration_locks(shared, dedicated):
        plan = _prepare(archive_root)
        if plan["status"] == "already_migrated":
            return _report(plan, applied=True)
        if not plan["backup"].exists():
            _atomic_write_state(plan["snapshot"], plan["backup"])
        if plan["status"] == "migration_required":
            _atomic_write_state(plan["shared_after"], shared)
        if plan["status"] != "finalize_required":
            save_chatgpt_state(plan["snapshot"], dedicated)
        if load_chatgpt_state(dedicated) != plan["current"]:
            raise ValueError("dedicated state verification failed")
        if _read_document(shared) != plan["shared_after"]:
            raise ValueError("shared state verification failed")
        completed_shared = deepcopy(plan["shared_after"])
        completed_shared[RELOCATIONS_KEY]["chatgpt"]["phase"] = "complete"
        _atomic_write_state(completed_shared, shared)
        return _report(plan, applied=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply the split; default is a read-only dry run.")
    args = parser.parse_args()
    try:
        report = split_chatgpt_state(args.archive_root, apply=args.apply)
    except (OSError, ValueError) as error:
        print(json.dumps({"operation": "split_chatgpt_state", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
