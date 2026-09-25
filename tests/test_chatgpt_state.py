from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from ai_session_export import chatgpt_state, cli
from ai_session_export.chatgpt_incremental_cli import run as incremental_run
from ai_session_export.chatgpt_state import (
    load_chatgpt_state,
    save_chatgpt_state,
    split_chatgpt_state,
)
from ai_session_export.sources.chatgpt import export_chatgpt
from ai_session_export.state import DEFAULT_STATE, RELOCATIONS_KEY, load_state, save_state


def _fixture(root: Path) -> tuple[Path, Path, dict]:
    shared = root / ".export_state.json"
    dedicated = root / "chatgpt" / ".pipeline" / "export_state.json"
    state = deepcopy(DEFAULT_STATE)
    state["codex"]["sessions"] = {"codex-fixture": {"latest_timestamp": 123}}
    state["custom_source"] = {"opaque": [False, 0, {"cursor": "unchanged"}]}
    state["chatgpt"] = {
        "sessions": {
            "fixture-thread": {
                "thread_updated_at": 123,
                "output_file": "01_Project/fixture.md",
                "messages": [{"message_id": "message-fixture", "content_sha256": "a" * 64}],
            }
        },
        "archive_generation": 18,
        "observer_handoffs": [{"generation": 18, "previous_generation": 17, "run_id": "fixture-run"}],
        "opaque_future_field": {"retain": [1, 2, 3]},
    }
    shared.write_text(json.dumps(state), encoding="utf-8")
    return shared, dedicated, state


def test_dry_run_makes_no_files_or_checkpoint_changes(tmp_path: Path) -> None:
    shared, dedicated, _ = _fixture(tmp_path)
    before = shared.read_bytes()
    result = split_chatgpt_state(tmp_path)
    assert result["applied"] is False
    assert result["status"] == "migration_required"
    assert result["archive_generation"] == 18
    assert result["session_count"] == 1
    assert shared.read_bytes() == before
    assert not dedicated.parent.exists()
    assert list(tmp_path.iterdir()) == [shared]


def test_split_preserves_all_sources_and_exact_chatgpt_state(tmp_path: Path) -> None:
    shared, dedicated, before = _fixture(tmp_path)
    markdown = tmp_path / "chatgpt" / "01_Project" / "fixture.md"
    markdown.parent.mkdir(parents=True)
    markdown.write_text("unmodified archive", encoding="utf-8")
    result = split_chatgpt_state(tmp_path, apply=True)
    assert result["status"] == "migrated"
    assert result["markdown_modified"] is False
    assert load_chatgpt_state(dedicated) == {"chatgpt": before["chatgpt"]}
    assert load_chatgpt_state(dedicated.with_name("export_state.before-split.json")) == {"chatgpt": before["chatgpt"]}
    after = json.loads(shared.read_text())
    assert "chatgpt" not in after
    assert {key: value for key, value in after.items() if key != RELOCATIONS_KEY} == {
        key: value for key, value in before.items() if key != "chatgpt"
    }
    assert after[RELOCATIONS_KEY]["chatgpt"]["phase"] == "complete"
    assert markdown.read_text() == "unmodified archive"
    assert dedicated.with_name("export_state.json.chatgpt.lock").exists()


def test_completed_retry_never_rolls_back_new_generation(tmp_path: Path) -> None:
    shared, dedicated, _ = _fixture(tmp_path)
    split_chatgpt_state(tmp_path, apply=True)
    state = load_chatgpt_state(dedicated)
    state["chatgpt"]["archive_generation"] = 19
    state["chatgpt"]["observer_handoffs"].append({"generation": 19})
    save_chatgpt_state(state, dedicated)
    before = (shared.read_bytes(), dedicated.read_bytes())
    result = split_chatgpt_state(tmp_path, apply=True)
    assert result["status"] == "already_migrated"
    assert result["archive_generation"] == 19
    assert (shared.read_bytes(), dedicated.read_bytes()) == before


@pytest.mark.parametrize("failed_write", [1, 2, 3, 4])
def test_each_atomic_write_interruption_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_write: int
) -> None:
    _, dedicated, before = _fixture(tmp_path)
    actual = chatgpt_state._atomic_write_state
    count = 0

    def fail_write(state: dict, path: Path) -> None:
        nonlocal count
        count += 1
        if count == failed_write:
            raise OSError("synthetic interrupted write")
        actual(state, path)

    monkeypatch.setattr(chatgpt_state, "_atomic_write_state", fail_write)
    with pytest.raises(OSError, match="synthetic"):
        split_chatgpt_state(tmp_path, apply=True)
    monkeypatch.setattr(chatgpt_state, "_atomic_write_state", actual)
    assert split_chatgpt_state(tmp_path, apply=True)["status"] == "migrated"
    assert load_chatgpt_state(dedicated) == {"chatgpt": before["chatgpt"]}


def test_missing_completed_checkpoint_does_not_restore_old_backup(tmp_path: Path) -> None:
    _, dedicated, _ = _fixture(tmp_path)
    split_chatgpt_state(tmp_path, apply=True)
    dedicated.unlink()
    with pytest.raises(ValueError, match="refusing to restore"):
        split_chatgpt_state(tmp_path, apply=True)
    assert not dedicated.exists()


def test_pending_finalize_preserves_same_generation_partial_checkpoint(tmp_path: Path) -> None:
    shared, dedicated, _ = _fixture(tmp_path)
    split_chatgpt_state(tmp_path, apply=True)
    pending = json.loads(shared.read_text())
    pending[RELOCATIONS_KEY]["chatgpt"]["phase"] = "pending"
    chatgpt_state._atomic_write_state(pending, shared)
    state = load_chatgpt_state(dedicated)
    state["chatgpt"]["sessions"]["fixture-thread"]["last_error"] = "retryable synthetic failure"
    save_chatgpt_state(state, dedicated)
    assert split_chatgpt_state(tmp_path, apply=True)["status"] == "migrated"
    assert load_chatgpt_state(dedicated) == state
    assert json.loads(shared.read_text())[RELOCATIONS_KEY]["chatgpt"]["phase"] == "complete"


def test_pending_migration_backup_hash_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, dedicated, _ = _fixture(tmp_path)
    actual = chatgpt_state.save_chatgpt_state
    monkeypatch.setattr(chatgpt_state, "save_chatgpt_state", lambda *_: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        split_chatgpt_state(tmp_path, apply=True)
    backup = dedicated.with_name("export_state.before-split.json")
    changed = load_chatgpt_state(backup)
    changed["chatgpt"]["archive_generation"] = 17
    actual(changed, backup)
    monkeypatch.setattr(chatgpt_state, "save_chatgpt_state", actual)
    with pytest.raises(ValueError, match="recorded hash"):
        split_chatgpt_state(tmp_path, apply=True)


def test_other_exporter_continues_without_resurrecting_chatgpt(tmp_path: Path) -> None:
    shared, dedicated, _ = _fixture(tmp_path)
    split_chatgpt_state(tmp_path, apply=True)
    before = dedicated.read_bytes()
    result = cli.run_export(
        "codex", full=False, dry_run=False, base_dir=tmp_path,
        state_file=shared, codex_session_dirs=(tmp_path / "missing-sessions",),
        codex_session_index=tmp_path / "missing-index.jsonl",
    )
    assert result[0]["exported"] == 0
    assert "chatgpt" not in load_state(shared)
    assert json.loads(shared.read_text())[RELOCATIONS_KEY]["chatgpt"]["phase"] == "complete"
    assert dedicated.read_bytes() == before


def test_stale_shared_writer_cannot_undo_relocation(tmp_path: Path) -> None:
    shared, _, before = _fixture(tmp_path)
    split_chatgpt_state(tmp_path, apply=True)
    after = shared.read_bytes()
    with pytest.raises(ValueError, match="relocation metadata"):
        save_state(before, shared)
    state = load_state(shared)
    state["chatgpt"] = before["chatgpt"]
    with pytest.raises(ValueError, match="restore a relocated"):
        save_state(state, shared)
    assert shared.read_bytes() == after


def test_old_chatgpt_entrypoint_fails_before_reading_input(tmp_path: Path) -> None:
    shared, _, _ = _fixture(tmp_path)
    split_chatgpt_state(tmp_path, apply=True)
    with pytest.raises(ValueError, match="relocated"):
        cli.run_export(
            "chatgpt", full=False, dry_run=False, base_dir=tmp_path, state_file=shared,
            chatgpt_input=tmp_path / "missing", chatgpt_project_config=tmp_path / "missing",
        )
    with pytest.raises(ValueError, match="relocated"):
        export_chatgpt(
            tmp_path / "chatgpt", load_state(shared), source_input=Path("-"),
            project_config=tmp_path / "missing", full=False, dry_run=False,
            since_date=None,
        )


def test_incremental_cli_rejects_retired_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shared, _, _ = _fixture(tmp_path)
    split_chatgpt_state(tmp_path, apply=True)
    monkeypatch.setattr("ai_session_export.chatgpt_incremental_cli.load_incremental_scope_allowlist", lambda _: {})
    args = argparse.Namespace(action="activate-live", state_file=shared, chatgpt_project_config=tmp_path / "unused", at="2026-09-08T00:00:00Z")
    with pytest.raises(ValueError, match="relocated"):
        incremental_run(args, None)


def test_dedicated_io_never_defaults_or_accepts_other_sources(tmp_path: Path) -> None:
    path = tmp_path / "dedicated.json"
    with pytest.raises(FileNotFoundError):
        load_chatgpt_state(path)
    with pytest.raises(ValueError, match="only"):
        save_chatgpt_state(deepcopy(DEFAULT_STATE), path)
    state = {"chatgpt": {"sessions": {}, "archive_generation": 18}}
    save_chatgpt_state(state, path)
    assert load_chatgpt_state(path) == state
    assert load_state(path, sources=("chatgpt",))["chatgpt"]["archive_generation"] == 18
    assert set(load_state(path, sources=("chatgpt",))) == {"chatgpt"}


def test_migration_waits_for_legacy_writer_lock(tmp_path: Path) -> None:
    shared, dedicated, _ = _fixture(tmp_path)
    script = """
import sys
from pathlib import Path
from ai_session_export.chatgpt_state import split_chatgpt_state
print('ready', flush=True)
split_chatgpt_state(Path(sys.argv[1]), apply=True)
"""
    with cli._state_write_lock(shared):
        process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "ready"
        assert process.poll() is None
        assert not dedicated.exists()
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    assert dedicated.exists()


def test_symlinked_target_rejected_without_writes(tmp_path: Path) -> None:
    shared, _, _ = _fixture(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (tmp_path / "chatgpt").symlink_to(outside, target_is_directory=True)
    before = shared.read_bytes()
    with pytest.raises(ValueError, match="symlinked"):
        split_chatgpt_state(tmp_path, apply=True)
    assert shared.read_bytes() == before
    assert not list(outside.iterdir())


def test_existing_dedicated_state_requires_proven_relocation(tmp_path: Path) -> None:
    shared, dedicated, state = _fixture(tmp_path)
    save_chatgpt_state({"chatgpt": state["chatgpt"]}, dedicated)
    before = shared.read_bytes()
    with pytest.raises(ValueError, match="without a relocation marker"):
        split_chatgpt_state(tmp_path, apply=True)
    assert shared.read_bytes() == before
