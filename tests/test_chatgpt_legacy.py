from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_session_export.archive import content_sha256, parse_marked_markdown
from ai_session_export.chatgpt_legacy import (
    ChatGPTLegacyImportError,
    apply_legacy_import,
    parse_legacy_markdown,
    plan_legacy_import,
)
from ai_session_export.markdown import render_markdown
from ai_session_export.models import MessageTurn, SessionRecord
from ai_session_export.state import DEFAULT_STATE, load_state, save_state


PROJECT_ID = "g-p-example"
PROJECT_LABEL = "01_Example"


def _config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "projects": {PROJECT_ID: {"label": PROJECT_LABEL}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _legacy(path: Path, *, conversation_id: str = "conversation-1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Example title

- **创建**: 2026-01-02 09:00
- **最后更新**: 2026-01-03 10:00
- **模型**: gpt-example
- **conversation_id**: `CONVERSATION_ID`

---

### 🧑 user  ·  09:00

Question

### 🤖 assistant  ·  09:01

```markdown
### 🧑 user  ·  10:30
```

Answer
""".replace("CONVERSATION_ID", conversation_id),
        encoding="utf-8",
    )
    return path


def _empty_state(path: Path) -> dict[str, object]:
    state = json.loads(json.dumps(DEFAULT_STATE))
    save_state(state, path)
    return state


def test_parser_is_fence_aware_and_message_ids_are_deterministic(tmp_path: Path) -> None:
    path = _legacy(tmp_path / "legacy.md")

    first = parse_legacy_markdown(
        path,
        project_id=PROJECT_ID,
        project_label=PROJECT_LABEL,
        timezone_name="America/Chicago",
    )
    second = parse_legacy_markdown(
        path,
        project_id=PROJECT_ID,
        project_label=PROJECT_LABEL,
        timezone_name="America/Chicago",
    )

    assert [turn.role for turn in first.messages] == ["user", "assistant"]
    assert "### 🧑 user" in first.messages[1].content
    assert [turn.message_id for turn in first.messages] == [
        turn.message_id for turn in second.messages
    ]
    assert first.messages[0].time_created < first.messages[1].time_created


def test_apply_converts_legacy_file_updates_state_and_indexes(tmp_path: Path) -> None:
    base = tmp_path / "archive"
    root = base / "chatgpt"
    source = _legacy(root / PROJECT_LABEL / "2026-01-02_Example title.md")
    state_file = tmp_path / ".export_state.json"
    state = _empty_state(state_file)
    config = _config(tmp_path / "config.json")
    plan = plan_legacy_import(
        root,
        state,
        config,
        timezone_name="America/Chicago",
        observed_at="2026-08-28T10:00:00-05:00",
    )

    report = apply_legacy_import(plan, state_file)

    assert report["legacy_conversations"] == 1
    assert not source.exists()
    output = root / "01_Example/20260102_Example_title.md"
    turns = parse_marked_markdown(output)
    assert len(turns) == 2
    assert turns[0].content_sha256 == content_sha256("Question")
    stored = load_state(state_file)["chatgpt"]["sessions"]["conversation-1"]
    assert stored["output_file"] == "01_Example/20260102_Example_title.md"
    assert stored["coverage"] == "full_history"
    assert stored["legacy_import"]["merged_with_live"] is False
    assert stored["messages"][0]["message_id"] == turns[0].message_id
    assert "Example title" in (root / PROJECT_LABEL / "README.md").read_text()


def test_live_archive_wins_duplicate_content_and_path(tmp_path: Path) -> None:
    base = tmp_path / "archive"
    root = base / "chatgpt"
    source = _legacy(root / PROJECT_LABEL / "2026-01-02_Example title.md")
    state_file = tmp_path / ".export_state.json"
    state = _empty_state(state_file)
    live_path = root / PROJECT_LABEL / "20260827_Live_title.md"
    live_turns = [
        MessageTurn(
            "user", "Question", 1_788_000_000_000, message_id="live-user"
        ),
        MessageTurn(
            "assistant", "Live answer", 1_788_000_001_000, message_id="live-assistant"
        ),
    ]
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text(
        render_markdown(
            SessionRecord(
                "chatgpt", "conversation-1", "Live title", "2026-08-27", live_turns
            )
        ),
        encoding="utf-8",
    )
    state["chatgpt"]["sessions"]["conversation-1"] = {
        "project_id": PROJECT_ID,
        "project_label": PROJECT_LABEL,
        "output_file": "01_Example/20260827_Live_title.md",
        "thread_updated_at": 1_788_000_001,
        "messages": [
            {
                "message_id": turn.message_id,
                "role": turn.role,
                "created_at": turn.time_created,
                "content_sha256": content_sha256(turn.content),
                "complete": True,
            }
            for turn in live_turns
        ],
    }
    save_state(state, state_file)
    plan = plan_legacy_import(
        root,
        load_state(state_file),
        _config(tmp_path / "config.json"),
        timezone_name="America/Chicago",
        observed_at="2026-08-28T10:00:00-05:00",
    )

    report = apply_legacy_import(plan, state_file)

    assert report["live_overlaps"] == 1
    assert report["duplicate_legacy_turns_removed"] == 1
    assert not source.exists()
    turns = parse_marked_markdown(live_path)
    assert [turn.message_id for turn in turns][-2:] == ["live-user", "live-assistant"]
    assert [turn.content for turn in turns] == [
        "```markdown\n### 🧑 user  ·  10:30\n```\n\nAnswer",
        "Question",
        "Live answer",
    ]
    stored = load_state(state_file)["chatgpt"]["sessions"]["conversation-1"]
    assert stored["output_file"] == "01_Example/20260827_Live_title.md"
    assert stored["coverage"] == "full_history"
    assert stored["legacy_import"]["merged_with_live"] is True


def test_failure_rolls_back_archive_state_and_legacy_source(tmp_path: Path) -> None:
    base = tmp_path / "archive"
    root = base / "chatgpt"
    source = _legacy(root / PROJECT_LABEL / "2026-01-02_Example title.md")
    state_file = tmp_path / ".export_state.json"
    state = _empty_state(state_file)
    original_state = state_file.read_bytes()
    original_source = source.read_bytes()
    plan = plan_legacy_import(
        root,
        state,
        _config(tmp_path / "config.json"),
        timezone_name="America/Chicago",
        observed_at="2026-08-28T10:00:00-05:00",
    )

    def fail_checkpoint(_state: dict[str, object], _path: Path) -> None:
        raise OSError("checkpoint failed")

    with pytest.raises(OSError, match="checkpoint failed"):
        apply_legacy_import(plan, state_file, checkpoint_state=fail_checkpoint)

    assert source.read_bytes() == original_source
    assert state_file.read_bytes() == original_state
    assert not (root / PROJECT_LABEL / "20260102_Example_title.md").exists()


def test_unknown_project_directory_fails_before_mutation(tmp_path: Path) -> None:
    base = tmp_path / "archive"
    root = base / "chatgpt"
    source = _legacy(root / "99_Unknown" / "legacy.md")
    state_file = tmp_path / ".export_state.json"
    state = _empty_state(state_file)

    with pytest.raises(ChatGPTLegacyImportError, match="unapproved Project"):
        plan_legacy_import(
            root,
            state,
            _config(tmp_path / "config.json"),
            timezone_name="America/Chicago",
            observed_at="2026-08-28T10:00:00-05:00",
        )

    assert source.exists()
