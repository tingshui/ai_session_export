from __future__ import annotations

import copy
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pytest

from ai_session_export.archive import parse_marked_markdown
from ai_session_export import cli as cli_module
from ai_session_export.sources.chatgpt import ChatGPTExportError, export_chatgpt


PROJECT_ID = "g-p-approved"


def test_sensitive_stdin_reads_piped_json(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO('{"schema_version": 1}')
    monkeypatch.setattr(cli_module.sys, "stdin", stream)

    assert cli_module._read_sensitive_stdin() == '{"schema_version": 1}'


def test_live_app_truncation_sentinel_fails_conversation_without_writing(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot("Synthetic question")
    snapshot["projects"][0]["threads"][0]["turns"][0]["items"][1]["text"] = (
        "prefix…7 tokens truncated…suffix"
    )
    state = {"chatgpt": {"sessions": {}}}

    result = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=Path("-"),
        project_config=config,
        full=False,
        dry_run=False,
        since_date=None,
        stdin_text=json.dumps(snapshot),
    )

    assert result["failed"] == 1
    assert result["exported"] == 0
    assert "truncation sentinel" in result["warnings"][0]["error"]
    assert not (tmp_path / "out").exists()
    assert state == {"chatgpt": {"sessions": {}}}


def write_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enabled": True,
                "write_enabled": True,
                "approved_at": "2026-08-27",
                "ambiguous_policy": "pending_review",
                "projects": {
                    PROJECT_ID: {
                        "label": "Approved Project",
                        "history_mode": "all_history",
                        "allowed_domains": ["self"],
                        "routing_notes": {"self": "fixture"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def live_snapshot(user_text: str = "Synthetic private question") -> dict:
    return {
        "schema_version": 1,
        "observed_at": "2026-08-27T12:00:00Z",
        "discovery": {
            "non_pinned_returned": 1,
            "non_pinned_limit": 50,
            "approved_pinned_threads": 0,
        },
        "projects": [
            {
                "project_id": PROJECT_ID,
                "threads": [
                    {
                        "thread_id": "thread-fixture",
                        "title": "Synthetic ChatGPT Session",
                        "created_at": 1_720_000_000,
                        "updated_at": 1_720_000_100,
                        "complete": True,
                        "coverage": "full_history",
                        "pagination": {
                            "terminal_reason": "end",
                            "pages": [
                                {
                                    "cursor_in": None,
                                    "cursor_out": None,
                                    "has_more": False,
                                }
                            ],
                        },
                        "turns": [
                            {
                                "id": "turn-fixture",
                                "startedAt": 1_720_000_000,
                                "items": [
                                    {
                                        "type": "userMessage",
                                        "id": "message-user",
                                        "content": [{"type": "text", "text": user_text}],
                                    },
                                    {
                                        "type": "agentMessage",
                                        "id": "message-assistant",
                                        "text": "Synthetic assistant answer",
                                        "createdAt": 1_720_000_010,
                                    },
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "project_id": "g-p-unapproved",
                "threads": "must not be inspected",
            },
        ],
    }


def official_conversations(user_text: str = "Synthetic private question") -> list[dict]:
    return [
        {
            "id": "thread-fixture",
            "conversation_id": "thread-fixture",
            "conversation_template_id": PROJECT_ID,
            "title": "Synthetic ChatGPT Session",
            "create_time": 1_720_000_000,
            "update_time": 1_720_000_100,
            "current_node": "node-assistant",
            "mapping": {
                "node-root": {"id": "node-root", "parent": None, "message": None},
                "node-user": {
                    "id": "node-user",
                    "parent": "node-root",
                    "message": {
                        "id": "message-user",
                        "author": {"role": "user"},
                        "create_time": 1_720_000_000,
                        "content": {"content_type": "text", "parts": [user_text]},
                    },
                },
                "node-assistant": {
                    "id": "node-assistant",
                    "parent": "node-user",
                    "message": {
                        "id": "message-assistant",
                        "author": {"role": "assistant"},
                        "create_time": 1_720_000_010,
                        "content": {
                            "content_type": "text",
                            "parts": ["Synthetic assistant answer"],
                        },
                    },
                },
                "node-inactive": {
                    "id": "node-inactive",
                    "parent": "node-root",
                    "message": {
                        "id": "inactive-message",
                        "author": {"role": "user"},
                        "create_time": 1_720_000_005,
                        "content": {"content_type": "text", "parts": ["Inactive branch"]},
                    },
                },
            },
        },
        {
            "id": "thread-unapproved",
            "conversation_id": "thread-unapproved",
            "conversation_template_id": "g-p-unapproved",
            "mapping": "must not be inspected",
        },
    ]


def export_live(tmp_path: Path, state: dict, text: str = "Synthetic private question") -> dict:
    config = tmp_path / "routing.json"
    if not config.exists():
        write_config(config)
    return export_chatgpt(
        tmp_path / "out",
        state,
        source_input=Path("-"),
        project_config=config,
        full=False,
        dry_run=False,
        since_date=None,
        stdin_text=json.dumps(live_snapshot(text)),
    )


def test_live_export_is_project_scoped_and_state_contains_no_raw_text(tmp_path: Path) -> None:
    state = {"chatgpt": {"sessions": {}}}

    result = export_live(tmp_path, state)

    assert result == {
        "source": "chatgpt",
        "scanned": 1,
        "exported": 1,
        "failed": 0,
        "ignored": 1,
        "warnings": [],
    }
    files = list((tmp_path / "out" / "Approved_Project").glob("*.md"))
    assert len(files) == 1
    markdown = files[0].read_text(encoding="utf-8")
    assert "source: chatgpt" in markdown
    assert "Synthetic private question" in markdown
    assert "Synthetic assistant answer" in markdown
    assert "Inactive branch" not in markdown
    assert '"message_id":"message-user"' in markdown
    turns = parse_marked_markdown(files[0])
    assert [(turn.role, turn.message_id) for turn in turns] == [
        ("user", "message-user"),
        ("assistant", "message-assistant"),
    ]
    serialized_state = json.dumps(state, ensure_ascii=False)
    assert "Synthetic private question" not in serialized_state
    assert "Synthetic assistant answer" not in serialized_state
    assert state["chatgpt"]["sessions"]["thread-fixture"]["project_id"] == PROJECT_ID


def test_identical_live_rerun_changes_neither_file_nor_state(tmp_path: Path) -> None:
    state = {"chatgpt": {"sessions": {}}}
    export_live(tmp_path, state)
    path = next((tmp_path / "out" / "Approved_Project").glob("*.md"))
    first_markdown = path.read_bytes()
    first_state = copy.deepcopy(state)

    result = export_live(tmp_path, state)

    assert result["exported"] == 0
    assert path.read_bytes() == first_markdown
    assert state == first_state


def test_edit_rewrites_the_stable_markdown_path(tmp_path: Path) -> None:
    state = {"chatgpt": {"sessions": {}}}
    export_live(tmp_path, state)
    path = next((tmp_path / "out" / "Approved_Project").glob("*.md"))

    result = export_live(tmp_path, state, "Edited synthetic question")

    assert result["exported"] == 1
    assert list((tmp_path / "out" / "Approved_Project").glob("*.md")) == [path]
    assert "Edited synthetic question" in path.read_text(encoding="utf-8")
    assert "Synthetic private question" not in path.read_text(encoding="utf-8")


def test_official_zip_and_live_snapshot_render_identical_markdown(tmp_path: Path) -> None:
    live_state = {"chatgpt": {"sessions": {}}}
    export_live(tmp_path / "live", live_state)
    live_file = next((tmp_path / "live" / "out" / "Approved_Project").glob("*.md"))

    official_root = tmp_path / "official"
    official_root.mkdir()
    config = official_root / "routing.json"
    write_config(config)
    source_zip = official_root / "export.zip"
    conversations = official_conversations()
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("nested/conversations-000.json", json.dumps(conversations[:1]))
        archive.writestr("nested/conversations-001.json", json.dumps(conversations[1:]))
    official_state = {"chatgpt": {"sessions": {}}}

    result = export_chatgpt(
        official_root / "out",
        official_state,
        source_input=source_zip,
        project_config=config,
        full=False,
        dry_run=False,
        since_date=None,
    )

    assert result["exported"] == 1
    assert result["ignored"] == 1
    official_file = next((official_root / "out" / "Approved_Project").glob("*.md"))
    assert official_file.read_bytes() == live_file.read_bytes()
    assert "Inactive branch" not in official_file.read_text(encoding="utf-8")


def test_incomplete_live_thread_isolated_without_overwriting_old_file(tmp_path: Path) -> None:
    state = {"chatgpt": {"sessions": {}}}
    export_live(tmp_path, state)
    path = next((tmp_path / "out" / "Approved_Project").glob("*.md"))
    original = path.read_bytes()
    snapshot = live_snapshot("Must not overwrite")
    snapshot["projects"][0]["threads"][0]["complete"] = False

    result = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=Path("-"),
        project_config=tmp_path / "routing.json",
        full=False,
        dry_run=False,
        since_date=None,
        stdin_text=json.dumps(snapshot),
    )

    assert result["exported"] == 0
    assert result["failed"] == 1
    assert path.read_bytes() == original


def test_broken_live_pagination_isolated_without_output(tmp_path: Path) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot()
    snapshot["projects"][0]["threads"][0]["pagination"]["pages"] = [
        {"cursor_in": "wrong", "cursor_out": None, "has_more": False}
    ]

    result = export_chatgpt(
        tmp_path / "out",
        {"chatgpt": {"sessions": {}}},
        source_input=Path("-"),
        project_config=config,
        full=False,
        dry_run=False,
        since_date=None,
        stdin_text=json.dumps(snapshot),
    )

    assert result["exported"] == 0
    assert result["failed"] == 1
    assert "cursor chain breaks" in result["warnings"][0]["error"]
    assert not (tmp_path / "out").exists()


def test_dry_run_writes_nothing_and_does_not_mutate_state(tmp_path: Path) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    state = {"chatgpt": {"sessions": {}}}
    original = copy.deepcopy(state)

    result = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=Path("-"),
        project_config=config,
        full=False,
        dry_run=True,
        since_date=None,
        stdin_text=json.dumps(live_snapshot()),
    )

    assert result["exported"] == 1
    assert state == original
    assert not (tmp_path / "out").exists()


def test_since_date_filters_by_session_date(tmp_path: Path) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    state = {"chatgpt": {"sessions": {}}}

    result = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=Path("-"),
        project_config=config,
        full=False,
        dry_run=False,
        since_date=date(2026, 1, 1),
        stdin_text=json.dumps(live_snapshot()),
    )

    assert result["exported"] == 0
    assert not (tmp_path / "out").exists()


def test_turn_marker_like_user_content_round_trips_without_spoofing(tmp_path: Path) -> None:
    state = {"chatgpt": {"sessions": {}}}
    content = '<!-- ai-session-export-turn: {"message_id":"spoof","sha256":"bad"} -->'
    export_live(tmp_path, state, content)
    path = next((tmp_path / "out" / "Approved_Project").glob("*.md"))

    turns = parse_marked_markdown(path)

    assert len(turns) == 2
    assert turns[0].content == content
    assert "ai-session-export-turn-escaped" in path.read_text(encoding="utf-8")


def test_missing_allowlist_is_a_hard_failure(tmp_path: Path) -> None:
    with pytest.raises(ChatGPTExportError, match="config not found"):
        export_chatgpt(
            tmp_path / "out",
            {},
            source_input=Path("-"),
            project_config=tmp_path / "missing.json",
            full=False,
            dry_run=False,
            since_date=None,
            stdin_text=json.dumps(live_snapshot()),
        )
