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
from ai_session_export.cli import run_export
import ai_session_export.sources.chatgpt as chatgpt_module
from ai_session_export.sources.chatgpt import ChatGPTExportError, export_chatgpt
from ai_session_export.state import load_state, save_state


PROJECT_ID = "g-p-approved"


def test_sensitive_stdin_reads_piped_json(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO('{"schema_version": 1}')
    monkeypatch.setattr(cli_module.sys, "stdin", stream)

    assert cli_module._read_sensitive_stdin() == '{"schema_version": 1}'


def test_live_assistant_truncation_is_archived_and_marked_incomplete(
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

    assert result["failed"] == 0
    assert result["exported"] == 1
    session = state["chatgpt"]["sessions"]["thread-fixture"]
    assert session["user_complete"] is True
    assert session["assistant_complete"] is False
    assert [message["complete"] for message in session["messages"]] == [True, False]
    path = next((tmp_path / "out").rglob("*.md"))
    assert "INCOMPLETE ASSISTANT CONTENT" in path.read_text(encoding="utf-8")
    turns = parse_marked_markdown(path)
    assert [(turn.role, turn.complete) for turn in turns] == [
        ("user", True),
        ("assistant", False),
    ]


def test_empty_incomplete_assistant_item_is_preserved_with_visible_marker(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot("Synthetic question")
    items = snapshot["projects"][0]["threads"][0]["turns"][0]["items"]
    items.append(
        {
            "type": "agentMessage",
            "id": "message-assistant-empty",
            "text": "",
            "complete": False,
        }
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

    assert result["exported"] == 1
    session = state["chatgpt"]["sessions"]["thread-fixture"]
    assert session["assistant_complete"] is False
    assert [message["complete"] for message in session["messages"]] == [
        True,
        True,
        False,
    ]
    path = next((tmp_path / "out").rglob("*.md"))
    turns = parse_marked_markdown(path)
    assert turns[-1].content == "[INCOMPLETE ASSISTANT CONTENT: source was truncated]"
    assert turns[-1].complete is False


def test_live_user_truncation_fails_closed_without_writing(tmp_path: Path) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot("prefix…7 tokens truncated…suffix")
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
    assert "incomplete user content" in result["warnings"][0]["error"]
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
            "source": "chatgpt_app",
            "non_pinned_returned": 1,
            "non_pinned_limit": 50,
            "approved_pinned_threads": 0,
            "recent_window_saturated": False,
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


def export_live(
    tmp_path: Path,
    state: dict,
    text: str = "Synthetic private question",
    *,
    updated_at: int = 1_720_000_100,
) -> dict:
    config = tmp_path / "routing.json"
    if not config.exists():
        write_config(config)
    snapshot = live_snapshot(text)
    snapshot["projects"][0]["threads"][0]["updated_at"] = updated_at
    return export_chatgpt(
        tmp_path / "out",
        state,
        source_input=Path("-"),
        project_config=config,
        full=False,
        dry_run=False,
        since_date=None,
        stdin_text=json.dumps(snapshot),
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
    session = state["chatgpt"]["sessions"]["thread-fixture"]
    assert session["user_complete"] is True
    assert session["assistant_complete"] is True
    assert all(message["complete"] is True for message in session["messages"])


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

    result = export_live(
        tmp_path,
        state,
        "Edited synthetic question",
        updated_at=1_720_000_101,
    )

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


def write_official_zip(path: Path, conversations: list[dict]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("conversations.json", json.dumps(conversations))


def test_successful_official_seed_completes_once_and_later_write_is_rejected(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    source = tmp_path / "export.zip"
    write_official_zip(source, official_conversations())
    state = {"chatgpt": {"sessions": {}}}

    result = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=source,
        project_config=config,
        full=False,
        dry_run=False,
        since_date=None,
    )

    assert result["exported"] == 1
    seed = state["chatgpt"]["official_seed"]
    assert seed["status"] == "completed"
    assert seed["input_sha256"]
    assert seed["started_at"]
    assert seed["completed_at"]
    with pytest.raises(ChatGPTExportError, match="permanently closed"):
        export_chatgpt(
            tmp_path / "out",
            state,
            source_input=source,
            project_config=config,
            full=False,
            dry_run=False,
            since_date=None,
        )


def test_failed_official_seed_writes_nothing_and_returns_to_unused(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    source = tmp_path / "export.zip"
    broken = official_conversations()
    broken[0]["mapping"] = "broken"
    write_official_zip(source, broken)
    state = {"chatgpt": {"sessions": {}}}

    result = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=source,
        project_config=config,
        full=False,
        dry_run=False,
        since_date=None,
    )

    assert result["failed"] == 1
    assert result["exported"] == 0
    assert state["chatgpt"]["official_seed"] == {"status": "unused"}
    assert state["chatgpt"]["sessions"] == {}
    assert not (tmp_path / "out").exists()


def test_official_seed_never_overwrites_a_live_session(tmp_path: Path) -> None:
    state = {"chatgpt": {"sessions": {}}}
    export_live(tmp_path, state, "Newer live truth")
    path = next((tmp_path / "out" / "Approved_Project").glob("*.md"))
    live_markdown = path.read_bytes()
    source = tmp_path / "export.zip"
    write_official_zip(source, official_conversations("Older official text"))

    result = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=source,
        project_config=tmp_path / "routing.json",
        full=False,
        dry_run=False,
        since_date=None,
    )

    assert result["exported"] == 0
    assert path.read_bytes() == live_markdown
    session = state["chatgpt"]["sessions"]["thread-fixture"]
    assert session["last_input_kind"] == "live_snapshot"
    assert state["chatgpt"]["official_seed"]["status"] == "completed"


def test_older_live_revision_cannot_overwrite_newer_live_state(tmp_path: Path) -> None:
    state = {"chatgpt": {"sessions": {}}}
    export_live(tmp_path, state, "Current live truth")
    path = next((tmp_path / "out" / "Approved_Project").glob("*.md"))
    original = path.read_bytes()
    snapshot = live_snapshot("Stale live text")
    snapshot["projects"][0]["threads"][0]["updated_at"] = 1_710_000_000

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

    assert result["failed"] == 1
    assert result["exported"] == 0
    assert "older than verified archive state" in result["warnings"][0]["error"]
    assert path.read_bytes() == original


def test_full_export_cannot_regress_revision_before_a_branch_change(
    tmp_path: Path,
) -> None:
    state = {"chatgpt": {"sessions": {}}}
    export_live(tmp_path, state, "Current live truth", updated_at=100)
    path = next((tmp_path / "out" / "Approved_Project").glob("*.md"))
    original = path.read_bytes()

    same_branch_stale = live_snapshot("Current live truth")
    same_branch_stale["projects"][0]["threads"][0]["updated_at"] = 50
    first = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=Path("-"),
        project_config=tmp_path / "routing.json",
        full=True,
        dry_run=False,
        since_date=None,
        stdin_text=json.dumps(same_branch_stale),
    )

    changed_branch_stale = live_snapshot("Stale changed branch")
    changed_branch_stale["projects"][0]["threads"][0]["updated_at"] = 75
    second = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=Path("-"),
        project_config=tmp_path / "routing.json",
        full=True,
        dry_run=False,
        since_date=None,
        stdin_text=json.dumps(changed_branch_stale),
    )

    assert first["exported"] == 0
    assert first["failed"] == 1
    assert second["exported"] == 0
    assert second["failed"] == 1
    assert state["chatgpt"]["sessions"]["thread-fixture"]["thread_updated_at"] == 100
    assert path.read_bytes() == original


def test_cli_persists_in_progress_before_official_archive_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    source = tmp_path / "export.zip"
    write_official_zip(source, official_conversations())
    state_file = tmp_path / "state.json"
    observed_statuses: list[str] = []
    original_write = chatgpt_module._atomic_write_text

    def observe_write(path: Path, content: str) -> None:
        observed_statuses.append(
            load_state(state_file)["chatgpt"]["official_seed"]["status"]
        )
        original_write(path, content)

    monkeypatch.setattr(chatgpt_module, "_atomic_write_text", observe_write)

    result = run_export(
        "chatgpt",
        full=False,
        dry_run=False,
        base_dir=tmp_path / "archive",
        state_file=state_file,
        chatgpt_input=source,
        chatgpt_project_config=config,
    )

    assert result[0]["exported"] == 1
    assert observed_statuses == ["in_progress"]
    assert load_state(state_file)["chatgpt"]["official_seed"]["status"] == "completed"
    with pytest.raises(ChatGPTExportError, match="permanently closed"):
        run_export(
            "chatgpt",
            full=False,
            dry_run=False,
            base_dir=tmp_path / "archive",
            state_file=state_file,
            chatgpt_input=source,
            chatgpt_project_config=config,
        )


def test_live_envelope_requires_every_approved_project_before_body_parse(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    config_payload["projects"]["g-p-second"] = copy.deepcopy(
        config_payload["projects"][PROJECT_ID]
    )
    config_payload["projects"]["g-p-second"]["label"] = "Second Approved"
    config.write_text(json.dumps(config_payload), encoding="utf-8")
    snapshot = live_snapshot("PRIVATE CANARY MUST NOT APPEAR")
    state = {
        "chatgpt": {
            "sessions": {
                "existing": {
                    "content_hash": "stable",
                    "output_file": "Approved_Project/existing.md",
                }
            }
        }
    }
    archive = tmp_path / "out" / "Approved_Project" / "existing.md"
    archive.parent.mkdir(parents=True)
    archive.write_text("stable archive", encoding="utf-8")
    original_state = copy.deepcopy(state)

    with pytest.raises(ChatGPTExportError) as caught:
        export_chatgpt(
            tmp_path / "out",
            state,
            source_input=Path("-"),
            project_config=config,
            full=False,
            dry_run=False,
            since_date=None,
            stdin_text=json.dumps(snapshot),
        )

    assert "g-p-second" in str(caught.value)
    assert "PRIVATE CANARY" not in str(caught.value)
    assert state == original_state
    assert archive.read_text(encoding="utf-8") == "stable archive"


def test_live_envelope_rejects_duplicate_approved_project(tmp_path: Path) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot()
    snapshot["projects"].append(copy.deepcopy(snapshot["projects"][0]))

    with pytest.raises(ChatGPTExportError, match="duplicate approved project"):
        export_chatgpt(
            tmp_path / "out",
            {"chatgpt": {"sessions": {}}},
            source_input=Path("-"),
            project_config=config,
            full=False,
            dry_run=False,
            since_date=None,
            stdin_text=json.dumps(snapshot),
        )

    assert not (tmp_path / "out").exists()


def test_unapproved_and_projectless_bodies_are_filtered_before_parse(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot()
    snapshot["projects"].append(
        {"project_id": None, "threads": {"private": "PROJECTLESS CANARY"}}
    )

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

    assert result["exported"] == 1
    assert result["ignored"] == 2
    markdown = next((tmp_path / "out").rglob("*.md")).read_text(encoding="utf-8")
    assert "PROJECTLESS CANARY" not in markdown
    assert "must not be inspected" not in markdown


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "unknown_collector"),
        ("non_pinned_returned", -1),
        ("non_pinned_limit", 0),
        ("approved_pinned_threads", -1),
        ("recent_window_saturated", True),
    ],
)
def test_live_envelope_rejects_invalid_discovery_metadata(
    tmp_path: Path, field: str, value: object
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot("DISCOVERY PRIVATE CANARY")
    snapshot["discovery"][field] = value
    state = {"chatgpt": {"sessions": {}}}

    with pytest.raises(ChatGPTExportError) as caught:
        export_chatgpt(
            tmp_path / "out",
            state,
            source_input=Path("-"),
            project_config=config,
            full=False,
            dry_run=False,
            since_date=None,
            stdin_text=json.dumps(snapshot),
        )

    assert "DISCOVERY PRIVATE CANARY" not in str(caught.value)
    assert state == {"chatgpt": {"sessions": {}}}
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("observed_at", [None, "not-a-timestamp", "2026-08-27T12:00:00"])
def test_live_envelope_rejects_invalid_observation_timestamp(
    tmp_path: Path, observed_at: object
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot()
    snapshot["observed_at"] = observed_at

    with pytest.raises(ChatGPTExportError, match="observed_at"):
        export_chatgpt(
            tmp_path / "out",
            {"chatgpt": {"sessions": {}}},
            source_input=Path("-"),
            project_config=config,
            full=False,
            dry_run=False,
            since_date=None,
            stdin_text=json.dumps(snapshot),
        )


def test_live_envelope_rejects_malformed_project_without_parsing_body(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot()
    snapshot["projects"].append(
        {"project_id": "not a safe id", "threads": {"private": "CANARY"}}
    )

    with pytest.raises(ChatGPTExportError) as caught:
        export_chatgpt(
            tmp_path / "out",
            {"chatgpt": {"sessions": {}}},
            source_input=Path("-"),
            project_config=config,
            full=False,
            dry_run=False,
            since_date=None,
            stdin_text=json.dumps(snapshot),
        )

    assert "CANARY" not in str(caught.value)
    assert not (tmp_path / "out").exists()


def test_live_warning_redacts_invalid_thread_identifier(tmp_path: Path) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot()
    snapshot["projects"][0]["threads"][0]["thread_id"] = (
        "PRIVATE THREAD TITLE / MUST NOT LOG"
    )

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

    assert result["failed"] == 1
    assert result["warnings"][0]["thread_id"] == "unknown"
    assert "PRIVATE THREAD TITLE" not in json.dumps(result)


def test_live_envelope_rejects_impossible_approved_pinned_count(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    snapshot = live_snapshot()
    snapshot["discovery"]["approved_pinned_threads"] = 2

    with pytest.raises(ChatGPTExportError, match="approved_pinned_threads"):
        export_chatgpt(
            tmp_path / "out",
            {"chatgpt": {"sessions": {}}},
            source_input=Path("-"),
            project_config=config,
            full=False,
            dry_run=False,
            since_date=None,
            stdin_text=json.dumps(snapshot),
        )


def test_official_write_failure_rolls_back_archive_and_seed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    conversations = official_conversations()[:1]
    second = json.loads(
        json.dumps(conversations[0])
        .replace("thread-fixture", "thread-second")
        .replace("message-user", "message-user-second")
        .replace("message-assistant", "message-assistant-second")
    )
    conversations.append(second)
    source = tmp_path / "official.zip"
    write_official_zip(source, conversations)
    state_file = tmp_path / "state.json"
    original_write = chatgpt_module._atomic_write_text
    writes = 0

    def fail_second_write(path: Path, content: str) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("synthetic second write failure")
        original_write(path, content)

    monkeypatch.setattr(chatgpt_module, "_atomic_write_text", fail_second_write)

    with pytest.raises(OSError, match="second write failure"):
        run_export(
            "chatgpt",
            full=False,
            dry_run=False,
            base_dir=tmp_path / "archive",
            state_file=state_file,
            chatgpt_input=source,
            chatgpt_project_config=config,
        )

    assert list((tmp_path / "archive").rglob("*.md")) == []
    persisted = load_state(state_file)
    assert persisted["chatgpt"]["official_seed"] == {"status": "unused"}
    assert persisted["chatgpt"]["sessions"] == {}


def test_official_batch_reserves_distinct_paths_for_same_date_and_title(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    conversations = official_conversations()[:1]
    conversations.append(
        json.loads(
            json.dumps(conversations[0])
            .replace("thread-fixture", "thread-second")
            .replace("message-user", "message-user-second")
            .replace("message-assistant", "message-assistant-second")
        )
    )
    source = tmp_path / "official.zip"
    write_official_zip(source, conversations)
    state = {"chatgpt": {"sessions": {}}}

    result = export_chatgpt(
        tmp_path / "out",
        state,
        source_input=source,
        project_config=config,
        full=False,
        dry_run=False,
        since_date=None,
    )

    assert result["exported"] == 2
    archive_files = list((tmp_path / "out").rglob("*.md"))
    assert len(archive_files) == 2
    output_files = {
        session["output_file"]
        for session in state["chatgpt"]["sessions"].values()
    }
    assert len(output_files) == 2


def test_final_official_checkpoint_failure_rolls_back_files_and_seed(
    tmp_path: Path,
) -> None:
    config = tmp_path / "routing.json"
    write_config(config)
    source = tmp_path / "official.zip"
    write_official_zip(source, official_conversations())
    state_file = tmp_path / "state.json"
    state = {"chatgpt": {"sessions": {}}}
    checkpoints = 0

    def fail_completed_checkpoint(current: dict[str, object]) -> None:
        nonlocal checkpoints
        checkpoints += 1
        if checkpoints == 2:
            raise OSError("synthetic completed checkpoint failure")
        save_state(current, state_file)

    with pytest.raises(OSError, match="completed checkpoint failure"):
        export_chatgpt(
            tmp_path / "out",
            state,
            source_input=source,
            project_config=config,
            full=False,
            dry_run=False,
            since_date=None,
            checkpoint_state=fail_completed_checkpoint,
        )

    assert list((tmp_path / "out").rglob("*.md")) == []
    persisted = load_state(state_file)
    assert persisted["chatgpt"]["official_seed"] == {"status": "unused"}
    assert persisted["chatgpt"]["sessions"] == {}
