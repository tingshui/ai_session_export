from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ai_session_export.chatgpt_incremental import (
    ChatGPTIncrementalError,
    CollectedTail,
    ThreadPlan,
    apply_incremental_batch,
    apply_incremental_payload,
    activate_live_authority,
    collect_incremental_tail,
    collect_window_tail,
    plan_window_threads,
    plan_historical_backfill,
    plan_incremental_threads,
)
import pytest

from ai_session_export.archive import content_sha256, parse_marked_markdown
from ai_session_export.markdown import render_markdown
from ai_session_export.models import MessageTurn, SessionRecord
from ai_session_export.state import save_state


PROJECT_ID = "g-p-approved"
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_plan_reads_only_changed_threads_and_separates_backfill() -> None:
    source_state = {
        "bootstrap_completed_at": "2026-08-27T12:00:00Z",
        "sessions": {
            "unchanged": {
                "thread_updated_at": 200,
                "output_file": "Approved/unchanged.md",
                "messages": [
                    {
                        "message_id": "anchor-unchanged",
                        "role": "assistant",
                        "created_at": 190,
                        "content_sha256": "a" * 64,
                        "complete": True,
                    }
                ],
            },
            "changed": {
                "thread_updated_at": 200,
                "output_file": "Approved/changed.md",
                "messages": [
                    {
                        "message_id": "anchor-changed",
                        "role": "assistant",
                        "created_at": 190,
                        "content_sha256": "b" * 64,
                        "complete": True,
                    }
                ],
            },
        },
    }
    summaries = [
        {
            "id": "unchanged",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Unchanged",
            "createdAt": 100,
            "updatedAt": 200,
        },
        {
            "id": "changed",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Changed",
            "createdAt": 100,
            "updatedAt": 201,
        },
        {
            "id": "new-after-bootstrap",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "New",
            "createdAt": 1_788_000_001,
            "updatedAt": 1_788_000_002,
        },
        {
            "id": "historical-gap",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Historical gap",
            "createdAt": 100,
            "updatedAt": 1_788_000_003,
        },
        {
            "id": "unapproved",
            "kind": "chatgpt",
            "projectId": "g-p-unapproved",
            "title": "Unapproved",
            "createdAt": 1_788_000_001,
            "updatedAt": 1_788_000_002,
        },
        {
            "id": "codex-task",
            "kind": "codex",
            "projectId": PROJECT_ID,
            "title": "Codex",
            "createdAt": 1_788_000_001,
            "updatedAt": 1_788_000_002,
        },
    ]

    result = plan_incremental_threads(
        summaries,
        source_state,
        {PROJECT_ID: "Approved"},
    )

    assert [(plan.thread_id, plan.mode) for plan in result.changed] == [
        ("changed", "incremental_tail"),
        ("new-after-bootstrap", "new_thread"),
    ]
    assert result.changed[0].anchor_message_id == "anchor-changed"
    assert result.changed[0].anchor_sha256 == "b" * 64
    assert result.changed[1].anchor_message_id is None
    assert [item.thread_id for item in result.needs_backfill] == ["historical-gap"]
    assert result.unchanged == 1
    assert result.ignored == 2


def test_plan_uses_full_history_reconciliation_for_legacy_import() -> None:
    source_state = {
        "sessions": {
            "legacy": {
                "thread_updated_at": 200,
                "output_file": "Approved/legacy.md",
                "legacy_import": {"imported_at": "2026-08-28T12:00:00Z"},
                "messages": [{
                    "message_id": "synthetic-anchor",
                    "role": "assistant",
                    "created_at": 190,
                    "content_sha256": "a" * 64,
                    "complete": True,
                }],
            }
        }
    }

    result = plan_incremental_threads(
        [{
            "id": "legacy",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Legacy",
            "createdAt": 100,
            "updatedAt": 201,
        }],
        source_state,
        {PROJECT_ID: "Approved"},
    )

    assert len(result.changed) == 1
    assert result.changed[0].mode == "legacy_reconcile"
    assert result.changed[0].anchor_message_id is None
    assert result.changed[0].anchor_sha256 is None


def test_plan_uses_native_live_anchor_after_legacy_archive_id_preservation() -> None:
    source_state = {
        "sessions": {
            "legacy": {
                "thread_updated_at": 200,
                "output_file": "Approved/legacy.md",
                "legacy_import": {
                    "imported_at": "2026-08-28T12:00:00Z",
                    "native_ids_reconciled": True,
                },
                "live_anchor": {
                    "message_id": "native-live-anchor",
                    "content_sha256": "b" * 64,
                },
                "messages": [{
                    "message_id": "synthetic-archive-anchor",
                    "role": "assistant",
                    "created_at": 190,
                    "content_sha256": "a" * 64,
                    "complete": True,
                }],
            }
        }
    }

    result = plan_incremental_threads(
        [{
            "id": "legacy",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Legacy",
            "createdAt": 100,
            "updatedAt": 201,
        }],
        source_state,
        {PROJECT_ID: "Approved"},
    )

    assert result.changed[0].mode == "incremental_tail"
    assert result.changed[0].anchor_message_id == "native-live-anchor"
    assert result.changed[0].anchor_sha256 == "b" * 64


def test_collect_stops_on_exact_anchor_without_reading_older_pages() -> None:
    plan = ThreadPlan(
        thread_id="changed",
        project_id=PROJECT_ID,
        project_label="Approved",
        title="Changed",
        created_at=100,
        updated_at=201,
        mode="incremental_tail",
        anchor_message_id="anchor",
        anchor_sha256="a" * 64,
        output_file="Approved/changed.md",
    )
    calls: list[str | None] = []

    def read_page(thread_id: str, cursor: str | None) -> dict[str, object]:
        calls.append(cursor)
        assert thread_id == "changed"
        if cursor is None:
            return {
                "thread_id": "changed",
                "cursor_in": None,
                "has_more": True,
                "cursor_out": "older-page",
                "messages": [
                    {
                        "id": "assistant-new",
                        "role": "assistant",
                        "content": "new answer",
                        "created_at": 201,
                        "complete": True,
                    },
                    {
                        "id": "user-new",
                        "role": "user",
                        "content": "new question",
                        "created_at": 200,
                        "complete": True,
                    },
                    {
                        "id": "anchor",
                        "role": "assistant",
                        "content": "old answer",
                        "content_sha256": "a" * 64,
                        "created_at": 190,
                        "complete": True,
                    },
                ],
            }
        raise AssertionError("collector read beyond the proven anchor")

    result = collect_incremental_tail(plan, read_page)

    assert calls == [None]
    assert result.terminal_reason == "anchor"
    assert result.pages_read == 1
    assert [(message.message_id, message.content) for message in result.messages] == [
        ("user-new", "new question"),
        ("assistant-new", "new answer"),
    ]


def test_collect_fails_closed_on_incomplete_user_message() -> None:
    plan = ThreadPlan(
        "new", PROJECT_ID, "Approved", "New", 100, 101, "new_thread", None, None, None
    )

    def read_page(_thread_id: str, cursor: str | None) -> dict[str, object]:
        return {
            "thread_id": "new",
            "cursor_in": cursor,
            "has_more": False,
            "cursor_out": None,
            "messages": [
                {
                    "id": "user-truncated",
                    "role": "user",
                    "content": "prefix…7 tokens truncated…suffix",
                    "complete": True,
                }
            ],
        }

    with pytest.raises(ChatGPTIncrementalError, match="incomplete user content"):
        collect_incremental_tail(plan, read_page)


def test_collect_preserves_empty_incomplete_assistant_with_marker() -> None:
    plan = ThreadPlan(
        "new", PROJECT_ID, "Approved", "New", 100, 101, "new_thread", None, None, None
    )

    def read_page(_thread_id: str, cursor: str | None) -> dict[str, object]:
        return {
            "thread_id": "new",
            "cursor_in": cursor,
            "has_more": False,
            "cursor_out": None,
            "messages": [{
                "id": "assistant-empty",
                "role": "assistant",
                "content": "",
                "complete": False,
            }],
        }

    result = collect_incremental_tail(plan, read_page)

    assert result.messages[0].complete is False
    assert result.messages[0].content == (
        "[INCOMPLETE ASSISTANT CONTENT: source was truncated]"
    )


def test_collect_normalizes_seconds_or_milliseconds_without_double_scaling() -> None:
    plan = ThreadPlan(
        "new", PROJECT_ID, "Approved", "New", 100, 101, "new_thread", None, None, None
    )

    def read_page(_thread_id: str, cursor: str | None) -> dict[str, object]:
        return {
            "thread_id": "new",
            "cursor_in": cursor,
            "has_more": False,
            "cursor_out": None,
            "messages": [
                {"id": "ms", "role": "assistant", "content": "later",
                 "created_at": 1_788_000_002_000},
                {"id": "seconds", "role": "user", "content": "earlier",
                 "created_at": 1_788_000_001},
            ],
        }

    result = collect_incremental_tail(plan, read_page)

    assert [message.time_created for message in result.messages] == [
        1_788_000_001_000,
        1_788_000_002_000,
    ]


def test_collect_fails_closed_on_cursor_loop() -> None:
    plan = ThreadPlan(
        "changed", PROJECT_ID, "Approved", "Changed", 100, 201,
        "incremental_tail", "anchor", "a" * 64, "Approved/changed.md"
    )

    def read_page(_thread_id: str, cursor: str | None) -> dict[str, object]:
        return {
            "thread_id": "changed",
            "cursor_in": cursor,
            "has_more": True,
            "cursor_out": "same-cursor",
            "messages": [],
        }

    with pytest.raises(ChatGPTIncrementalError, match="pagination cursor loop"):
        collect_incremental_tail(plan, read_page)


def test_apply_rewrites_same_markdown_then_advances_checkpoint(tmp_path) -> None:
    output_dir = tmp_path / "chatgpt"
    output_path = output_dir / "Approved" / "changed.md"
    old_messages = [
        MessageTurn("user", "old question", 100_000, message_id="old-user"),
        MessageTurn("assistant", "old answer", 110_000, message_id="anchor"),
    ]
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        render_markdown(SessionRecord("chatgpt", "changed", "Changed", "1970-01-01", old_messages)),
        encoding="utf-8",
    )
    source_state = {
        "sessions": {
            "changed": {
                "project_id": PROJECT_ID,
                "project_label": "Approved",
                "output_file": "Approved/changed.md",
                "thread_updated_at": 200,
                "messages": [
                    {
                        "message_id": message.message_id,
                        "role": message.role,
                        "created_at": message.time_created,
                        "content_sha256": content_sha256(message.content),
                        "complete": message.complete,
                    }
                    for message in old_messages
                ],
            }
        }
    }
    plan = ThreadPlan(
        "changed", PROJECT_ID, "Approved", "Changed", 100, 201,
        "incremental_tail", "anchor", content_sha256("old answer"), "Approved/changed.md"
    )
    tail = CollectedTail(
        [
            MessageTurn("user", "new question", 200_000, message_id="new-user"),
            MessageTurn("assistant", "new answer", 201_000, message_id="new-assistant"),
        ],
        1,
        "anchor",
    )
    checkpoint_observations: list[tuple[int, list[str]]] = []

    def checkpoint(staged_state: dict[str, object]) -> None:
        on_disk = parse_marked_markdown(output_path)
        session = staged_state["sessions"]["changed"]  # type: ignore[index]
        checkpoint_observations.append(
            (session["thread_updated_at"], [turn.message_id for turn in on_disk])  # type: ignore[index]
        )

    result = apply_incremental_batch(
        output_dir,
        source_state,
        [(plan, tail)],
        observed_at="2026-08-28T12:00:00Z",
        checkpoint_state=checkpoint,
    )

    assert result.exported == 1
    assert output_path == result.output_paths[0]
    assert [turn.message_id for turn in parse_marked_markdown(output_path)] == [
        "old-user", "anchor", "new-user", "new-assistant"
    ]
    assert checkpoint_observations == [
        (201, ["old-user", "anchor", "new-user", "new-assistant"])
    ]
    assert source_state["sessions"]["changed"]["thread_updated_at"] == 201
    assert source_state["sessions"]["changed"]["coverage"] == "full_history"
    assert source_state["archive_generation"] == 1
    assert source_state["observer_handoffs"] == [result.observer_handoff]
    assert result.observer_handoff["previous_generation"] == 0
    assert result.observer_handoff["archive_generation"] == 1
    assert result.observer_handoff["zero_user_delta"] is False
    assert result.observer_handoff["changes"][0]["start_after"] == {
        "message_id": "anchor",
        "content_sha256": content_sha256("old answer"),
    }
    assert result.observer_handoff["changes"][0]["user_messages"] == [
        {
            "message_id": "new-user",
            "content_sha256": content_sha256("new question"),
            "created_at": 200_000,
            "change": "new",
        }
    ]
    assert "new question" not in json.dumps(
        result.observer_handoff, ensure_ascii=False
    )


def test_apply_rolls_back_markdown_and_state_when_checkpoint_fails(tmp_path) -> None:
    output_dir = tmp_path / "chatgpt"
    output_path = output_dir / "Approved" / "changed.md"
    old_message = MessageTurn("assistant", "old answer", 110_000, message_id="anchor")
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        render_markdown(
            SessionRecord("chatgpt", "changed", "Changed", "1970-01-01", [old_message])
        ),
        encoding="utf-8",
    )
    before_bytes = output_path.read_bytes()
    source_state = {
        "sessions": {
            "changed": {
                "project_id": PROJECT_ID,
                "project_label": "Approved",
                "output_file": "Approved/changed.md",
                "thread_updated_at": 200,
                "messages": [{
                    "message_id": "anchor",
                    "role": "assistant",
                    "created_at": 110_000,
                    "content_sha256": content_sha256("old answer"),
                    "complete": True,
                }],
            }
        }
    }
    before_state = {"sessions": {"changed": dict(source_state["sessions"]["changed"])}}
    plan = ThreadPlan(
        "changed", PROJECT_ID, "Approved", "Changed", 100, 201,
        "incremental_tail", "anchor", content_sha256("old answer"), "Approved/changed.md"
    )
    tail = CollectedTail(
        [MessageTurn("user", "new", 200_000, message_id="new-user")], 1, "anchor"
    )

    def broken_checkpoint(_staged_state: dict[str, object]) -> None:
        raise OSError("synthetic checkpoint failure")

    with pytest.raises(OSError, match="synthetic checkpoint failure"):
        apply_incremental_batch(
            output_dir,
            source_state,
            [(plan, tail)],
            observed_at="2026-08-28T12:00:00Z",
            checkpoint_state=broken_checkpoint,
        )

    assert output_path.read_bytes() == before_bytes
    assert source_state == before_state


def test_payload_orchestrator_reads_only_planned_threads(tmp_path) -> None:
    source_state = {
        "bootstrap_completed_at": "2026-08-27T12:00:00Z",
        "sessions": {
            "unchanged": {
                "thread_updated_at": 200,
                "output_file": "Approved/unchanged.md",
                "messages": [{
                    "message_id": "anchor-unchanged",
                    "role": "assistant",
                    "created_at": 190_000,
                    "content_sha256": "a" * 64,
                    "complete": True,
                }],
            }
        },
    }
    payload = {
        "schema_version": 2,
        "observed_at": "2026-08-28T12:00:00Z",
        "discovery": {
            "non_pinned_returned": 50,
            "non_pinned_limit": 50,
            "recent_window_saturated": True,
        },
        "summaries": [
            {"id": "unchanged", "kind": "chatgpt", "projectId": PROJECT_ID,
             "title": "Unchanged", "createdAt": 100, "updatedAt": 200},
            {"id": "new", "kind": "chatgpt", "projectId": PROJECT_ID,
             "title": "New", "createdAt": 1_788_000_001, "updatedAt": 1_788_000_002},
        ],
        "pages": {
            "new": [{
                "thread_id": "new",
                "cursor_in": None,
                "has_more": False,
                "cursor_out": None,
                "messages": [{
                    "id": "new-user",
                    "role": "user",
                    "content": "hello",
                    "created_at": 1_788_000_001,
                    "complete": True,
                }],
            }]
        },
    }

    result = apply_incremental_payload(
        tmp_path / "chatgpt", source_state, {PROJECT_ID: "Approved"}, payload
    )

    assert result["exported"] == 1
    assert result["failed"] == 0
    assert result["unchanged"] == 1
    assert result["backfill_pending"] == 0
    assert result["discovery_saturated"] is True
    assert "new" in source_state["sessions"]
    assert result["observer_handoff"]["archive_generation"] == 1
    assert result["observer_handoff"]["status"] == "success"
    assert result["observer_handoff"]["zero_user_delta"] is False
    assert result["observer_handoff"]["discovery_saturated"] is True
    assert "hello" not in json.dumps(
        result["observer_handoff"], ensure_ascii=False
    )


def test_payload_does_not_partially_apply_when_any_page_chain_is_incomplete(
    tmp_path,
) -> None:
    output_dir = tmp_path / "chatgpt"
    sessions = {}
    before_files = {}
    for thread_id in ("complete", "incomplete"):
        output_file = f"Approved/{thread_id}.md"
        output_path = output_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anchor = MessageTurn(
            "assistant",
            f"old answer {thread_id}",
            110_000,
            message_id=f"anchor-{thread_id}",
        )
        output_path.write_text(
            render_markdown(
                SessionRecord(
                    "chatgpt",
                    thread_id,
                    thread_id.title(),
                    "1970-01-01",
                    [anchor],
                )
            ),
            encoding="utf-8",
        )
        before_files[thread_id] = output_path.read_bytes()
        sessions[thread_id] = {
            "project_id": PROJECT_ID,
            "project_label": "Approved",
            "output_file": output_file,
            "thread_updated_at": 200,
            "messages": [{
                "message_id": anchor.message_id,
                "role": anchor.role,
                "created_at": anchor.time_created,
                "content_sha256": content_sha256(anchor.content),
                "complete": True,
            }],
        }

    source_state = {"sessions": sessions}
    before_state = json.loads(json.dumps(source_state))
    payload = {
        "schema_version": 2,
        "observed_at": "2026-08-30T12:00:00Z",
        "discovery": {
            "non_pinned_returned": 2,
            "non_pinned_limit": 50,
            "recent_window_saturated": False,
        },
        "summaries": [
            {
                "id": thread_id,
                "kind": "chatgpt",
                "projectId": PROJECT_ID,
                "title": thread_id.title(),
                "createdAt": 100,
                "updatedAt": 201,
            }
            for thread_id in ("complete", "incomplete")
        ],
        "pages": {
            "complete": [{
                "thread_id": "complete",
                "cursor_in": None,
                "cursor_out": "older-complete",
                "has_more": True,
                "messages": [{
                    "id": "new-complete",
                    "role": "user",
                    "content": "new complete message",
                    "created_at": 200_000,
                    "complete": True,
                }],
            }, {
                "thread_id": "complete",
                "cursor_in": "older-complete",
                "cursor_out": None,
                "has_more": False,
                "messages": [{
                    "id": "anchor-complete",
                    "role": "assistant",
                    "content": "old answer complete",
                    "created_at": 110_000,
                    "complete": True,
                }],
            }],
            "incomplete": [{
                "thread_id": "incomplete",
                "cursor_in": None,
                "cursor_out": "older-incomplete",
                "has_more": True,
                "messages": [{
                    "id": "new-incomplete",
                    "role": "user",
                    "content": "new incomplete message",
                    "created_at": 200_000,
                    "complete": True,
                }],
            }],
        },
    }

    result = apply_incremental_payload(
        output_dir,
        source_state,
        {PROJECT_ID: "Approved"},
        payload,
    )

    assert result["exported"] == 0
    assert result["failed"] == 1
    assert result["output_paths"] == []
    assert result["observer_handoff"] is None
    assert source_state == before_state
    for thread_id in ("complete", "incomplete"):
        path = output_dir / f"Approved/{thread_id}.md"
        assert path.read_bytes() == before_files[thread_id]


def test_legacy_reconcile_preserves_archive_ids_and_emits_only_live_tail(
    tmp_path,
) -> None:
    output_dir = tmp_path / "chatgpt"
    output_path = output_dir / "Approved/legacy.md"
    output_path.parent.mkdir(parents=True)
    legacy_messages = [
        MessageTurn("user", "old question", 100_000, message_id="synthetic-user"),
        MessageTurn(
            "assistant", "old answer", 101_000, message_id="synthetic-assistant"
        ),
    ]
    output_path.write_text(
        render_markdown(
            SessionRecord(
                "chatgpt", "legacy", "Legacy", "1970-01-01", legacy_messages
            )
        ),
        encoding="utf-8",
    )
    source_state = {
        "sessions": {
            "legacy": {
                "project_id": PROJECT_ID,
                "project_label": "Approved",
                "output_file": "Approved/legacy.md",
                "thread_updated_at": 200,
                "coverage": "full_history",
                "messages": [
                    {
                        "message_id": message.message_id,
                        "role": message.role,
                        "created_at": message.time_created,
                        "content_sha256": content_sha256(message.content),
                        "complete": True,
                    }
                    for message in legacy_messages
                ],
                "legacy_import": {
                    "imported_at": "2026-08-28T12:00:00Z",
                    "message_count": 2,
                },
            }
        }
    }
    payload = {
        "schema_version": 2,
        "observed_at": "2026-08-30T12:00:00Z",
        "discovery": {
            "non_pinned_returned": 1,
            "non_pinned_limit": 50,
            "recent_window_saturated": False,
        },
        "summaries": [{
            "id": "legacy",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Legacy",
            "createdAt": 100,
            "updatedAt": 201,
        }],
        "pages": {
            "legacy": [{
                "thread_id": "legacy",
                "cursor_in": None,
                "cursor_out": None,
                "has_more": False,
                "messages": [
                    {
                        "id": "live-new-assistant",
                        "role": "assistant",
                        "content": "new answer",
                        "created_at": 301_000,
                        "complete": True,
                    },
                    {
                        "id": "live-new-user",
                        "role": "user",
                        "content": "new question",
                        "created_at": 300_000,
                        "complete": True,
                    },
                    {
                        "id": "live-old-assistant",
                        "role": "assistant",
                        "content": "old answer",
                        "created_at": 201_000,
                        "complete": True,
                    },
                    {
                        "id": "live-old-user",
                        "role": "user",
                        "content": "old question",
                        "created_at": 200_000,
                        "complete": True,
                    },
                ],
            }]
        },
    }

    result = apply_incremental_payload(
        output_dir,
        source_state,
        {PROJECT_ID: "Approved"},
        payload,
    )

    assert result["failed"] == 0
    assert result["exported"] == 1
    assert [
        message.message_id for message in parse_marked_markdown(output_path)
    ] == [
        "synthetic-user",
        "synthetic-assistant",
        "live-new-user",
        "live-new-assistant",
    ]
    stored = source_state["sessions"]["legacy"]
    assert stored["thread_updated_at"] == 201
    assert stored["legacy_import"]["native_ids_reconciled"] is True
    assert stored["live_anchor"] == {
        "message_id": "live-new-assistant",
        "content_sha256": content_sha256("new answer"),
    }
    change = result["observer_handoff"]["changes"][0]
    assert change["start_after"] == {
        "message_id": "synthetic-assistant",
        "content_sha256": content_sha256("old answer"),
    }
    assert change["user_messages"] == [{
        "message_id": "live-new-user",
        "content_sha256": content_sha256("new question"),
        "created_at": 300_000_000,
        "change": "new",
    }]


def test_legacy_reconcile_live_authority_rewrites_divergent_branch_without_replaying_history(
    tmp_path,
) -> None:
    output_dir = tmp_path / "chatgpt"
    output_path = output_dir / "Approved/divergent.md"
    output_path.parent.mkdir(parents=True)
    legacy_messages = [
        MessageTurn("user", "shared question", 100_000, message_id="synthetic-user"),
        MessageTurn("assistant", "legacy answer", 101_000, message_id="synthetic-assistant"),
    ]
    output_path.write_text(
        render_markdown(
            SessionRecord(
                "chatgpt", "divergent", "Divergent", "1970-01-01", legacy_messages
            )
        ),
        encoding="utf-8",
    )
    source_state = {
        "live_authority_started_at": "2026-08-28T14:17:44Z",
        "sessions": {
            "divergent": {
                "project_id": PROJECT_ID,
                "project_label": "Approved",
                "output_file": "Approved/divergent.md",
                "thread_updated_at": 200,
                "coverage": "full_history",
                "messages": [
                    {
                        "message_id": message.message_id,
                        "role": message.role,
                        "created_at": message.time_created,
                        "content_sha256": content_sha256(message.content),
                        "complete": True,
                    }
                    for message in legacy_messages
                ],
                "legacy_import": {
                    "imported_at": "2026-08-28T12:00:00Z",
                    "message_count": 2,
                },
            }
        },
    }
    payload = {
        "schema_version": 2,
        "observed_at": "2026-08-30T12:00:00Z",
        "summaries": [{
            "id": "divergent",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Divergent",
            "createdAt": 100,
            "updatedAt": 201,
        }],
        "pages": {
            "divergent": [{
                "thread_id": "divergent",
                "cursor_in": None,
                "cursor_out": None,
                "has_more": False,
                "messages": [
                    {
                        "id": "live-new-assistant",
                        "role": "assistant",
                        "content": "new answer",
                        "created_at": 1_788_061_621,
                        "complete": True,
                    },
                    {
                        "id": "live-new-user",
                        "role": "user",
                        "content": "new question",
                        "created_at": 1_788_061_620,
                        "complete": True,
                    },
                    {
                        "id": "live-edited-assistant",
                        "role": "assistant",
                        "content": "current answer",
                        "created_at": 101,
                        "complete": True,
                    },
                    {
                        "id": "live-old-user",
                        "role": "user",
                        "content": "shared question",
                        "created_at": 100,
                        "complete": True,
                    },
                ],
            }]
        },
    }

    result = apply_incremental_payload(
        output_dir,
        source_state,
        {PROJECT_ID: "Approved"},
        payload,
    )

    assert result["failed"] == 0
    assert [message.message_id for message in parse_marked_markdown(output_path)] == [
        "synthetic-user",
        "live-edited-assistant",
        "live-new-user",
        "live-new-assistant",
    ]
    stored = source_state["sessions"]["divergent"]
    assert stored["legacy_import"]["reconciliation"] == "live_authority_branch"
    assert stored["legacy_import"]["legacy_messages_removed"] == 1
    assert stored["live_anchor"]["message_id"] == "live-new-assistant"
    change = result["observer_handoff"]["changes"][0]
    assert change["start_after"] == {
        "message_id": "live-edited-assistant",
        "content_sha256": content_sha256("current answer"),
    }
    assert change["user_messages"] == [{
        "message_id": "live-new-user",
        "content_sha256": content_sha256("new question"),
        "created_at": 1_788_061_620_000,
        "change": "new",
    }]


def test_apply_emits_contiguous_zero_delta_generations(tmp_path) -> None:
    source_state = {"sessions": {}}

    first = apply_incremental_batch(
        tmp_path / "chatgpt",
        source_state,
        [],
        observed_at="2026-08-28T12:00:00Z",
    )
    second = apply_incremental_batch(
        tmp_path / "chatgpt",
        source_state,
        [],
        observed_at="2026-08-29T12:00:00Z",
        producer_trigger="scheduled_automation",
    )

    assert first.observer_handoff["archive_generation"] == 1
    assert first.observer_handoff["previous_generation"] == 0
    assert first.observer_handoff["zero_user_delta"] is True
    assert first.observer_handoff["changes"] == []
    assert first.observer_handoff["producer_trigger"] == "manual_validation"
    assert second.observer_handoff["archive_generation"] == 2
    assert second.observer_handoff["previous_generation"] == 1
    assert second.observer_handoff["producer_trigger"] == "scheduled_automation"
    assert source_state["archive_generation"] == 2


def test_handoff_keeps_each_changed_thread_anchor_and_user_delta(tmp_path) -> None:
    output_dir = tmp_path / "chatgpt"
    source_state = {"sessions": {}}
    batches = []

    for index, thread_id in enumerate(("health-thread", "social-thread"), start=1):
        output_file = f"Project-{index}/{thread_id}.md"
        output_path = output_dir / output_file
        output_path.parent.mkdir(parents=True)
        anchor = MessageTurn(
            "assistant",
            f"prior answer {index}",
            100_000 + index,
            message_id=f"anchor-{index}",
        )
        output_path.write_text(
            render_markdown(
                SessionRecord(
                    "chatgpt",
                    thread_id,
                    f"Thread {index}",
                    "1970-01-01",
                    [anchor],
                )
            ),
            encoding="utf-8",
        )
        source_state["sessions"][thread_id] = {
            "project_id": f"g-p-{index}",
            "project_label": f"Project-{index}",
            "output_file": output_file,
            "thread_updated_at": 100 + index,
            "messages": [
                {
                    "message_id": anchor.message_id,
                    "role": anchor.role,
                    "created_at": anchor.time_created,
                    "content_sha256": content_sha256(anchor.content),
                    "complete": True,
                }
            ],
        }
        plan = ThreadPlan(
            thread_id,
            f"g-p-{index}",
            f"Project-{index}",
            f"Thread {index}",
            1,
            200 + index,
            "incremental_tail",
            anchor.message_id,
            content_sha256(anchor.content),
            output_file,
        )
        tail = CollectedTail(
            [
                MessageTurn(
                    "user",
                    f"new user message {index}",
                    200_000 + index,
                    message_id=f"new-user-{index}",
                ),
                MessageTurn(
                    "assistant",
                    f"new assistant message {index}",
                    201_000 + index,
                    message_id=f"new-assistant-{index}",
                ),
            ],
            2,
            "anchor",
        )
        batches.append((plan, tail))

    result = apply_incremental_batch(
        output_dir,
        source_state,
        batches,
        observed_at="2026-08-28T12:00:00Z",
    )

    changes = {
        change["thread_id"]: change
        for change in result.observer_handoff["changes"]
    }
    assert result.exported == 2
    assert result.observer_handoff["zero_user_delta"] is False
    assert set(changes) == {"health-thread", "social-thread"}
    for index, thread_id in enumerate(("health-thread", "social-thread"), start=1):
        assert changes[thread_id]["start_after"] == {
            "message_id": f"anchor-{index}",
            "content_sha256": content_sha256(f"prior answer {index}"),
        }
        assert changes[thread_id]["user_messages"] == [
            {
                "message_id": f"new-user-{index}",
                "content_sha256": content_sha256(f"new user message {index}"),
                "created_at": 200_000 + index,
                "change": "new",
            }
        ]


def test_incremental_cli_plan_prints_only_metadata_read_plan(tmp_path) -> None:
    state_file = tmp_path / "state.json"
    config_file = tmp_path / "routing.json"
    save_state(
        {
            "chatgpt": {
                "bootstrap_completed_at": "2026-08-27T12:00:00Z",
                "sessions": {},
                "official_seed": {"status": "completed"},
            }
        },
        state_file,
    )
    config_file.write_text(
        json.dumps({
            "schema_version": 1,
            "projects": {PROJECT_ID: {"label": "Approved"}},
        }),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 2,
        "observed_at": "2026-08-28T12:00:00Z",
        "summaries": [{
            "id": "new",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "New",
            "createdAt": 1_788_000_001,
            "updatedAt": 1_788_000_002,
        }],
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "export_chatgpt_incremental.py"),
            "plan",
            "--state-file", str(state_file),
            "--chatgpt-project-config", str(config_file),
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["changed"] == [{
        "thread_id": "new", "mode": "new_thread", "anchor_message_id": None
    }]
    assert "New" not in completed.stdout


def test_plan_can_explicitly_include_projectless_chatgpt_as_personal() -> None:
    source_state = {
        "live_authority_started_at": "2026-08-27T12:00:00Z",
        "sessions": {},
    }
    summaries = [{
        "id": "personal-chat",
        "kind": "chatgpt",
        "projectId": None,
        "title": "Private",
        "updatedAt": 1_788_000_002,
    }]

    result = plan_incremental_threads(
        summaries, source_state, {"personal": "Personal"}
    )

    assert [(item.thread_id, item.project_id, item.mode) for item in result.changed] == [
        ("personal-chat", "personal", "new_thread")
    ]


def test_window_plan_uses_exclusive_upper_bound() -> None:
    summaries = [
        {
            "id": "before",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Before",
            "updatedAt": 99,
        },
        {
            "id": "start",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Start",
            "updatedAt": 100,
        },
        {
            "id": "inside",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "Inside",
            "updatedAt": 199,
        },
        {
            "id": "end",
            "kind": "chatgpt",
            "projectId": PROJECT_ID,
            "title": "End",
            "updatedAt": 200,
        },
    ]

    result = plan_window_threads(
        summaries,
        {PROJECT_ID: "01_Approved"},
        since_at=100,
        until_at=200,
    )

    assert [item.thread_id for item in result.changed] == ["start", "inside"]
    assert result.ignored == 2


def test_window_plan_rejects_empty_or_reversed_window() -> None:
    with pytest.raises(
        ChatGPTIncrementalError,
        match="until_at must be later than since_at",
    ):
        plan_window_threads(
            [],
            {PROJECT_ID: "01_Approved"},
            since_at=200,
            until_at=200,
        )


def test_live_authority_activation_closes_official_without_claiming_bootstrap() -> None:
    source_state = {"sessions": {}, "official_seed": {"status": "unused"}}

    activate_live_authority(source_state, "2026-08-28T12:00:00Z")

    assert source_state["live_authority_started_at"] == "2026-08-28T12:00:00Z"
    assert source_state["official_seed"] == {
        "status": "completed",
        "completed_at": "2026-08-28T12:00:00Z",
        "authority": "live_only",
    }
    assert "bootstrap_completed_at" not in source_state


def test_historical_backfill_is_explicit_and_separate_from_daily_plan() -> None:
    source_state = {
        "live_authority_started_at": "2026-08-28T12:00:00Z",
        "sessions": {},
    }
    summaries = [{
        "id": "historical",
        "kind": "chatgpt",
        "projectId": PROJECT_ID,
        "title": "Historical",
        "updatedAt": 100,
    }]

    daily = plan_incremental_threads(
        summaries, source_state, {PROJECT_ID: "Approved"}
    )
    backfill = plan_historical_backfill(
        summaries, source_state, {PROJECT_ID: "Approved"}
    )

    assert daily.changed == []
    assert [item.thread_id for item in daily.needs_backfill] == ["historical"]
    assert [(item.thread_id, item.mode) for item in backfill.changed] == [
        ("historical", "historical_backfill")
    ]


def test_three_day_window_stops_before_older_messages_without_reading_older_page() -> None:
    summaries = [{
        "id": "recent",
        "kind": "chatgpt",
        "projectId": PROJECT_ID,
        "title": "Recent",
        "updatedAt": 300,
    }]
    plan = plan_window_threads(
        summaries, {PROJECT_ID: "Approved"}, since_at=200
    ).changed[0]
    calls: list[str | None] = []

    def read_page(_thread_id: str, cursor: str | None) -> dict[str, object]:
        calls.append(cursor)
        if cursor is not None:
            raise AssertionError("window collector read before the cutoff")
        return {
            "thread_id": "recent",
            "cursor_in": None,
            "has_more": True,
            "cursor_out": "older",
            "messages": [
                {"id": "new-assistant", "role": "assistant", "content": "answer",
                 "created_at": 250},
                {"id": "new-user", "role": "user", "content": "question",
                 "created_at": 220},
                {"id": "old-user", "role": "user", "content": "old",
                 "created_at": 199},
            ],
        }

    result = collect_window_tail(plan, read_page, since_at=200)

    assert calls == [None]
    assert result.terminal_reason == "cutoff"
    assert [message.message_id for message in result.messages] == [
        "new-user", "new-assistant"
    ]
