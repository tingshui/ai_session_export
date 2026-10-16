from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from ai_session_export.archive import content_sha256
from ai_session_export.chatgpt_app_protocol import (
    archive_thread_request,
    discovery_request,
    next_cursor,
    normalize_discovery,
    normalize_page,
    page_is_terminal,
    page_request,
    parse_archive_ack,
)
from ai_session_export.chatgpt_incremental import (
    ChatGPTIncrementalError,
    ThreadPlan,
    collect_incremental_tail,
)
from ai_session_export.markdown import render_markdown
from ai_session_export.models import MessageTurn, SessionRecord


def _plan(mode: str = "incremental_tail") -> ThreadPlan:
    return ThreadPlan(
        thread_id="thread-1",
        project_id="g-p-project",
        project_label="self",
        title="Title",
        created_at=1,
        updated_at=2,
        mode=mode,
        anchor_message_id="user-anchor" if mode == "incremental_tail" else None,
        anchor_sha256=(
            hashlib.sha256(b"question").hexdigest()
            if mode == "incremental_tail"
            else None
        ),
        output_file=None,
    )


def test_protocol_generates_fixed_tool_requests() -> None:
    assert discovery_request() == {
        "schema_version": 1,
        "type": "tool_request",
        "tool": "list_threads",
        "arguments": {"limit": 50},
    }
    assert page_request("thread-1", "cursor-1") == {
        "schema_version": 1,
        "type": "tool_request",
        "tool": "read_thread",
        "arguments": {
            "threadId": "thread-1",
            "cursor": "cursor-1",
            "turnLimit": 1,
            "includeOutputs": False,
            "maxOutputCharsPerItem": 20_000,
        },
    }
    assert archive_thread_request() == {
        "schema_version": 1,
        "type": "tool_request",
        "tool": "set_thread_archived",
        "arguments": {"archived": True},
    }
    assert archive_thread_request("thread-1") == {
        "schema_version": 1,
        "type": "tool_request",
        "tool": "set_thread_archived",
        "arguments": {"archived": True, "threadId": "thread-1"},
    }


def test_archive_ack_requires_confirmed_current_thread() -> None:
    assert parse_archive_ack(
        '{"threadId":"thread-1","archived":true}',
        expected_thread_id="thread-1",
    ) == {"threadId": "thread-1", "archived": True}


def test_discovery_normalization_is_code_owned() -> None:
    result = normalize_discovery(
        {
            "schemaVersion": 4,
            "threads": [
                {
                    "id": "thread-1",
                    "kind": "chatgpt",
                    "projectId": "g-p-project",
                    "title": "One",
                    "createdAt": 1,
                    "updatedAt": 2,
                    "preview": "private preview",
                }
            ],
            "pinnedThreads": [],
        },
        observed_at="2026-08-31T15:00:00+00:00",
    )

    assert result == {
        "schema_version": 2,
        "observed_at": "2026-08-31T15:00:00+00:00",
        "discovery": {
            "non_pinned_returned": 1,
            "non_pinned_limit": 50,
            "recent_window_saturated": False,
        },
        "summaries": [
            {
                "id": "thread-1",
                "kind": "chatgpt",
                "projectId": "g-p-project",
                "title": "One",
                "createdAt": 1,
                "updatedAt": 2,
            }
        ],
        "pages": {},
    }
    assert "private preview" not in str(result)


def test_page_normalization_preserves_turn_order_and_pagination() -> None:
    result = normalize_page(
        {
            "schemaVersion": 1,
            "thread": {"id": "thread-1"},
            "page": {
                "order": "newest_first",
                "nextCursor": "cursor-2",
                "hasMore": True,
            },
            "turns": [
                {
                    "id": "turn-1",
                    "startedAt": 100,
                    "items": [
                        {
                            "type": "userMessage",
                            "id": "user-anchor",
                            "content": [{"type": "text", "text": "question"}],
                        },
                        {
                            "type": "agentMessage",
                            "id": "assistant-new",
                            "text": "answer",
                        },
                    ],
                }
            ],
        },
        expected_thread_id="thread-1",
        cursor_in="cursor-1",
    )

    assert [message["id"] for message in result["messages"]] == [
        "assistant-new",
        "user-anchor",
    ]
    assert result["cursor_in"] == "cursor-1"
    assert result["cursor_out"] == "cursor-2"
    assert result["has_more"] is True
    assert page_is_terminal(_plan(), result) is True
    assert next_cursor(_plan(), result) is None


def test_new_thread_continues_until_end() -> None:
    plan = _plan("new_thread")
    page = {
        "thread_id": "thread-1",
        "cursor_in": None,
        "cursor_out": "cursor-2",
        "has_more": True,
        "messages": [],
    }
    assert page_is_terminal(plan, page) is False
    assert next_cursor(plan, page) == "cursor-2"


def test_citation_rendering_change_preserves_anchor_proof(tmp_path) -> None:
    archived_text = "Source \ue200cite\ue202turn1search0\ue201 and \ue200cite\ue202turn1search1\ue201."
    observed_text = (
        'Source :chatgpt-content-reference{index="1"} and '
        ':chatgpt-content-reference{index="2"}.'
    )
    archive = tmp_path / "approved" / "thread.md"
    archive.parent.mkdir()
    archive.write_text(
        render_markdown(
            SessionRecord(
                "chatgpt", "thread-1", "Title", "2026-10-05",
                [MessageTurn("assistant", archived_text, message_id="assistant-anchor")],
            )
        ),
        encoding="utf-8",
    )
    plan = replace(
        _plan(),
        anchor_message_id="assistant-anchor",
        anchor_sha256=content_sha256(archived_text),
        output_file="approved/thread.md",
    )
    page = {
        "thread_id": "thread-1",
        "cursor_in": None,
        "cursor_out": "older",
        "has_more": True,
        "messages": [{
            "id": "assistant-anchor",
            "role": "assistant",
            "content": observed_text,
            "content_sha256": content_sha256(observed_text),
            "complete": True,
        }],
    }

    assert page_is_terminal(plan, page, archive_root=tmp_path) is True
    assert next_cursor(plan, page, archive_root=tmp_path) is None
    tail = collect_incremental_tail(plan, lambda _thread, _cursor: page,
                                    archive_root=tmp_path)
    assert tail.terminal_reason == "anchor"
    assert tail.messages == []

    changed = dict(page)
    changed["messages"] = [{
        **page["messages"][0],
        "content": observed_text.replace("Source", "Different source"),
        "content_sha256": content_sha256(
            observed_text.replace("Source", "Different source")
        ),
    }]
    with pytest.raises(ChatGPTIncrementalError, match="anchor hash mismatch"):
        page_is_terminal(plan, changed, archive_root=tmp_path)
    with pytest.raises(ChatGPTIncrementalError, match="anchor hash mismatch"):
        collect_incremental_tail(plan, lambda _thread, _cursor: changed,
                                 archive_root=tmp_path)

    with pytest.raises(ChatGPTIncrementalError, match="anchor hash mismatch"):
        page_is_terminal(plan, page)
    with pytest.raises(ChatGPTIncrementalError, match="anchor hash mismatch"):
        page_is_terminal(plan, page, archive_root=tmp_path / "missing")
