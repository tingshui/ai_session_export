from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .archive import content_sha256
from .chatgpt_incremental import ChatGPTIncrementalError, ThreadPlan


PROTOCOL_SCHEMA_VERSION = 1
LIST_THREADS_LIMIT = 50
READ_THREAD_TURN_LIMIT = 1
READ_THREAD_MAX_OUTPUT_CHARS = 20_000


def tool_request(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the only App-tool request shape used by scheduled Step A."""
    if tool not in {"list_threads", "read_thread", "set_thread_archived"}:
        raise ChatGPTIncrementalError(f"unsupported App tool request: {tool}")
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "type": "tool_request",
        "tool": tool,
        "arguments": arguments,
    }


def discovery_request() -> dict[str, Any]:
    return tool_request("list_threads", {"limit": LIST_THREADS_LIMIT})


def archive_thread_request(thread_id: str | None = None) -> dict[str, Any]:
    """Ask the App to archive one explicit task or the calling task."""
    arguments: dict[str, Any] = {"archived": True}
    if thread_id is not None:
        if not isinstance(thread_id, str) or not thread_id:
            raise ChatGPTIncrementalError("archive threadId must be non-empty")
        arguments["threadId"] = thread_id
    return tool_request("set_thread_archived", arguments)


def parse_archive_ack(
    line: str,
    *,
    expected_thread_id: str | None = None,
) -> dict[str, Any]:
    """Validate the App acknowledgement for task archival."""
    payload = parse_tool_text(line)
    thread_id = payload.get("threadId")
    if not isinstance(thread_id, str) or not thread_id:
        raise ChatGPTIncrementalError("archive acknowledgement lacks threadId")
    if payload.get("archived") is not True:
        raise ChatGPTIncrementalError(
            "archive acknowledgement does not confirm archival"
        )
    if expected_thread_id is not None and thread_id != expected_thread_id:
        raise ChatGPTIncrementalError(
            "archive acknowledgement threadId does not match request"
        )
    return payload


def page_request(thread_id: str, cursor: str | None) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "threadId": thread_id,
        "turnLimit": READ_THREAD_TURN_LIMIT,
        "includeOutputs": False,
        "maxOutputCharsPerItem": READ_THREAD_MAX_OUTPUT_CHARS,
    }
    if cursor is not None:
        arguments["cursor"] = cursor
    return tool_request("read_thread", arguments)


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChatGPTIncrementalError(f"{field} must be an object")
    return value


def _array(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ChatGPTIncrementalError(f"{field} must be an array")
    return value


def parse_tool_text(line: str) -> dict[str, Any]:
    """Parse the exact JSON text block returned by one Codex App tool call."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ChatGPTIncrementalError(
            f"App tool response is not JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ChatGPTIncrementalError("App tool response must be an object")
    return payload


def normalize_discovery(
    payload: dict[str, Any],
    *,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Convert App list_threads output into the exporter v2 metadata schema."""
    if not isinstance(payload.get("schemaVersion"), int):
        raise ChatGPTIncrementalError("list_threads lacks schemaVersion")
    threads = _array(payload.get("threads"), "list_threads.threads")
    pinned = _array(payload.get("pinnedThreads"), "list_threads.pinnedThreads")
    summaries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in [*threads, *pinned]:
        item = _object(raw, "list_threads item")
        thread_id = item.get("id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ChatGPTIncrementalError("list_threads item lacks id")
        if thread_id in seen:
            continue
        seen.add(thread_id)
        summary = {
            key: item.get(key)
            for key in ("id", "kind", "projectId", "title", "createdAt", "updatedAt")
            if key in item
        }
        summaries.append(summary)
    timestamp = observed_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    return {
        "schema_version": 2,
        "observed_at": timestamp,
        "discovery": {
            "non_pinned_returned": len(threads),
            "non_pinned_limit": LIST_THREADS_LIMIT,
            "recent_window_saturated": len(threads) >= LIST_THREADS_LIMIT,
        },
        "summaries": summaries,
        "pages": {},
    }


def _item_text(item: dict[str, Any]) -> str:
    if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
        return item["text"].strip()
    content = item.get("content")
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for part in content if isinstance(content, list) else []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") in {
            "text",
            "input_text",
            "output_text",
        }:
            parts.append(str(part.get("text") or ""))
    return "\n\n".join(part for part in parts if part).strip()


def _complete(item: dict[str, Any]) -> bool:
    for field in ("complete", "isComplete"):
        if field in item:
            value = item[field]
            if not isinstance(value, bool):
                raise ChatGPTIncrementalError(
                    f"read_thread item {field} must be boolean"
                )
            return value
    return not any(
        item.get(field) is True
        for field in ("truncated", "isTruncated", "contentTruncated")
    )


def _normalized_message(
    item: dict[str, Any],
    turn: dict[str, Any],
) -> dict[str, Any] | None:
    item_type = item.get("type")
    role = (
        "user"
        if item_type == "userMessage"
        else "assistant" if item_type == "agentMessage" else None
    )
    if role is None:
        return None
    message_id = item.get("id")
    if not isinstance(message_id, str) or not message_id:
        raise ChatGPTIncrementalError("read_thread message lacks id")
    text = _item_text(item)
    complete = _complete(item)
    if not text and complete:
        return None
    created_at = item.get("createdAt", item.get("created_at", turn.get("startedAt")))
    model = item.get("model", turn.get("model"))
    return {
        "id": message_id,
        "role": role,
        "content": text,
        "created_at": created_at,
        "model": model,
        "complete": complete,
        "content_sha256": content_sha256(text),
    }


def normalize_page(
    payload: dict[str, Any],
    *,
    expected_thread_id: str,
    cursor_in: str | None,
) -> dict[str, Any]:
    """Convert one App read_thread result to the exporter page schema."""
    if not isinstance(payload.get("schemaVersion"), int):
        raise ChatGPTIncrementalError("read_thread lacks schemaVersion")
    thread = _object(payload.get("thread"), "read_thread.thread")
    if thread.get("id") != expected_thread_id:
        raise ChatGPTIncrementalError("read_thread returned another thread")
    page = _object(payload.get("page"), "read_thread.page")
    if page.get("order") != "newest_first":
        raise ChatGPTIncrementalError("read_thread page order must be newest_first")
    has_more = page.get("hasMore")
    if not isinstance(has_more, bool):
        raise ChatGPTIncrementalError("read_thread page lacks hasMore boolean")
    cursor_out = page.get("nextCursor")
    if has_more and (not isinstance(cursor_out, str) or not cursor_out):
        raise ChatGPTIncrementalError("read_thread nonterminal page lacks nextCursor")
    if not has_more and cursor_out is not None:
        raise ChatGPTIncrementalError("read_thread terminal page has nextCursor")

    messages: list[dict[str, Any]] = []
    # Turns are newest-first, while items inside one turn are chronological.
    # Reverse only within each turn to produce a globally newest-first stream.
    for raw_turn in _array(payload.get("turns"), "read_thread.turns"):
        turn = _object(raw_turn, "read_thread turn")
        items = _array(turn.get("items"), "read_thread turn.items")
        for raw_item in reversed(items):
            item = _object(raw_item, "read_thread item")
            message = _normalized_message(item, turn)
            if message is not None:
                messages.append(message)
    return {
        "thread_id": expected_thread_id,
        "cursor_in": cursor_in,
        "cursor_out": cursor_out,
        "has_more": has_more,
        "messages": messages,
    }


def page_is_terminal(plan: ThreadPlan, page: dict[str, Any]) -> bool:
    if plan.mode == "incremental_tail":
        for message in page.get("messages", []):
            if message.get("id") != plan.anchor_message_id:
                continue
            if message.get("content_sha256") != plan.anchor_sha256:
                raise ChatGPTIncrementalError(
                    f"anchor hash mismatch: {plan.thread_id}"
                )
            return True
        return False
    return page.get("has_more") is False


def next_cursor(plan: ThreadPlan, page: dict[str, Any]) -> str | None:
    if page_is_terminal(plan, page):
        return None
    if page.get("has_more") is not True:
        raise ChatGPTIncrementalError(f"anchor not found: {plan.thread_id}")
    cursor = page.get("cursor_out")
    if not isinstance(cursor, str) or not cursor:
        raise ChatGPTIncrementalError(
            f"missing pagination cursor: {plan.thread_id}"
        )
    return cursor
