from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .archive import content_sha256, parse_marked_markdown
from .markdown import render_markdown
from .models import MessageTurn, SessionRecord
from .utils import sanitize_filename, unique_output_path


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
PERSONAL_SCOPE_ID = "personal"
TRUNCATION_SENTINEL_RE = re.compile(
    r"(?:…|\.{3})\s*\d+\s+tokens?\s+truncated\s*(?:…|\.{3})",
    re.IGNORECASE,
)


class ChatGPTIncrementalError(RuntimeError):
    """Raised when an incremental ChatGPT read cannot be proven complete."""


@dataclass(frozen=True)
class ThreadPlan:
    thread_id: str
    project_id: str
    project_label: str
    title: str
    created_at: float
    updated_at: float
    mode: Literal[
        "incremental_tail", "new_thread", "historical_backfill", "window_preview"
    ]
    anchor_message_id: str | None
    anchor_sha256: str | None
    output_file: str | None


@dataclass(frozen=True)
class BackfillNeed:
    thread_id: str
    project_id: str
    title: str
    reason: str


@dataclass(frozen=True)
class DiscoveryPlan:
    changed: list[ThreadPlan]
    needs_backfill: list[BackfillNeed]
    unchanged: int
    ignored: int


@dataclass(frozen=True)
class CollectedTail:
    messages: list[MessageTurn]
    pages_read: int
    terminal_reason: Literal["anchor", "end", "cutoff"]


@dataclass(frozen=True)
class ApplyResult:
    exported: int
    output_paths: list[Path]
    observer_handoff: dict[str, Any]


OBSERVER_HANDOFF_RETENTION = 90


def discovery_report(plan: DiscoveryPlan) -> dict[str, Any]:
    """Return the metadata-only contract used by the App-side reader."""

    return {
        "schema_version": 2,
        "changed": [
            {
                "thread_id": item.thread_id,
                "mode": item.mode,
                "anchor_message_id": item.anchor_message_id,
            }
            for item in plan.changed
        ],
        "needs_backfill": [
            {
                "thread_id": item.thread_id,
                "reason": item.reason,
            }
            for item in plan.needs_backfill
        ],
        "unchanged": plan.unchanged,
        "ignored": plan.ignored,
    }


def load_incremental_scope_allowlist(config_path: Path) -> dict[str, str]:
    """Load Project IDs plus an explicitly enabled projectless Personal scope."""

    from .sources.chatgpt import load_project_allowlist

    approved = load_project_allowlist(config_path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChatGPTIncrementalError(f"invalid ChatGPT scope config: {error}") from error
    personal = config.get("personal")
    if personal is None:
        return approved
    if not isinstance(personal, dict) or not isinstance(personal.get("enabled"), bool):
        raise ChatGPTIncrementalError("personal scope must declare enabled boolean")
    if personal["enabled"]:
        label = str(personal.get("label") or "").strip()
        if not label:
            raise ChatGPTIncrementalError("enabled personal scope lacks label")
        approved[PERSONAL_SCOPE_ID] = label
    return approved


def activate_live_authority(source_state: dict[str, Any], started_at: str) -> None:
    """Irreversibly close Official writes and start the daily Live frontier."""

    try:
        instant = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ChatGPTIncrementalError("live authority timestamp is invalid") from error
    if instant.tzinfo is None:
        raise ChatGPTIncrementalError("live authority timestamp must be timezone-aware")
    existing = source_state.get("live_authority_started_at")
    if existing is not None and existing != started_at:
        raise ChatGPTIncrementalError("live authority was already activated at another time")
    source_state["live_authority_started_at"] = started_at
    source_state["official_seed"] = {
        "status": "completed",
        "completed_at": started_at,
        "authority": "live_only",
    }


def discovery_saturation(payload: dict[str, Any]) -> bool | None:
    discovery = payload.get("discovery")
    if discovery is None:
        return None
    if not isinstance(discovery, dict):
        raise ChatGPTIncrementalError("incremental discovery must be an object")
    returned = discovery.get("non_pinned_returned")
    limit = discovery.get("non_pinned_limit")
    saturated = discovery.get("recent_window_saturated")
    if (
        isinstance(returned, bool)
        or not isinstance(returned, int)
        or returned < 0
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
        or not isinstance(saturated, bool)
    ):
        raise ChatGPTIncrementalError("incremental discovery counts are invalid")
    if saturated != (returned >= limit):
        raise ChatGPTIncrementalError("incremental discovery saturation is inconsistent")
    return saturated


def _revision_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        return None


def _safe_id(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if SAFE_ID_RE.fullmatch(text) else None


def _epoch_ms(value: Any) -> int | None:
    revision = _revision_number(value)
    if revision is None or revision <= 0:
        return None
    return int(revision if revision >= 1_000_000_000_000 else revision * 1000)


def _epoch_seconds(value: Any) -> float | None:
    revision = _revision_number(value)
    if revision is None:
        return None
    return revision / 1000 if revision >= 1_000_000_000_000 else revision


def _anchor(session_state: dict[str, Any]) -> tuple[str | None, str | None]:
    messages = session_state.get("messages")
    if not isinstance(messages, list):
        return None, None
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        message_id = _safe_id(item.get("message_id"))
        digest = str(item.get("content_sha256") or "").strip().lower()
        if message_id and re.fullmatch(r"[0-9a-f]{64}", digest):
            return message_id, digest
    return None, None


def plan_incremental_threads(
    summaries: list[dict[str, Any]],
    source_state: dict[str, Any],
    approved_projects: dict[str, str],
) -> DiscoveryPlan:
    """Plan cheap metadata-only reads without silently backfilling history."""

    sessions_value = source_state.get("sessions", {})
    sessions = sessions_value if isinstance(sessions_value, dict) else {}
    bootstrap_revision = _revision_number(
        source_state.get("live_authority_started_at")
        or source_state.get("bootstrap_completed_at")
    )
    changed: list[ThreadPlan] = []
    needs_backfill: list[BackfillNeed] = []
    unchanged = 0
    ignored = 0

    for summary in summaries:
        thread_id = _safe_id(summary.get("id"))
        raw_project_id = summary.get("projectId")
        project_id = (
            PERSONAL_SCOPE_ID
            if raw_project_id is None and PERSONAL_SCOPE_ID in approved_projects
            else str(raw_project_id or "").strip()
        )
        if summary.get("kind") != "chatgpt" or project_id not in approved_projects:
            ignored += 1
            continue
        if thread_id is None:
            needs_backfill.append(
                BackfillNeed("unknown", project_id, str(summary.get("title") or "Untitled"), "invalid_thread_id")
            )
            continue
        updated_at = _revision_number(summary.get("updatedAt"))
        created_at = _revision_number(summary.get("createdAt")) or updated_at
        if created_at is None or updated_at is None:
            needs_backfill.append(
                BackfillNeed(thread_id, project_id, str(summary.get("title") or "Untitled"), "invalid_revision")
            )
            continue

        prior_value = sessions.get(thread_id)
        prior = prior_value if isinstance(prior_value, dict) else None
        if prior is not None:
            prior_updated_at = _revision_number(prior.get("thread_updated_at"))
            if prior_updated_at is not None and updated_at <= prior_updated_at:
                unchanged += 1
                continue
            anchor_message_id, anchor_sha256 = _anchor(prior)
            if anchor_message_id is None:
                needs_backfill.append(
                    BackfillNeed(thread_id, project_id, str(summary.get("title") or "Untitled"), "missing_anchor")
                )
                continue
            changed.append(
                ThreadPlan(
                    thread_id=thread_id,
                    project_id=project_id,
                    project_label=approved_projects[project_id],
                    title=str(summary.get("title") or "Untitled"),
                    created_at=created_at,
                    updated_at=updated_at,
                    mode="incremental_tail",
                    anchor_message_id=anchor_message_id,
                    anchor_sha256=anchor_sha256,
                    output_file=str(prior.get("output_file") or "") or None,
                )
            )
            continue

        if bootstrap_revision is not None and created_at >= bootstrap_revision:
            changed.append(
                ThreadPlan(
                    thread_id=thread_id,
                    project_id=project_id,
                    project_label=approved_projects[project_id],
                    title=str(summary.get("title") or "Untitled"),
                    created_at=created_at,
                    updated_at=updated_at,
                    mode="new_thread",
                    anchor_message_id=None,
                    anchor_sha256=None,
                    output_file=None,
                )
            )
        else:
            needs_backfill.append(
                BackfillNeed(thread_id, project_id, str(summary.get("title") or "Untitled"), "historical_gap")
            )

    return DiscoveryPlan(changed, needs_backfill, unchanged, ignored)


def plan_historical_backfill(
    summaries: list[dict[str, Any]],
    source_state: dict[str, Any],
    approved_projects: dict[str, str],
) -> DiscoveryPlan:
    """Explicitly promote only historical gaps into terminal full-read work."""

    daily = plan_incremental_threads(summaries, source_state, approved_projects)
    gaps = {
        item.thread_id for item in daily.needs_backfill if item.reason == "historical_gap"
    }
    changed: list[ThreadPlan] = []
    for summary in summaries:
        thread_id = _safe_id(summary.get("id"))
        if thread_id not in gaps:
            continue
        raw_project_id = summary.get("projectId")
        project_id = (
            PERSONAL_SCOPE_ID
            if raw_project_id is None and PERSONAL_SCOPE_ID in approved_projects
            else str(raw_project_id or "").strip()
        )
        updated_at = _revision_number(summary.get("updatedAt"))
        created_at = _revision_number(summary.get("createdAt")) or updated_at
        if updated_at is None or created_at is None:
            continue
        changed.append(
            ThreadPlan(
                thread_id=thread_id,
                project_id=project_id,
                project_label=approved_projects[project_id],
                title=str(summary.get("title") or "Untitled"),
                created_at=created_at,
                updated_at=updated_at,
                mode="historical_backfill",
                anchor_message_id=None,
                anchor_sha256=None,
                output_file=None,
            )
        )
    remaining = [item for item in daily.needs_backfill if item.thread_id not in gaps]
    return DiscoveryPlan(changed, remaining, daily.unchanged, daily.ignored)


def plan_window_threads(
    summaries: list[dict[str, Any]],
    approved_projects: dict[str, str],
    *,
    since_at: float,
    until_at: float | None = None,
) -> DiscoveryPlan:
    """Plan an isolated half-open time window without consulting production state."""

    if until_at is not None and until_at <= since_at:
        raise ChatGPTIncrementalError("window until_at must be later than since_at")

    changed: list[ThreadPlan] = []
    ignored = 0
    for summary in summaries:
        thread_id = _safe_id(summary.get("id"))
        raw_project_id = summary.get("projectId")
        project_id = (
            PERSONAL_SCOPE_ID
            if raw_project_id is None and PERSONAL_SCOPE_ID in approved_projects
            else str(raw_project_id or "").strip()
        )
        updated_at = _epoch_seconds(summary.get("updatedAt"))
        if (
            summary.get("kind") != "chatgpt"
            or project_id not in approved_projects
            or thread_id is None
            or updated_at is None
            or updated_at < since_at
            or (until_at is not None and updated_at >= until_at)
        ):
            ignored += 1
            continue
        created_at = _epoch_seconds(summary.get("createdAt")) or updated_at
        changed.append(
            ThreadPlan(
                thread_id,
                project_id,
                approved_projects[project_id],
                str(summary.get("title") or "Untitled"),
                created_at,
                updated_at,
                "window_preview",
                None,
                None,
                None,
            )
        )
    return DiscoveryPlan(changed, [], 0, ignored)


def _message_from_summary(item: dict[str, Any]) -> MessageTurn:
    message_id = _safe_id(item.get("id"))
    role = str(item.get("role") or "")
    content = item.get("content")
    complete = item.get("complete", True)
    if message_id is None:
        raise ChatGPTIncrementalError("invalid message ID")
    if role not in {"user", "assistant"}:
        raise ChatGPTIncrementalError(f"invalid message role: {message_id}")
    if not isinstance(complete, bool):
        raise ChatGPTIncrementalError(f"invalid message completeness: {message_id}")
    if not isinstance(content, str):
        raise ChatGPTIncrementalError(f"invalid message content: {message_id}")
    if not content.strip() and not (role == "assistant" and complete is False):
        raise ChatGPTIncrementalError(f"empty message content: {message_id}")
    complete = complete and not bool(TRUNCATION_SENTINEL_RE.search(content))
    if not complete and role == "user":
        raise ChatGPTIncrementalError(f"incomplete user content: {message_id}")
    if not complete:
        marker = "[INCOMPLETE ASSISTANT CONTENT: source was truncated]"
        content = f"{marker}\n\n{content}" if content.strip() else marker
    return MessageTurn(
        role=role,
        content=content.strip(),
        time_created=_epoch_ms(item.get("created_at")),
        model=str(item.get("model") or "").strip() or None,
        message_id=message_id,
        complete=complete,
    )


def collect_incremental_tail(
    plan: ThreadPlan,
    read_page: Callable[[str, str | None], dict[str, Any]],
) -> CollectedTail:
    """Read newest-first pages until the prior message anchor is proven."""

    cursor: str | None = None
    seen_cursors: set[str | None] = {None}
    newest_first: list[MessageTurn] = []
    pages_read = 0
    while True:
        page = read_page(plan.thread_id, cursor)
        pages_read += 1
        if page.get("thread_id") != plan.thread_id or page.get("cursor_in") != cursor:
            raise ChatGPTIncrementalError(f"page response mismatch: {plan.thread_id}")
        messages = page.get("messages")
        if not isinstance(messages, list):
            raise ChatGPTIncrementalError(f"page messages must be an array: {plan.thread_id}")
        for item in messages:
            if not isinstance(item, dict):
                raise ChatGPTIncrementalError(f"page contains non-object message: {plan.thread_id}")
            message = _message_from_summary(item)
            if message.message_id == plan.anchor_message_id:
                observed_hash = str(item.get("content_sha256") or content_sha256(message.content))
                if observed_hash != plan.anchor_sha256:
                    raise ChatGPTIncrementalError(f"anchor hash mismatch: {plan.thread_id}")
                return CollectedTail(list(reversed(newest_first)), pages_read, "anchor")
            newest_first.append(message)

        has_more = page.get("has_more")
        if not isinstance(has_more, bool):
            raise ChatGPTIncrementalError(f"page has_more must be boolean: {plan.thread_id}")
        if not has_more:
            if plan.mode == "incremental_tail":
                raise ChatGPTIncrementalError(f"anchor not found: {plan.thread_id}")
            return CollectedTail(list(reversed(newest_first)), pages_read, "end")
        next_cursor = _safe_id(page.get("cursor_out"))
        if next_cursor is None:
            raise ChatGPTIncrementalError(f"missing pagination cursor: {plan.thread_id}")
        if next_cursor in seen_cursors:
            raise ChatGPTIncrementalError(f"pagination cursor loop: {plan.thread_id}")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def collect_window_tail(
    plan: ThreadPlan,
    read_page: Callable[[str, str | None], dict[str, Any]],
    *,
    since_at: float,
) -> CollectedTail:
    """Collect only messages in a time window and stop at the first older item."""

    cursor: str | None = None
    seen_cursors: set[str | None] = {None}
    newest_first: list[MessageTurn] = []
    pages_read = 0
    while True:
        page = read_page(plan.thread_id, cursor)
        pages_read += 1
        if page.get("thread_id") != plan.thread_id or page.get("cursor_in") != cursor:
            raise ChatGPTIncrementalError(f"page response mismatch: {plan.thread_id}")
        messages = page.get("messages")
        if not isinstance(messages, list):
            raise ChatGPTIncrementalError(f"page messages must be an array: {plan.thread_id}")
        for item in messages:
            if not isinstance(item, dict):
                raise ChatGPTIncrementalError(
                    f"page contains non-object message: {plan.thread_id}"
                )
            message = _message_from_summary(item)
            if message.time_created is None:
                raise ChatGPTIncrementalError(
                    f"window message lacks timestamp: {plan.thread_id}"
                )
            if message.time_created / 1000 < since_at:
                return CollectedTail(
                    list(reversed(newest_first)), pages_read, "cutoff"
                )
            newest_first.append(message)
        has_more = page.get("has_more")
        if not isinstance(has_more, bool):
            raise ChatGPTIncrementalError(f"page has_more must be boolean: {plan.thread_id}")
        if not has_more:
            return CollectedTail(list(reversed(newest_first)), pages_read, "end")
        next_cursor = _safe_id(page.get("cursor_out"))
        if next_cursor is None:
            raise ChatGPTIncrementalError(f"missing pagination cursor: {plan.thread_id}")
        if next_cursor in seen_cursors:
            raise ChatGPTIncrementalError(f"pagination cursor loop: {plan.thread_id}")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _message_metadata(messages: list[MessageTurn]) -> list[dict[str, Any]]:
    return [
        {
            "message_id": message.message_id,
            "role": message.role,
            "created_at": message.time_created,
            "content_sha256": content_sha256(message.content),
            "complete": message.complete,
        }
        for message in messages
    ]


def _branch_fingerprint(metadata: list[dict[str, Any]]) -> str:
    canonical = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _observer_handoff(
    staged_state: dict[str, Any],
    changes: list[dict[str, Any]],
    *,
    observed_at: str,
    status: Literal["success", "partial_failure"],
) -> dict[str, Any]:
    """Append one raw-text-free, generation-ordered Observer handoff."""

    generation_value = staged_state.get("archive_generation", 0)
    if isinstance(generation_value, bool) or not isinstance(generation_value, int):
        raise ChatGPTIncrementalError("invalid archive_generation")
    generation = generation_value + 1
    existing = staged_state.get("observer_handoffs", [])
    if not isinstance(existing, list) or any(
        not isinstance(item, dict) for item in existing
    ):
        raise ChatGPTIncrementalError("invalid observer_handoffs")
    material = {
        "schema_version": 1,
        "archive_generation": generation,
        "previous_generation": generation_value,
        "observed_at": observed_at,
        "status": status,
        "zero_user_delta": not any(
            change.get("user_messages") for change in changes
        ),
        "changes": changes,
    }
    handoff = {
        **material,
        "run_id": _canonical_sha256(material)[:32],
    }
    staged_state["archive_generation"] = generation
    staged_state["observer_handoffs"] = (
        existing + [handoff]
    )[-OBSERVER_HANDOFF_RETENTION:]
    return handoff


def _archive_path(output_dir: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ChatGPTIncrementalError("invalid archive output path")
    root = output_dir.resolve()
    path = (output_dir / relative).resolve()
    if path != root and root not in path.parents:
        raise ChatGPTIncrementalError("archive output path escapes root")
    return path


def _existing_messages(
    output_path: Path, session_state: dict[str, Any]
) -> list[MessageTurn]:
    if not output_path.is_file():
        raise ChatGPTIncrementalError(f"existing archive is missing: {output_path}")
    try:
        archived = parse_marked_markdown(output_path)
    except (OSError, ValueError) as error:
        raise ChatGPTIncrementalError(f"existing archive validation failed: {error}") from error
    metadata = session_state.get("messages")
    if not isinstance(metadata, list) or len(metadata) != len(archived):
        raise ChatGPTIncrementalError("archive and checkpoint message counts differ")
    messages: list[MessageTurn] = []
    # Length equality is checked above; avoid ``zip(strict=...)`` so the public
    # CLI also runs under the workspace's Python 3.9 automation runtime.
    for turn, item in zip(archived, metadata):
        if not isinstance(item, dict):
            raise ChatGPTIncrementalError("invalid checkpoint message metadata")
        expected = (
            str(item.get("message_id") or ""),
            str(item.get("role") or ""),
            str(item.get("content_sha256") or ""),
            item.get("complete", True),
        )
        actual = (turn.message_id, turn.role, turn.content_sha256, turn.complete)
        if expected != actual:
            raise ChatGPTIncrementalError(
                f"archive and checkpoint differ at message: {turn.message_id}"
            )
        created_at = _revision_number(item.get("created_at"))
        messages.append(
            MessageTurn(
                turn.role,
                turn.content,
                int(created_at) if created_at is not None else None,
                message_id=turn.message_id,
                complete=turn.complete,
            )
        )
    return messages


def _atomic_write_verified(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incremental.tmp")
    try:
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.read_text(encoding="utf-8") != rendered:
        raise ChatGPTIncrementalError(f"Markdown reread verification failed: {path}")
    parse_marked_markdown(path)


def _restore_file(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
        return
    temporary = path.with_name(f".{path.name}.incremental-rollback.tmp")
    temporary.write_bytes(previous)
    temporary.replace(path)


def apply_incremental_batch(
    output_dir: Path,
    source_state: dict[str, Any],
    batches: list[tuple[ThreadPlan, CollectedTail]],
    *,
    observed_at: str,
    handoff_status: Literal["success", "partial_failure"] = "success",
    checkpoint_state: Callable[[dict[str, Any]], None] | None = None,
) -> ApplyResult:
    """Atomically apply verified tails, then advance the source checkpoint."""

    sessions_value = source_state.get("sessions")
    if not isinstance(sessions_value, dict):
        raise ChatGPTIncrementalError("invalid ChatGPT source state")
    staged_state = deepcopy(source_state)
    staged_sessions = staged_state["sessions"]
    planned_writes: list[tuple[Path, str]] = []
    handoff_changes: list[dict[str, Any]] = []
    reserved_paths: set[Path] = set()
    seen_threads: set[str] = set()

    for plan, tail in batches:
        if plan.thread_id in seen_threads:
            raise ChatGPTIncrementalError(f"duplicate planned thread: {plan.thread_id}")
        seen_threads.add(plan.thread_id)
        prior_value = sessions_value.get(plan.thread_id)
        prior = prior_value if isinstance(prior_value, dict) else None
        if plan.mode == "incremental_tail":
            if prior is None or tail.terminal_reason != "anchor":
                raise ChatGPTIncrementalError(f"incremental thread lacks verified anchor: {plan.thread_id}")
            relative = str(prior.get("output_file") or plan.output_file or "")
            output_path = _archive_path(output_dir, relative)
            old_messages = _existing_messages(output_path, prior)
        else:
            terminal_is_valid = tail.terminal_reason == "end" or (
                plan.mode == "window_preview" and tail.terminal_reason == "cutoff"
            )
            if prior is not None or not terminal_is_valid or not tail.messages:
                raise ChatGPTIncrementalError(f"new thread lacks terminal history: {plan.thread_id}")
            project_dir = output_dir / sanitize_filename(plan.project_label)
            archive_started_at = (
                tail.messages[0].time_created / 1000
                if plan.mode == "window_preview" and tail.messages[0].time_created
                else plan.created_at
            )
            day = datetime.fromtimestamp(archive_started_at).date().isoformat()
            output_path = unique_output_path(
                project_dir, day, plan.title, reserved_paths=reserved_paths
            )
            relative = output_path.relative_to(output_dir).as_posix()
            old_messages = []
        reserved_paths.add(output_path)

        old_ids = {message.message_id for message in old_messages}
        new_ids = [message.message_id for message in tail.messages]
        if len(new_ids) != len(set(new_ids)) or old_ids.intersection(new_ids):
            raise ChatGPTIncrementalError(f"duplicate message during merge: {plan.thread_id}")
        merged = old_messages + tail.messages
        metadata = _message_metadata(merged)
        record_started_at = (
            merged[0].time_created / 1000
            if plan.mode == "window_preview" and merged[0].time_created
            else plan.created_at
        )
        record = SessionRecord(
            source="chatgpt",
            session_id=plan.thread_id,
            title=plan.title,
            date=datetime.fromtimestamp(record_started_at).date().isoformat(),
            messages=merged,
            models_used=sorted({message.model for message in merged if message.model}),
        )
        rendered = render_markdown(record)
        planned_writes.append((output_path, rendered))
        staged_session = {
            "project_id": plan.project_id,
            "project_label": plan.project_label,
            "output_file": relative,
            "thread_updated_at": plan.updated_at,
            "coverage": (
                "window_preview"
                if plan.mode == "window_preview"
                else "full_history"
            ),
            "branch_fingerprint": _branch_fingerprint(metadata),
            "messages": metadata,
            "user_complete": all(m.complete for m in merged if m.role == "user"),
            "assistant_complete": all(m.complete for m in merged if m.role == "assistant"),
            "last_seen_at": observed_at,
            "last_input_kind": "app_incremental",
        }
        staged_sessions[plan.thread_id] = staged_session
        prior_anchor = old_messages[-1] if old_messages else None
        handoff_changes.append(
            {
                "project_id": plan.project_id,
                "thread_id": plan.thread_id,
                "archive_path": relative,
                "thread_updated_at": plan.updated_at,
                "coverage": staged_session["coverage"],
                "branch_status": "linear",
                "before_archive_sha256": (
                    hashlib.sha256(output_path.read_bytes()).hexdigest()
                    if output_path.is_file()
                    else None
                ),
                "after_archive_sha256": hashlib.sha256(
                    rendered.encode("utf-8")
                ).hexdigest(),
                "exporter_session_sha256": _canonical_sha256(staged_session),
                "start_after": (
                    {
                        "message_id": prior_anchor.message_id,
                        "content_sha256": content_sha256(prior_anchor.content),
                    }
                    if prior_anchor is not None
                    else None
                ),
                "user_messages": [
                    {
                        "message_id": message.message_id,
                        "content_sha256": content_sha256(message.content),
                        "created_at": message.time_created,
                        "change": "new",
                    }
                    for message in tail.messages
                    if message.role == "user"
                ],
            }
        )

    handoff = _observer_handoff(
        staged_state,
        handoff_changes,
        observed_at=observed_at,
        status=handoff_status,
    )

    previous_files = [
        (path, path.read_bytes() if path.is_file() else None) for path, _ in planned_writes
    ]
    try:
        for path, rendered in planned_writes:
            _atomic_write_verified(path, rendered)
        if checkpoint_state is not None:
            checkpoint_state(staged_state)
    except Exception:
        for path, previous in reversed(previous_files):
            _restore_file(path, previous)
        raise

    source_state.clear()
    source_state.update(staged_state)
    return ApplyResult(
        len(planned_writes),
        [path for path, _ in planned_writes],
        handoff,
    )


def apply_incremental_payload(
    output_dir: Path,
    source_state: dict[str, Any],
    approved_projects: dict[str, str],
    payload: dict[str, Any],
    *,
    historical_backfill: bool = False,
    window_since_at: float | None = None,
    window_until_at: float | None = None,
    checkpoint_state: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Apply an App-normalized v2 payload without inspecting unchanged bodies."""

    if payload.get("schema_version") != 2:
        raise ChatGPTIncrementalError("unsupported incremental payload schema")
    observed_at = payload.get("observed_at")
    if not isinstance(observed_at, str):
        raise ChatGPTIncrementalError("incremental payload lacks observed_at")
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ChatGPTIncrementalError("incremental observed_at is invalid") from error
    if observed.tzinfo is None:
        raise ChatGPTIncrementalError("incremental observed_at must be timezone-aware")
    saturated = discovery_saturation(payload)
    summaries = payload.get("summaries")
    pages_value = payload.get("pages")
    if not isinstance(summaries, list) or not all(isinstance(item, dict) for item in summaries):
        raise ChatGPTIncrementalError("incremental summaries must be an object array")
    if not isinstance(pages_value, dict):
        raise ChatGPTIncrementalError("incremental pages must be an object")

    if historical_backfill and window_since_at is not None:
        raise ChatGPTIncrementalError("backfill and window modes are mutually exclusive")
    if window_until_at is not None and window_since_at is None:
        raise ChatGPTIncrementalError("window until_at requires since_at")
    if window_since_at is not None:
        discovery = plan_window_threads(
            summaries,
            approved_projects,
            since_at=window_since_at,
            until_at=window_until_at,
        )
    else:
        planner = plan_historical_backfill if historical_backfill else plan_incremental_threads
        discovery = planner(summaries, source_state, approved_projects)
    planned_ids = {item.thread_id for item in discovery.changed}
    supplied_ids = set(pages_value)
    extra_ids = supplied_ids - planned_ids
    if extra_ids:
        raise ChatGPTIncrementalError(
            "payload contains unplanned thread pages: " + ",".join(sorted(extra_ids))
        )

    successful: list[tuple[ThreadPlan, CollectedTail]] = []
    warnings: list[dict[str, str]] = []
    for plan in discovery.changed:
        raw_pages = pages_value.get(plan.thread_id)
        if not isinstance(raw_pages, list) or not raw_pages:
            warnings.append({"thread_id": plan.thread_id, "error": "planned thread pages are missing"})
            continue
        page_index = 0

        def read_page(thread_id: str, cursor: str | None) -> dict[str, Any]:
            nonlocal page_index
            if page_index >= len(raw_pages):
                raise ChatGPTIncrementalError(f"page chain ended early: {thread_id}")
            page = raw_pages[page_index]
            page_index += 1
            if not isinstance(page, dict):
                raise ChatGPTIncrementalError(f"page must be an object: {thread_id}")
            return page

        try:
            tail = (
                collect_window_tail(plan, read_page, since_at=window_since_at)
                if window_since_at is not None
                else collect_incremental_tail(plan, read_page)
            )
            if page_index != len(raw_pages):
                raise ChatGPTIncrementalError(f"page chain continues past terminal proof: {plan.thread_id}")
            successful.append((plan, tail))
        except ChatGPTIncrementalError as error:
            warnings.append({"thread_id": plan.thread_id, "error": str(error)})

    apply_result = apply_incremental_batch(
        output_dir,
        source_state,
        successful,
        observed_at=observed_at,
        handoff_status="partial_failure" if warnings else "success",
        checkpoint_state=checkpoint_state,
    )
    return {
        "source": "chatgpt",
        "scanned": len(summaries),
        "exported": apply_result.exported,
        "failed": len(warnings),
        "ignored": discovery.ignored,
        "unchanged": discovery.unchanged,
        "backfill_pending": len(discovery.needs_backfill),
        "discovery_saturated": saturated,
        "warnings": warnings,
        "output_paths": [path.relative_to(output_dir).as_posix() for path in apply_result.output_paths],
        "observer_handoff": apply_result.observer_handoff,
    }
