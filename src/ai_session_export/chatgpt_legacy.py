from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .archive import content_sha256, parse_marked_markdown
from .chatgpt_incremental import (
    ChatGPTIncrementalError,
    _archive_path,
    _atomic_write_verified,
    _branch_fingerprint,
    _message_metadata,
)
from .markdown import render_markdown
from .models import MessageTurn, SessionRecord
from .sources.chatgpt import load_project_allowlist
from .state import assert_chatgpt_state_local, save_state
from .utils import unique_output_path


LEGACY_ID_RE = re.compile(
    r"(?m)^- \*\*conversation_id\*\*: `([^`]+)`\s*$"
)
LEGACY_METADATA_RE = re.compile(
    r"(?m)^- \*\*(创建|最后更新|模型)\*\*: (.*?)\s*$"
)
LEGACY_HEADING_RE = re.compile(
    r"^### (🧑 user|🤖 assistant)\s+·\s+(\d{2}):(\d{2})\s*$"
)
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
FRONTMATTER_VALUE_RE = re.compile(r'(?m)^(title|date):\s+"(.*)"\s*$')
LEGACY_MESSAGE_NAMESPACE = uuid.UUID("9b7dc400-ae2e-47ef-9d9d-3954a2c39288")


class ChatGPTLegacyImportError(RuntimeError):
    """Raised when a legacy Markdown archive cannot be migrated losslessly."""


@dataclass(frozen=True)
class LegacyConversation:
    source_path: Path
    project_id: str
    project_label: str
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    model: str
    messages: list[MessageTurn]


@dataclass(frozen=True)
class PlannedConversation:
    legacy: LegacyConversation
    output_path: Path
    output_relative: str
    record: SessionRecord
    state_entry: dict[str, Any]
    overlap_with_live: bool
    duplicate_legacy_turns_removed: int


@dataclass(frozen=True)
class LegacyImportPlan:
    conversations: list[PlannedConversation]
    state: dict[str, Any]
    index_writes: dict[Path, str]


def _metadata(text: str) -> dict[str, str]:
    values = {key: value.strip() for key, value in LEGACY_METADATA_RE.findall(text)}
    missing = {"创建", "最后更新", "模型"}.difference(values)
    if missing:
        raise ChatGPTLegacyImportError(
            f"legacy Markdown lacks required metadata: {', '.join(sorted(missing))}"
        )
    return values


def _parse_local_datetime(value: str, timezone_name: str) -> datetime:
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ChatGPTLegacyImportError(f"unknown timezone: {timezone_name}") from error
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=zone)
    except ValueError as error:
        raise ChatGPTLegacyImportError(f"invalid legacy timestamp: {value!r}") from error


def _legacy_sections(text: str) -> list[tuple[str, int, int, str]]:
    """Parse old emoji headings while ignoring heading-shaped text in fences."""

    lines = text.splitlines()
    headings: list[tuple[int, str, int, int]] = []
    fence_character: str | None = None
    fence_length = 0
    for index, line in enumerate(lines):
        fence = FENCE_RE.match(line)
        if fence:
            token = fence.group(1)
            if fence_character is None:
                fence_character = token[0]
                fence_length = len(token)
            elif token[0] == fence_character and len(token) >= fence_length:
                fence_character = None
                fence_length = 0
            continue
        if fence_character is not None:
            continue
        match = LEGACY_HEADING_RE.fullmatch(line)
        if match:
            role = "user" if match.group(1) == "🧑 user" else "assistant"
            headings.append((index, role, int(match.group(2)), int(match.group(3))))

    if not headings:
        raise ChatGPTLegacyImportError("legacy Markdown contains no dialogue turns")
    sections: list[tuple[str, int, int, str]] = []
    for position, (line_index, role, hour, minute) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        body_lines = lines[line_index + 1 : end]
        if body_lines and body_lines[0] == "":
            body_lines = body_lines[1:]
        body = "\n".join(body_lines).rstrip()
        if not body:
            raise ChatGPTLegacyImportError(
                f"legacy Markdown contains an empty {role} turn at heading {position + 1}"
            )
        sections.append((role, hour, minute, body))
    return sections


def _turn_timestamps(
    sections: list[tuple[str, int, int, str]], created_at: datetime
) -> list[int]:
    """Preserve HH:MM and order; unavailable historical dates remain approximate."""

    current_day = created_at.date()
    previous: datetime | None = None
    same_minute_offset = 0
    result: list[int] = []
    for _role, hour, minute, _body in sections:
        candidate = datetime.combine(
            current_day,
            datetime.min.time().replace(hour=hour, minute=minute),
            tzinfo=created_at.tzinfo,
        )
        if previous is not None and candidate < previous.replace(microsecond=0):
            current_day += timedelta(days=1)
            candidate += timedelta(days=1)
        if previous is not None and candidate.replace(second=0, microsecond=0) == previous.replace(
            second=0, microsecond=0
        ):
            same_minute_offset += 1
        else:
            same_minute_offset = 0
        candidate += timedelta(milliseconds=same_minute_offset)
        previous = candidate
        result.append(int(candidate.timestamp() * 1000))
    return result


def _stable_message_id(
    conversation_id: str, role: str, ordinal: int, content: str
) -> str:
    seed = f"chatgpt-legacy:{conversation_id}:{role}:{ordinal}:{content_sha256(content)}"
    return str(uuid.uuid5(LEGACY_MESSAGE_NAMESPACE, seed))


def parse_legacy_markdown(
    path: Path,
    *,
    project_id: str,
    project_label: str,
    timezone_name: str,
) -> LegacyConversation:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ChatGPTLegacyImportError(f"cannot read legacy Markdown: {path}") from error
    if text.startswith("---\nsource: chatgpt\n"):
        raise ChatGPTLegacyImportError("canonical Markdown is not a legacy input")
    lines = text.splitlines()
    title_line = lines[0] if lines else ""
    if not title_line.startswith("# ") or not title_line[2:].strip():
        raise ChatGPTLegacyImportError("legacy Markdown lacks a title")
    title = title_line[2:].strip()
    identifier = LEGACY_ID_RE.search(text)
    if identifier is None:
        raise ChatGPTLegacyImportError("legacy Markdown lacks conversation_id")
    conversation_id = identifier.group(1).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}", conversation_id):
        raise ChatGPTLegacyImportError("legacy Markdown has invalid conversation_id")
    metadata = _metadata(text)
    created_at = _parse_local_datetime(metadata["创建"], timezone_name)
    updated_at = _parse_local_datetime(metadata["最后更新"], timezone_name)
    if updated_at < created_at:
        raise ChatGPTLegacyImportError("legacy update timestamp precedes creation")
    sections = _legacy_sections(text)
    timestamps = _turn_timestamps(sections, created_at)
    if len(sections) != len(timestamps):
        raise ChatGPTLegacyImportError("legacy turn timestamp count mismatch")
    messages = [
        MessageTurn(
            role=role,
            content=content,
            time_created=timestamp,
            model=metadata["模型"] if role == "assistant" else None,
            message_id=_stable_message_id(conversation_id, role, ordinal, content),
            complete=True,
        )
        for ordinal, ((role, _hour, _minute, content), timestamp) in enumerate(
            zip(sections, timestamps), start=1
        )
    ]
    return LegacyConversation(
        source_path=path,
        project_id=project_id,
        project_label=project_label,
        conversation_id=conversation_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        model=metadata["模型"],
        messages=messages,
    )


def _frontmatter_identity(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    values = {key: json.loads(f'"{value}"') for key, value in FRONTMATTER_VALUE_RE.findall(text)}
    return str(values.get("title") or path.stem), str(values.get("date") or "")


def _existing_messages(
    archive_root: Path, session_state: dict[str, Any]
) -> tuple[Path, list[MessageTurn]]:
    relative = str(session_state.get("output_file") or "")
    try:
        output_path = _archive_path(archive_root, relative)
        archived = parse_marked_markdown(output_path)
    except (ChatGPTIncrementalError, OSError, ValueError) as error:
        raise ChatGPTLegacyImportError(f"existing Live archive validation failed: {error}") from error
    metadata = session_state.get("messages")
    if not isinstance(metadata, list) or len(metadata) != len(archived):
        raise ChatGPTLegacyImportError("Live archive and state message counts differ")
    messages: list[MessageTurn] = []
    # Length equality is checked above; avoid ``zip(strict=...)`` so the
    # workspace's Python 3.9 automation runtime can execute the importer.
    for turn, item in zip(archived, metadata):
        if not isinstance(item, dict):
            raise ChatGPTLegacyImportError("Live state contains invalid message metadata")
        expected = (
            str(item.get("message_id") or ""),
            str(item.get("role") or ""),
            str(item.get("content_sha256") or ""),
            item.get("complete", True),
        )
        actual = (turn.message_id, turn.role, turn.content_sha256, turn.complete)
        if expected != actual:
            raise ChatGPTLegacyImportError(
                f"Live archive and state differ at message: {turn.message_id}"
            )
        created_at = item.get("created_at")
        messages.append(
            MessageTurn(
                role=turn.role,
                content=turn.content,
                time_created=int(created_at) if isinstance(created_at, (int, float)) else None,
                message_id=turn.message_id,
                complete=turn.complete,
            )
        )
    return output_path, messages


def _merge_live_wins(
    legacy: list[MessageTurn], live: list[MessageTurn]
) -> tuple[list[MessageTurn], int]:
    live_counts = Counter((turn.role, content_sha256(turn.content)) for turn in live)
    kept_reversed: list[MessageTurn] = []
    removed = 0
    for turn in reversed(legacy):
        key = (turn.role, content_sha256(turn.content))
        if live_counts[key]:
            live_counts[key] -= 1
            removed += 1
        else:
            kept_reversed.append(turn)
    merged = list(reversed(kept_reversed)) + live
    ids = [turn.message_id for turn in merged]
    if len(ids) != len(set(ids)):
        raise ChatGPTLegacyImportError("legacy and Live message IDs collide")
    return merged, removed


def _discover_legacy(
    archive_root: Path,
    label_to_id: dict[str, str],
    timezone_name: str,
) -> list[LegacyConversation]:
    conversations: list[LegacyConversation] = []
    for path in sorted(archive_root.glob("*/*.md")):
        if path.name == "README.md":
            continue
        text_prefix = path.read_text(encoding="utf-8")[:512]
        if "**conversation_id**" not in text_prefix:
            continue
        project_label = path.parent.name
        if project_label not in label_to_id:
            raise ChatGPTLegacyImportError(
                f"legacy Markdown is under an unapproved Project directory: {project_label}"
            )
        conversations.append(
            parse_legacy_markdown(
                path,
                project_id=label_to_id[project_label],
                project_label=project_label,
                timezone_name=timezone_name,
            )
        )
    identifiers = [item.conversation_id for item in conversations]
    duplicates = sorted(key for key, count in Counter(identifiers).items() if count > 1)
    if duplicates:
        raise ChatGPTLegacyImportError(
            f"duplicate legacy conversation IDs: {', '.join(duplicates)}"
        )
    return conversations


def _index_writes(
    archive_root: Path,
    project_labels: list[str],
    planned: list[PlannedConversation],
    sessions: dict[str, Any],
) -> dict[Path, str]:
    entries: dict[str, list[tuple[str, str, str]]] = {label: [] for label in project_labels}
    planned_by_relative = {item.output_relative: item for item in planned}
    for value in sessions.values():
        if not isinstance(value, dict):
            continue
        label = str(value.get("project_label") or "")
        relative = str(value.get("output_file") or "")
        if label not in entries or not relative:
            continue
        if relative in planned_by_relative:
            item = planned_by_relative[relative]
            title, date = item.record.title, item.record.date
        else:
            path = _archive_path(archive_root, relative)
            title, date = _frontmatter_identity(path)
        filename = Path(relative).name
        entries[label].append((date, title, filename))

    writes: dict[Path, str] = {}
    root_lines = ["# ChatGPT Projects 导出索引", ""]
    for label in project_labels:
        values = sorted(entries[label], key=lambda item: (item[0], item[1], item[2]), reverse=True)
        root_lines.append(f"- [{label}]({label}/README.md) — {len(values)} conversations")
        project_lines = [f"# {label}", ""]
        project_lines.extend(
            f"- [{title}]({filename}) — {date}" for date, title, filename in values
        )
        writes[archive_root / label / "README.md"] = "\n".join(project_lines).rstrip() + "\n"
    writes[archive_root / "README.md"] = "\n".join(root_lines).rstrip() + "\n"
    return writes


def plan_legacy_import(
    archive_root: Path,
    state: dict[str, Any],
    config_path: Path,
    *,
    timezone_name: str,
    observed_at: str,
) -> LegacyImportPlan:
    assert_chatgpt_state_local(state)
    approved = load_project_allowlist(config_path)
    labels = sorted(approved.values())
    if len(labels) != len(set(labels)):
        raise ChatGPTLegacyImportError("approved Project labels must be unique")
    label_to_id = {label: project_id for project_id, label in approved.items()}
    conversations = _discover_legacy(archive_root, label_to_id, timezone_name)
    source_state = state.get("chatgpt")
    if not isinstance(source_state, dict) or not isinstance(source_state.get("sessions"), dict):
        raise ChatGPTLegacyImportError("state lacks ChatGPT sessions")
    sessions = source_state["sessions"]

    for value in sessions.values():
        if not isinstance(value, dict):
            raise ChatGPTLegacyImportError("state contains an invalid ChatGPT session")
        project_id = str(value.get("project_id") or "")
        project_label = str(value.get("project_label") or "")
        if approved.get(project_id) != project_label:
            raise ChatGPTLegacyImportError(
                "state contains a ChatGPT session outside the approved Project mapping"
            )
        _existing_messages(archive_root, value)

    staged_state = deepcopy(state)
    staged_sessions = staged_state["chatgpt"]["sessions"]
    planned: list[PlannedConversation] = []
    reserved: set[Path] = set()
    for legacy in conversations:
        prior_value = sessions.get(legacy.conversation_id)
        prior = prior_value if isinstance(prior_value, dict) else None
        if prior is not None:
            if str(prior.get("project_id") or "") != legacy.project_id:
                raise ChatGPTLegacyImportError(
                    "legacy and Live Project assignments conflict for one conversation"
                )
            output_path, live_messages = _existing_messages(archive_root, prior)
            live_title, _live_date = _frontmatter_identity(output_path)
            merged, removed = _merge_live_wins(legacy.messages, live_messages)
            title = live_title
            overlap = True
            relative = output_path.relative_to(archive_root).as_posix()
            prior_revision = prior.get("thread_updated_at")
            try:
                thread_updated_at = max(float(prior_revision), legacy.updated_at.timestamp())
            except (TypeError, ValueError):
                thread_updated_at = legacy.updated_at.timestamp()
        else:
            project_dir = archive_root / legacy.project_label
            output_path = unique_output_path(
                project_dir,
                legacy.created_at.date().isoformat(),
                legacy.title,
                reserved_paths=reserved,
            )
            merged = legacy.messages
            removed = 0
            title = legacy.title
            overlap = False
            relative = output_path.relative_to(archive_root).as_posix()
            thread_updated_at = legacy.updated_at.timestamp()
        reserved.add(output_path)
        metadata = _message_metadata(merged)
        record = SessionRecord(
            source="chatgpt",
            session_id=legacy.conversation_id,
            title=title,
            date=legacy.created_at.date().isoformat(),
            messages=merged,
            models_used=sorted({turn.model for turn in merged if turn.model}),
        )
        state_entry = {
            "project_id": legacy.project_id,
            "project_label": legacy.project_label,
            "output_file": relative,
            "thread_updated_at": thread_updated_at,
            "coverage": "full_history",
            "branch_fingerprint": _branch_fingerprint(metadata),
            "messages": metadata,
            "user_complete": all(turn.complete for turn in merged if turn.role == "user"),
            "assistant_complete": all(
                turn.complete for turn in merged if turn.role == "assistant"
            ),
            "last_seen_at": observed_at,
            "last_input_kind": "legacy_markdown_import",
            "legacy_import": {
                "imported_at": observed_at,
                "message_count": len(legacy.messages),
                "merged_with_live": overlap,
                "timestamp_quality": "date_approximate_time_preserved",
            },
        }
        staged_sessions[legacy.conversation_id] = state_entry
        planned.append(
            PlannedConversation(
                legacy=legacy,
                output_path=output_path,
                output_relative=relative,
                record=record,
                state_entry=state_entry,
                overlap_with_live=overlap,
                duplicate_legacy_turns_removed=removed,
            )
        )
    indexes = _index_writes(archive_root, labels, planned, staged_sessions)
    return LegacyImportPlan(planned, staged_state, indexes)


def import_report(plan: LegacyImportPlan, *, applied: bool) -> dict[str, Any]:
    return {
        "source": "chatgpt",
        "mode": "legacy_markdown_import",
        "applied": applied,
        "legacy_conversations": len(plan.conversations),
        "legacy_messages": sum(len(item.legacy.messages) for item in plan.conversations),
        "unique_sessions_after_import": len(plan.state["chatgpt"]["sessions"]),
        "live_overlaps": sum(item.overlap_with_live for item in plan.conversations),
        "duplicate_legacy_turns_removed": sum(
            item.duplicate_legacy_turns_removed for item in plan.conversations
        ),
        "projects": dict(
            sorted(Counter(item.legacy.project_label for item in plan.conversations).items())
        ),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.legacy-import.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _restore(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
        return
    temporary = path.with_name(f".{path.name}.legacy-import-rollback.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(previous)
    temporary.replace(path)


def apply_legacy_import(
    plan: LegacyImportPlan,
    state_file: Path,
    *,
    checkpoint_state: Callable[[dict[str, Any], Path], None] = save_state,
) -> dict[str, Any]:
    conversation_writes = {
        item.output_path: render_markdown(item.record) for item in plan.conversations
    }
    all_write_paths = set(conversation_writes).union(plan.index_writes)
    source_paths = {item.legacy.source_path for item in plan.conversations}
    affected_paths = all_write_paths.union(source_paths, {state_file})
    previous = {
        path: path.read_bytes() if path.is_file() else None for path in affected_paths
    }
    try:
        for path, rendered in conversation_writes.items():
            _atomic_write_verified(path, rendered)
        for path, rendered in plan.index_writes.items():
            _atomic_write_text(path, rendered)
        for item in plan.conversations:
            parsed = parse_marked_markdown(item.output_path)
            if len(parsed) != len(item.state_entry["messages"]):
                raise ChatGPTLegacyImportError(
                    "post-import archive/state count mismatch"
                )
        checkpoint_state(plan.state, state_file)
        for source_path in source_paths:
            if source_path not in conversation_writes:
                source_path.unlink()
    except Exception:
        for path in sorted(affected_paths, key=lambda item: len(item.parts), reverse=True):
            _restore(path, previous[path])
        raise

    return import_report(plan, applied=True)
