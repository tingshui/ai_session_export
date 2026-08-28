from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from ..markdown import render_markdown
from ..models import MessageTurn, SessionRecord
from ..utils import sanitize_filename, unique_output_path


PROJECT_ID_RE = re.compile(r"^g-p-[A-Za-z0-9]+$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
OFFICIAL_MEMBER_RE = re.compile(r"^conversations(?:-[^.]+)?\.json$")
MAX_OFFICIAL_MEMBER_BYTES = 512 * 1024 * 1024
APP_TRUNCATION_SENTINEL_RE = re.compile(
    r"(?:…|\.{3})\s*\d+\s+tokens?\s+truncated\s*(?:…|\.{3})",
    re.IGNORECASE,
)
LIVE_DISCOVERY_SOURCE = "chatgpt_app"


class ChatGPTExportError(RuntimeError):
    pass


class ParsedChatGPTConversation(NamedTuple):
    record: SessionRecord
    project_id: str
    project_label: str
    thread_updated_at: int | float | str | None
    coverage: str
    branch_fingerprint: str
    message_metadata: list[dict[str, Any]]
    user_complete: bool
    assistant_complete: bool
    input_kind: str


def _require_id(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not SAFE_ID_RE.fullmatch(text):
        raise ChatGPTExportError(f"invalid {field}")
    return text


def _diagnostic_id(value: Any) -> str:
    text = str(value or "").strip()
    return text if SAFE_ID_RE.fullmatch(text) else "unknown"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _official_input_sha256(source: Path) -> str:
    digest = hashlib.sha256()
    if source.is_file():
        digest.update(source.read_bytes())
        return digest.hexdigest()
    if source.is_dir():
        members = sorted(
            path
            for path in source.rglob("*.json")
            if OFFICIAL_MEMBER_RE.fullmatch(path.name)
        )
        for path in members:
            digest.update(path.relative_to(source).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
        return digest.hexdigest()
    raise ChatGPTExportError(f"official ChatGPT export not found: {source}")


def _official_seed(source_state: dict[str, Any]) -> dict[str, Any]:
    value = source_state.get("official_seed")
    if value is None:
        return {"status": "unused"}
    if not isinstance(value, dict):
        raise ChatGPTExportError("invalid ChatGPT official seed state")
    status = value.get("status")
    if status not in {"unused", "in_progress", "completed"}:
        raise ChatGPTExportError("invalid ChatGPT official seed status")
    return dict(value)


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


def _epoch_ms(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return int(numeric if numeric >= 1_000_000_000_000 else numeric * 1000)


def _date_from_epoch_ms(value: int | None) -> str:
    if value is None:
        return "1970-01-01"
    return datetime.fromtimestamp(value / 1000).date().isoformat()


def load_project_allowlist(config_path: Path) -> dict[str, str]:
    if not config_path.is_file():
        raise ChatGPTExportError(f"ChatGPT Project config not found: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChatGPTExportError(f"invalid ChatGPT Project config: {error}") from error
    if config.get("schema_version") != 1 or not isinstance(config.get("projects"), dict):
        raise ChatGPTExportError("unsupported ChatGPT Project config")
    projects: dict[str, str] = {}
    for project_id, policy in config["projects"].items():
        if not PROJECT_ID_RE.fullmatch(str(project_id)):
            raise ChatGPTExportError(f"invalid approved project ID: {project_id!r}")
        if not isinstance(policy, dict):
            raise ChatGPTExportError(f"invalid project policy: {project_id}")
        label = str(policy.get("label") or "").strip()
        if not label:
            raise ChatGPTExportError(f"approved project lacks label: {project_id}")
        projects[str(project_id)] = label
    if not projects:
        raise ChatGPTExportError("ChatGPT Project allowlist is empty")
    return projects


def _load_json_array(text: str, source_name: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ChatGPTExportError(f"invalid JSON in {source_name}: {error}") from error
    if not isinstance(payload, list):
        raise ChatGPTExportError(f"{source_name} must contain a JSON array")
    if not all(isinstance(item, dict) for item in payload):
        raise ChatGPTExportError(f"{source_name} contains a non-object conversation")
    return payload


def load_official_conversations(source: Path) -> list[dict[str, Any]]:
    conversations: list[dict[str, Any]] = []
    if source.is_dir():
        files = sorted(
            path
            for path in source.rglob("*.json")
            if OFFICIAL_MEMBER_RE.fullmatch(path.name)
        )
        if not files:
            raise ChatGPTExportError(f"no conversations JSON found in {source}")
        for path in files:
            conversations.extend(_load_json_array(path.read_text(encoding="utf-8"), path.name))
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = sorted(
                (
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and OFFICIAL_MEMBER_RE.fullmatch(Path(info.filename).name)
                ),
                key=lambda info: info.filename,
            )
            if not members:
                raise ChatGPTExportError(f"no conversations JSON found in {source}")
            for member in members:
                if member.file_size > MAX_OFFICIAL_MEMBER_BYTES:
                    raise ChatGPTExportError(f"official export member is too large: {member.filename}")
                try:
                    text = archive.read(member).decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ChatGPTExportError(
                        f"official export member is not UTF-8: {member.filename}"
                    ) from error
                conversations.extend(_load_json_array(text, member.filename))
    else:
        raise ChatGPTExportError(f"official ChatGPT export not found: {source}")

    seen: set[str] = set()
    for conversation in conversations:
        thread_id = str(conversation.get("conversation_id") or conversation.get("id") or "").strip()
        if not thread_id:
            raise ChatGPTExportError("official export conversation lacks ID")
        if thread_id in seen:
            raise ChatGPTExportError(f"duplicate official export conversation ID: {thread_id}")
        seen.add(thread_id)
    return conversations


def _render_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return str(part)
    content_type = part.get("content_type") or part.get("type")
    if content_type in {"image_asset_pointer", "image"}:
        return f"[Image: {part.get('asset_pointer') or part.get('image_url') or ''}]"
    if content_type == "audio_asset_pointer":
        return f"[Audio: {part.get('asset_pointer') or ''}]"
    if content_type == "video_container_asset_pointer":
        return f"[Video: {part.get('asset_pointer') or ''}]"
    if "text" in part:
        return str(part["text"])
    return f"[{content_type or 'unknown'}]"


def _official_message_body(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    if not isinstance(content, dict):
        raise ChatGPTExportError("official message content must be an object")
    content_type = content.get("content_type")
    if content_type in {"user_editable_context", "system_error"}:
        return ""
    if content_type == "code":
        language = str(content.get("language") or "")
        return f"```{language}\n{content.get('text') or ''}\n```".strip()
    if content_type == "execution_output":
        return f"```\n{content.get('text') or ''}\n```".strip()
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        raise ChatGPTExportError("official message parts must be an array")
    return "\n\n".join(_render_part(part) for part in parts if part is not None).strip()


def _linearize_official(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    mapping = conversation.get("mapping")
    current_node = conversation.get("current_node")
    if not isinstance(mapping, dict) or not isinstance(current_node, str) or not current_node:
        raise ChatGPTExportError("official conversation lacks mapping/current_node")
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    node_id: str | None = current_node
    while node_id:
        if node_id in seen:
            raise ChatGPTExportError(f"cycle in official conversation at node {node_id}")
        seen.add(node_id)
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            raise ChatGPTExportError(f"broken official conversation parent chain at {node_id}")
        chain.append(node)
        parent = node.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise ChatGPTExportError(f"invalid parent ID at node {node_id}")
        node_id = parent
    chain.reverse()
    return chain


def _message_metadata(messages: list[MessageTurn]) -> list[dict[str, Any]]:
    return [
        {
            "message_id": message.message_id,
            "role": message.role,
            "created_at": message.time_created,
            "content_sha256": _sha256(message.content.rstrip()),
            "complete": message.complete,
        }
        for message in messages
    ]


def _branch_fingerprint(metadata: list[dict[str, Any]]) -> str:
    canonical = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return _sha256(canonical)


def parse_official_conversation(
    conversation: dict[str, Any], project_id: str, project_label: str
) -> ParsedChatGPTConversation:
    thread_id = _require_id(
        conversation.get("conversation_id") or conversation.get("id"), "thread_id"
    )
    messages: list[MessageTurn] = []
    models: set[str] = set()
    seen_message_ids: set[str] = set()
    for node in _linearize_official(conversation):
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        author = message.get("author") or {}
        role = str(author.get("role") or "").strip().lower() if isinstance(author, dict) else ""
        if role not in {"user", "assistant"}:
            continue
        body = _official_message_body(message)
        if not body:
            continue
        message_id = _require_id(message.get("id"), "message_id")
        if message_id in seen_message_ids:
            raise ChatGPTExportError(f"duplicate official message ID: {message_id}")
        seen_message_ids.add(message_id)
        metadata = message.get("metadata") or {}
        model = None
        if isinstance(metadata, dict):
            model = str(metadata.get("model_slug") or "").strip() or None
        if model:
            models.add(model)
        messages.append(
            MessageTurn(
                role=role,
                content=body,
                time_created=_epoch_ms(message.get("create_time")),
                model=model,
                message_id=message_id,
            )
        )
    if not messages:
        raise ChatGPTExportError(f"official conversation has no visible messages: {thread_id}")
    started_at = _epoch_ms(conversation.get("create_time")) or messages[0].time_created
    title = str(conversation.get("title") or "").strip() or messages[0].content.splitlines()[0][:120]
    metadata = _message_metadata(messages)
    return ParsedChatGPTConversation(
        record=SessionRecord(
            source="chatgpt",
            session_id=thread_id,
            title=title,
            date=_date_from_epoch_ms(started_at),
            messages=messages,
            models_used=sorted(models),
        ),
        project_id=project_id,
        project_label=project_label,
        thread_updated_at=conversation.get("update_time"),
        coverage="full_history",
        branch_fingerprint=_branch_fingerprint(metadata),
        message_metadata=metadata,
        user_complete=True,
        assistant_complete=True,
        input_kind="official_export",
    )


def _live_item_body(item: dict[str, Any], role: str) -> tuple[str, bool]:
    item_type = item.get("type")
    body = ""
    if item_type == "agentMessage":
        text = item.get("text")
        if isinstance(text, str):
            body = text.strip()
    if not body:
        content = item.get("content")
        if isinstance(content, str):
            body = content.strip()
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") in {
                    "text",
                    "input_text",
                    "output_text",
                }:
                    parts.append(str(part.get("text") or ""))
            body = "\n\n".join(part for part in parts if part).strip()
    explicit_complete = item.get("complete", True)
    if not isinstance(explicit_complete, bool):
        raise ChatGPTExportError("live item complete must be boolean")
    complete = explicit_complete and not bool(APP_TRUNCATION_SENTINEL_RE.search(body))
    if not complete and role == "user":
        raise ChatGPTExportError("live item contains incomplete user content")
    if not complete:
        marker = "[INCOMPLETE ASSISTANT CONTENT: source was truncated]"
        body = f"{marker}\n\n{body}" if body else marker
    return body, complete


def _validate_full_history_pagination(thread: dict[str, Any], thread_id: str) -> None:
    proof = thread.get("pagination")
    if not isinstance(proof, dict):
        raise ChatGPTExportError(f"live thread lacks pagination proof: {thread_id}")
    pages = proof.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ChatGPTExportError(f"pagination.pages must be a non-empty array: {thread_id}")
    expected_cursor = None
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ChatGPTExportError(f"pagination page must be an object: {thread_id}")
        if page.get("cursor_in") != expected_cursor:
            raise ChatGPTExportError(
                f"pagination cursor chain breaks at page {index}: {thread_id}"
            )
        has_more = page.get("has_more")
        cursor_out = page.get("cursor_out")
        if not isinstance(has_more, bool):
            raise ChatGPTExportError(f"pagination has_more must be boolean: {thread_id}")
        if has_more:
            expected_cursor = _require_id(cursor_out, "pagination cursor")
        elif cursor_out is not None:
            raise ChatGPTExportError(
                f"terminal pagination page must have null cursor_out: {thread_id}"
            )
        else:
            expected_cursor = None
        if index < len(pages) - 1 and not has_more:
            raise ChatGPTExportError(
                f"pagination continues after terminal page: {thread_id}"
            )
    if proof.get("terminal_reason") != "end" or pages[-1].get("has_more") is not False:
        raise ChatGPTExportError(
            f"full_history requires terminal has_more=false: {thread_id}"
        )


def parse_live_thread(
    thread: dict[str, Any], project_id: str, project_label: str
) -> ParsedChatGPTConversation:
    thread_id = _require_id(thread.get("thread_id") or thread.get("id"), "thread_id")
    if not isinstance(thread.get("complete"), bool):
        raise ChatGPTExportError(f"live thread complete must be boolean: {thread_id}")
    coverage = str(thread.get("coverage") or "")
    if coverage != "full_history":
        raise ChatGPTExportError(
            f"live thread coverage is not full_history: {thread_id} ({coverage or 'missing'})"
        )
    _validate_full_history_pagination(thread, thread_id)
    turns = thread.get("turns")
    if not isinstance(turns, list):
        raise ChatGPTExportError(f"live thread turns must be an array: {thread_id}")
    messages: list[MessageTurn] = []
    models: set[str] = set()
    seen: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict):
            raise ChatGPTExportError(f"live thread contains non-object turn: {thread_id}")
        turn_timestamp = _epoch_ms(turn.get("startedAt") or turn.get("created_at"))
        turn_model = str(turn.get("model") or "").strip() or None
        items = turn.get("items")
        if not isinstance(items, list):
            raise ChatGPTExportError(f"live turn items must be an array: {thread_id}")
        for item in items:
            if not isinstance(item, dict):
                raise ChatGPTExportError(f"live turn contains non-object item: {thread_id}")
            item_type = item.get("type")
            role = "user" if item_type == "userMessage" else "assistant" if item_type == "agentMessage" else ""
            if not role:
                continue
            body, complete = _live_item_body(item, role)
            if not body:
                continue
            message_id = _require_id(item.get("id"), "message_id")
            if message_id in seen:
                raise ChatGPTExportError(f"duplicate live message ID: {message_id}")
            seen.add(message_id)
            model = str(item.get("model") or turn_model or "").strip() or None
            if model:
                models.add(model)
            messages.append(
                MessageTurn(
                    role=role,
                    content=body,
                    time_created=_epoch_ms(item.get("created_at") or item.get("createdAt"))
                    or turn_timestamp,
                    model=model,
                    message_id=message_id,
                    complete=complete,
                )
            )
    if not messages:
        raise ChatGPTExportError(f"live thread has no visible messages: {thread_id}")
    title = str(thread.get("title") or "").strip() or messages[0].content.splitlines()[0][:120]
    started_at = _epoch_ms(thread.get("created_at") or thread.get("createdAt")) or messages[0].time_created
    metadata = _message_metadata(messages)
    user_complete = all(
        message.complete for message in messages if message.role == "user"
    )
    assistant_complete = all(
        message.complete for message in messages if message.role == "assistant"
    )
    if thread.get("complete") is False and user_complete and assistant_complete:
        raise ChatGPTExportError(
            f"incomplete live thread lacks role-specific evidence: {thread_id}"
        )
    return ParsedChatGPTConversation(
        record=SessionRecord(
            source="chatgpt",
            session_id=thread_id,
            title=title,
            date=_date_from_epoch_ms(started_at),
            messages=messages,
            models_used=sorted(models),
        ),
        project_id=project_id,
        project_label=project_label,
        thread_updated_at=thread.get("updated_at") or thread.get("updatedAt"),
        coverage=coverage,
        branch_fingerprint=_branch_fingerprint(metadata),
        message_metadata=metadata,
        user_complete=user_complete,
        assistant_complete=assistant_complete,
        input_kind="live_snapshot",
    )


def parse_live_snapshot(
    payload: dict[str, Any], allowlist: dict[str, str]
) -> tuple[list[ParsedChatGPTConversation], int, list[dict[str, str]]]:
    if payload.get("schema_version") != 1:
        raise ChatGPTExportError("unsupported live ChatGPT snapshot")
    observed_at = payload.get("observed_at")
    if not isinstance(observed_at, str) or not observed_at.strip():
        raise ChatGPTExportError("live snapshot observed_at must be an ISO-8601 timestamp")
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ChatGPTExportError(
            "live snapshot observed_at must be an ISO-8601 timestamp"
        ) from error
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ChatGPTExportError("live snapshot observed_at must include a timezone")

    discovery = payload.get("discovery")
    if not isinstance(discovery, dict):
        raise ChatGPTExportError("live snapshot discovery must be an object")
    if discovery.get("source") != LIVE_DISCOVERY_SOURCE:
        raise ChatGPTExportError("live snapshot discovery source is unsupported")

    def require_count(field: str, *, positive: bool = False) -> int:
        value = discovery.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ChatGPTExportError(f"live snapshot discovery {field} must be an integer")
        minimum = 1 if positive else 0
        if value < minimum:
            qualifier = "positive" if positive else "non-negative"
            raise ChatGPTExportError(
                f"live snapshot discovery {field} must be {qualifier}"
            )
        return value

    returned = require_count("non_pinned_returned")
    limit = require_count("non_pinned_limit", positive=True)
    approved_pinned = require_count("approved_pinned_threads")
    if returned > limit:
        raise ChatGPTExportError(
            "live snapshot discovery non_pinned_returned exceeds non_pinned_limit"
        )
    saturated = discovery.get("recent_window_saturated")
    if not isinstance(saturated, bool):
        raise ChatGPTExportError(
            "live snapshot discovery recent_window_saturated must be boolean"
        )
    if saturated != (returned >= limit):
        raise ChatGPTExportError(
            "live snapshot discovery recent_window_saturated is inconsistent"
        )

    projects = payload.get("projects")
    if not isinstance(projects, list):
        raise ChatGPTExportError("live snapshot projects must be an array")
    approved_projects: list[tuple[str, str, dict[str, Any]]] = []
    seen_projects: set[str] = set()
    ignored = 0
    for project in projects:
        if not isinstance(project, dict):
            raise ChatGPTExportError("live snapshot contains non-object project")
        project_id = project.get("project_id")
        if project_id is None or project_id == "":
            ignored += 1
            continue
        if not isinstance(project_id, str) or not PROJECT_ID_RE.fullmatch(project_id):
            raise ChatGPTExportError("live snapshot contains malformed project identifier")
        label = allowlist.get(project_id)
        if label is None:
            ignored += 1
            continue
        if project_id in seen_projects:
            raise ChatGPTExportError(f"duplicate approved project: {project_id}")
        seen_projects.add(project_id)
        approved_projects.append((project_id, label, project))
    missing_projects = sorted(set(allowlist) - seen_projects)
    if missing_projects:
        raise ChatGPTExportError(
            "live snapshot missing approved projects: " + ",".join(missing_projects)
        )
    approved_thread_count = 0
    for project_id, _label, project in approved_projects:
        threads = project.get("threads")
        if not isinstance(threads, list):
            raise ChatGPTExportError(
                f"approved project threads must be an array: {project_id}"
            )
        approved_thread_count += len(threads)
    if approved_pinned > approved_thread_count:
        raise ChatGPTExportError(
            "approved_pinned_threads exceeds approved thread population"
        )

    parsed: list[ParsedChatGPTConversation] = []
    warnings: list[dict[str, str]] = []
    seen_threads: set[str] = set()
    for project_id, label, project in approved_projects:
        threads = project.get("threads")
        if not isinstance(threads, list):
            raise ChatGPTExportError(f"approved project threads must be an array: {project_id}")
        for thread in threads:
            if not isinstance(thread, dict):
                warnings.append({"thread_id": "unknown", "error": "thread must be an object"})
                continue
            candidate_id = _diagnostic_id(
                thread.get("thread_id") or thread.get("id")
            )
            try:
                conversation = parse_live_thread(thread, project_id, label)
                if conversation.record.session_id in seen_threads:
                    raise ChatGPTExportError(
                        f"duplicate live thread ID: {conversation.record.session_id}"
                    )
                seen_threads.add(conversation.record.session_id)
                parsed.append(conversation)
            except ChatGPTExportError as error:
                warnings.append({"thread_id": candidate_id, "error": str(error)})
    return parsed, ignored, warnings


def parse_official_snapshot(
    conversations: list[dict[str, Any]], allowlist: dict[str, str]
) -> tuple[list[ParsedChatGPTConversation], int, list[dict[str, str]]]:
    parsed: list[ParsedChatGPTConversation] = []
    ignored = 0
    warnings: list[dict[str, str]] = []
    for conversation in conversations:
        project_id = str(conversation.get("conversation_template_id") or "")
        label = allowlist.get(project_id)
        if label is None:
            ignored += 1
            continue
        candidate_id = _diagnostic_id(
            conversation.get("conversation_id") or conversation.get("id") or "unknown"
        )
        try:
            parsed.append(parse_official_conversation(conversation, project_id, label))
        except ChatGPTExportError as error:
            warnings.append({"thread_id": candidate_id, "error": str(error)})
    return parsed, ignored, warnings


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    if path.read_text(encoding="utf-8") != content:
        raise ChatGPTExportError(f"Markdown reread verification failed: {path}")


def _restore_archive_file(path: Path, previous: bytes | None) -> None:
    if previous is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.rollback.tmp")
    temporary.write_bytes(previous)
    temporary.replace(path)


def export_chatgpt(
    output_dir: Path,
    state: dict[str, Any],
    *,
    source_input: Path,
    project_config: Path,
    full: bool,
    dry_run: bool,
    since_date: date | None,
    stdin_text: str | None = None,
    checkpoint_state: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    allowlist = load_project_allowlist(project_config)
    is_live = str(source_input) == "-"
    source_state = state.get("chatgpt", {})
    existing_sessions = source_state.get("sessions", {})
    if not isinstance(existing_sessions, dict):
        raise ChatGPTExportError("invalid ChatGPT exporter state")
    seed = _official_seed(source_state)
    official_input_sha256 = None
    seed_started_at = None
    if not is_live and not dry_run:
        if seed["status"] == "completed":
            raise ChatGPTExportError("Official export writer is permanently closed")
        official_input_sha256 = _official_input_sha256(source_input)
        if (
            seed["status"] == "in_progress"
            and seed.get("input_sha256") != official_input_sha256
        ):
            raise ChatGPTExportError(
                "Official export seed is already in progress for another input"
            )
        seed_started_at = str(
            seed.get("started_at") or datetime.now(timezone.utc).isoformat()
        )
        state["chatgpt"] = {
            **source_state,
            "sessions": dict(existing_sessions),
            "official_seed": {
                "status": "in_progress",
                "input_sha256": official_input_sha256,
                "started_at": seed_started_at,
            },
        }
        if checkpoint_state is not None:
            checkpoint_state(state)

    if is_live:
        if stdin_text is None:
            raise ChatGPTExportError("live ChatGPT snapshot stdin is empty")
        try:
            payload = json.loads(stdin_text)
        except json.JSONDecodeError as error:
            raise ChatGPTExportError(f"invalid live ChatGPT snapshot JSON: {error}") from error
        if not isinstance(payload, dict):
            raise ChatGPTExportError("live ChatGPT snapshot must be an object")
        parsed, ignored, warnings = parse_live_snapshot(payload, allowlist)
        observed_at = str(payload.get("observed_at") or datetime.now(timezone.utc).isoformat())
    else:
        try:
            parsed, ignored, warnings = parse_official_snapshot(
                load_official_conversations(source_input), allowlist
            )
        except Exception:
            if not dry_run:
                state["chatgpt"] = {
                    **source_state,
                    "sessions": dict(existing_sessions),
                    "official_seed": {"status": "unused"},
                }
                if checkpoint_state is not None:
                    checkpoint_state(state)
            raise
        observed_at = datetime.now(timezone.utc).isoformat()

    scanned = len(parsed) + len(warnings)
    if not is_live and warnings and not dry_run:
        state["chatgpt"] = {
            **source_state,
            "sessions": dict(existing_sessions),
            "official_seed": {"status": "unused"},
        }
        if checkpoint_state is not None:
            checkpoint_state(state)
        return {
            "source": "chatgpt",
            "scanned": scanned,
            "exported": 0,
            "failed": len(warnings),
            "ignored": ignored,
            "warnings": warnings,
        }

    updated_sessions = dict(existing_sessions)
    exported = 0
    official_writes: list[tuple[Path, str]] = []
    reserved_output_paths: set[Path] = set()

    for conversation in parsed:
        if since_date and date.fromisoformat(conversation.record.date) < since_date:
            continue
        previous = existing_sessions.get(conversation.record.session_id, {})
        if not isinstance(previous, dict):
            previous = {}
        previous_kind = previous.get("last_input_kind")
        if not is_live and previous_kind == "live_snapshot":
            continue
        if is_live and previous_kind == "live_snapshot":
            previous_revision = _revision_number(previous.get("thread_updated_at"))
            incoming_revision = _revision_number(conversation.thread_updated_at)
            branch_changed = (
                previous.get("branch_fingerprint")
                != conversation.branch_fingerprint
            )
            revision_regressed = (
                previous_revision is not None
                and incoming_revision is not None
                and incoming_revision < previous_revision
            )
            incoming_revision_is_ambiguous = (
                previous_revision is not None and incoming_revision is None
            )
            changed_branch_is_not_newer = branch_changed and (
                previous_revision is None
                or incoming_revision is None
                or incoming_revision <= previous_revision
            )
            if (
                revision_regressed
                or incoming_revision_is_ambiguous
                or changed_branch_is_not_newer
            ):
                warnings.append(
                    {
                        "thread_id": conversation.record.session_id,
                        "error": (
                            "live revision is older than verified archive state"
                            if revision_regressed
                            else "live revision authority is ambiguous"
                        ),
                    }
                )
                continue
        previous_output = str(previous.get("output_file") or "")
        output_path = output_dir / previous_output if previous_output else None
        output_exists = bool(output_path and output_path.is_file())
        unchanged = (
            not full
            and output_exists
            and previous.get("project_id") == conversation.project_id
            and previous.get("branch_fingerprint") == conversation.branch_fingerprint
        )
        if unchanged:
            continue

        project_slug = sanitize_filename(conversation.project_label)
        project_dir = output_dir / project_slug
        if output_path is None:
            output_path = unique_output_path(
                project_dir,
                conversation.record.date,
                conversation.record.title,
                reserved_paths=reserved_output_paths,
            )
        reserved_output_paths.add(output_path)
        rendered = render_markdown(conversation.record)
        if not dry_run:
            if is_live:
                _atomic_write_text(output_path, rendered)
            else:
                official_writes.append((output_path, rendered))
            updated_sessions[conversation.record.session_id] = {
                "project_id": conversation.project_id,
                "project_label": conversation.project_label,
                "output_file": output_path.relative_to(output_dir).as_posix(),
                "thread_updated_at": conversation.thread_updated_at,
                "coverage": conversation.coverage,
                "branch_fingerprint": conversation.branch_fingerprint,
                "messages": conversation.message_metadata,
                "user_complete": conversation.user_complete,
                "assistant_complete": conversation.assistant_complete,
                "last_seen_at": observed_at,
                "last_input_kind": conversation.input_kind,
            }
        exported += 1

    if not is_live and not dry_run:
        previous_files: list[tuple[Path, bytes | None]] = []
        try:
            for path, rendered in official_writes:
                previous_files.append(
                    (path, path.read_bytes() if path.is_file() else None)
                )
                _atomic_write_text(path, rendered)
            state["chatgpt"] = {
                **source_state,
                "sessions": updated_sessions,
                "official_seed": {
                    "status": "completed",
                    "input_sha256": official_input_sha256,
                    "started_at": seed_started_at,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            if checkpoint_state is not None:
                checkpoint_state(state)
        except Exception:
            for path, previous in reversed(previous_files):
                _restore_archive_file(path, previous)
            state["chatgpt"] = {
                **source_state,
                "sessions": dict(existing_sessions),
                "official_seed": {"status": "unused"},
            }
            if checkpoint_state is not None:
                checkpoint_state(state)
            raise

    if is_live and not dry_run:
        final_state = {**source_state, "sessions": updated_sessions}
        if "official_seed" in source_state:
            final_state["official_seed"] = seed
        state["chatgpt"] = final_state
        if checkpoint_state is not None:
            checkpoint_state(state)

    return {
        "source": "chatgpt",
        "scanned": scanned,
        "exported": exported,
        "failed": len(warnings),
        "ignored": ignored,
        "warnings": warnings,
    }
