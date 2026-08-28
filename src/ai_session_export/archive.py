from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import NamedTuple


TURN_MARKER_PREFIX = "<!-- ai-session-export-turn: "
TURN_MARKER_ESCAPED_PREFIX = "<!-- ai-session-export-turn-escaped: "
TURN_MARKER_RE = re.compile(r"(?m)^<!-- ai-session-export-turn: (\{.*\}) -->\n")
TURN_HEADING_RE = re.compile(r"^## (User|Assistant)(?: \[(\d{2}:\d{2})\])?$")


class MarkedTurn(NamedTuple):
    role: str
    content: str
    message_id: str
    content_sha256: str
    complete: bool


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.rstrip().encode("utf-8")).hexdigest()


def escape_turn_marker_content(content: str) -> str:
    return content.replace(TURN_MARKER_PREFIX, TURN_MARKER_ESCAPED_PREFIX)


def unescape_turn_marker_content(content: str) -> str:
    return content.replace(TURN_MARKER_ESCAPED_PREFIX, TURN_MARKER_PREFIX)


def parse_marked_markdown_text(text: str) -> list[MarkedTurn]:
    matches = list(TURN_MARKER_RE.finditer(text))
    if not matches:
        raise ValueError("Markdown contains no ChatGPT turn markers")
    turns: list[MarkedTurn] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        try:
            marker = json.loads(match.group(1))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid ChatGPT turn marker JSON: {error}") from error
        if not isinstance(marker, dict):
            raise ValueError("ChatGPT turn marker must be an object")
        message_id = str(marker.get("message_id") or "").strip()
        expected_sha256 = str(marker.get("sha256") or "").strip().lower()
        complete = marker.get("complete", True)
        if not message_id or message_id in seen:
            raise ValueError(f"missing or duplicate ChatGPT message ID: {message_id!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"invalid ChatGPT content hash: {message_id}")
        if not isinstance(complete, bool):
            raise ValueError(f"invalid ChatGPT completeness marker: {message_id}")
        seen.add(message_id)

        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end() : section_end]
        heading, separator, body = section.partition("\n")
        heading_match = TURN_HEADING_RE.fullmatch(heading)
        if not heading_match or not separator:
            raise ValueError(f"missing ChatGPT turn heading: {message_id}")
        if not body.startswith("\n"):
            raise ValueError(f"missing blank line after ChatGPT turn heading: {message_id}")
        logical_body = unescape_turn_marker_content(body[1:].rstrip())
        actual_sha256 = content_sha256(logical_body)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"ChatGPT turn hash mismatch: {message_id}")
        turns.append(
            MarkedTurn(
                role=heading_match.group(1).lower(),
                content=logical_body,
                message_id=message_id,
                content_sha256=actual_sha256,
                complete=complete,
            )
        )
    return turns


def parse_marked_markdown(path: Path) -> list[MarkedTurn]:
    return parse_marked_markdown_text(path.read_text(encoding="utf-8"))
