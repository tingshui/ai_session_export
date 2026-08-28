# Product Requirements Document — AI Session Export

## Background

AI assistants (OpenCode, Claude Code, Codex, ChatGPT Projects, Google Antigravity, Cursor, DeepSeek Harness, and Second Mind) expose session transcripts through different local stores or user-controlled exports. Developers who switch between them end up with conversation history scattered across formats that are hard to search, back up, or feed into downstream workflows.

This project collapses those source silos into one stable, human-readable Markdown archive that any text tool can index.

## Goal

Provide a single Python CLI that reads supported session transcripts and writes them into a unified, deterministic Markdown file format with YAML frontmatter. The output must be the contract: downstream readers should not need the source-native schema.

## Users

The primary users are individual developers who run multiple AI coding agents locally and want to archive, search, or re-process their own session history. A secondary audience is AI coding agents themselves, which can install this project as a skill (see `skill.md`) and invoke it on the user's behalf.

## Requirements

### Functional

1. Export sessions from OpenCode, Claude Code, Codex, approved ChatGPT Projects, Google Antigravity, Cursor, DeepSeek Harness, and Second Mind into Markdown files.
2. Each source writes into its own subdirectory under the configured base directory (`second_mind/`, `opencode/`, `claude_code/`, `codex/`, `antigravity/`, `cursor/`).
3. Support incremental export: persisted source-specific state ensures only new or changed sessions are written on repeat runs. Sources with independently mutable surfaces must isolate state per surface and session.
4. Support `--full` to ignore the cursor and re-export everything.
5. Support `--since-date YYYY-MM-DD` to bound exports by session date.
6. Support `--dry-run` to scan and report counts without writing files or mutating state.
7. Support `--source <name>` to export a single source, or `all` (default) for every source.
8. Provide deterministic file naming (`YYYYMMDD_<sanitized-title>.md`) with collision-safe suffixes.
9. Drop noise: tool-call echos, sub-agent chatter, sidechain events, and system-role messages are not exported.

### Non-Functional

1. **Deterministic output.** The same input session must always produce the same Markdown bytes.
2. **No network access.** All parsing is local against files on disk.
3. **Read-only on source data.** Adapters open databases read-only and never write back to source files.
4. **Zero heavy dependencies.** The runtime depends only on the Python standard library; `pytest` is only needed for development.
5. **Python 3.11+.**
6. **Public-repo safe.** Tests use only synthetic fixtures; no real emails, paths, or personal data are committed.
7. **Extensible.** Adding a new source is a single adapter file plus registration in two places (see `AGENTS.md`).

## Success Criteria

- Every supported source produces at least one valid Markdown file from its native data when run against a populated local machine.
- Repeat runs with no new sessions write zero new files (incremental export verified by state cursor).
- The exported Markdown is self-describing via frontmatter and renders correctly in any standard Markdown viewer.
- The full non-live test suite passes in CI.

## Supported Sources

| Source | Native Data Location | Incremental Cursor |
|---|---|---|
| OpenCode | `~/.local/share/opencode/opencode.db` (SQLite) | `opencode.last_session_time` (ms epoch) |
| Claude Code | `~/.claude/projects/**/*.jsonl` plus history at `~/.claude/history.jsonl` | `claude_code.last_timestamp` (ms epoch) |
| Codex | `~/.codex/sessions/**/*.jsonl`, `~/.codex/archived_sessions/*.jsonl`, and `~/.codex/session_index.jsonl` | Per-session latest timestamp, output filename, and source mtime |
| ChatGPT Projects | Transient Codex App snapshot or official export directory/zip | Per-thread Project ID, branch fingerprint, message IDs/hashes, and output filename |
| Google Antigravity | `~/.gemini/antigravity/brain/*/.system_generated/logs/transcript_full.jsonl` (2.0), `~/.gemini/antigravity-ide/brain/*/...` (IDE), and `~/.gemini/antigravity-cli/brain/*/...` (CLI) | Per-surface, per-session source fingerprint, parse status, latest timestamp, and output filename |
| Cursor | `~/Library/Application Support/Cursor/User/globalStorage/state.vscdb` (SQLite) | Per-session latest timestamp and output filename |
| Second Mind | `second_mind_export.json` (a single JSON array export) | `second_mind.last_export_count` |

All paths shown are defaults and can be overridden via CLI flags (`--opencode-db`, `--antigravity-dir`, `--codex-dir`, `--codex-session-index`, `--cursor-db`, `--second-mind-json`, and the base directory).

ChatGPT live snapshots are accepted only as complete `full_history` page chains.
An App `…N tokens truncated…` sentinel is treated as an incomplete conversation:
the adapter emits a failure warning and preserves the previous archive/state.
Official exports are the completeness path for conversations whose individual
items exceed the App read surface's output limit.

## Markdown Output Contract

Every exported session is one Markdown file. The file begins with a YAML frontmatter block delimited by `---`, followed by a blank line, an `H1` title, and then alternating `## User` / `## Assistant` sections. The exact shape is:

```markdown
---
source: <source-name>
surface: "<optional product surface>"
session_id: "<opaque id>"
title: "<session title>"
date: "YYYY-MM-DD"
message_count: <int>
project_directory: "<path>"
models_used: ["model-a", "model-b"]
turn_models: ["model-a", "model-a"]
---
# <session title>

## User [HH:MM]

<user message content>

## Assistant [HH:MM]

<assistant message content>
```

Field rules:

- `source` is one of `opencode`, `claude_code`, `codex`, `chatgpt`, `antigravity`, `cursor`, `dsh`, `second_mind`.
- `surface` is optional and is currently emitted only for Antigravity. Its stable values are `"2"`, `"ide"`, and `"cli"`; custom single-root overrides are treated as `"ide"` for backward compatibility.
- All string frontmatter values are JSON-quoted so YAML-special characters are safe.
- `message_count` is the count of exported turns (after noise filtering).
- `project_directory` is emitted only when non-empty (Antigravity and Second Mind never set it).
- `models_used` is emitted only when non-empty; Antigravity leaves it empty because model identity is not exposed in its transcript steps.
- `turn_models` is emitted when at least one turn has attributable model identity. It is a JSON array aligned one-to-one with all rendered turn sections, uses `null` for unknown entries, and is omitted when every entry is unknown. Consumers must not infer per-turn attribution from `models_used`.
- Each turn header is `## User` or `## Assistant`. When a per-turn timestamp is known, it is appended as `[HH:MM]` in local time.
- The file is single-trailing-newline terminated; trailing whitespace is stripped from each message body.
- Sources that expose stable message IDs may emit an invisible HTML turn marker containing the ID and body hash immediately before the corresponding heading. The visible heading/body contract remains unchanged.

This contract is the project's stability boundary. It must not change without an explicit decision recorded in `docs/working.md`.

## Non-Goals

- **No live editing or writing back** to any source's native store.
- **No UI.** Output is Markdown files on disk; rendering is delegated to the user's editor or static-site tool.
- **No consolidation across sources.** Sessions stay in per-source subdirectories; there is no cross-source deduplication or merge.
- **No cloud sync.** The tool never touches the network.
- **No schema discovery for opaque binary formats.** Sources that store transcripts in binary protobuf without a published schema (for example Antigravity `.pb` files) are not reverse-engineered; only the documented JSONL transcript is parsed.
- **No re-encoding into other formats** (PDF, DOCX, etc.). Markdown is the only output.
