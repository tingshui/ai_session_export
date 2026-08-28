# AI Session Archive Context

This context defines how native assistant conversations become durable Markdown archives and how ChatGPT Project material crosses into the separate knowledge-ingestion pipeline.

## Sources and scope

**Source adapter**:
A deterministic, source-specific reader that converts native session data into the shared one-session-per-file Markdown contract.
_Avoid_: Connector, importer

**ChatGPT Project**:
A ChatGPT conversation container identified by a stable `g-p-...` Project ID.
_Avoid_: Folder, workspace

**Approved Project**:
A ChatGPT Project whose exact ID appears in the user-approved allowlist. Only conversations in Approved Projects are eligible for ChatGPT Markdown export or knowledge ingestion.
_Avoid_: Known Project, discovered Project

**Project-scoped archive**:
The ChatGPT Markdown archive containing every eligible conversation in Approved Projects and no Projectless or unapproved conversations.
_Avoid_: Full ChatGPT archive, all-chat archive

## Downstream boundaries

**Archive export**:
Lossless conversion of eligible user and assistant conversation turns into the shared Markdown contract for browsing, search, and backup.
_Avoid_: Reflection, extraction

**Project ingestion**:
The separate allowlisted pipeline that treats only user-authored turns as reflection evidence and writes distilled key ideas to approved thought-review domains.
_Avoid_: Archive export, session export

**Project allowlist**:
The single user-approved mapping from stable ChatGPT Project IDs to project metadata and allowed knowledge domains, shared by archive export and Project ingestion.
_Avoid_: Folder mapping, discovered projects

## ChatGPT inputs

**Live snapshot**:
A complete, transient representation of the currently selected branches of recently discovered Approved Project conversations, supplied by the authenticated Codex App automation over stdin.
_Avoid_: Live API, cached chat dump

**Truncated live item**:
An App response that contains a truncation sentinel or otherwise states that a message body was shortened. It is not a complete message and cannot be archived as `full_history`; the Official export snapshot is its recovery path.
_Avoid_: Partial success, best-effort body

**Official export snapshot**:
An untouched OpenAI data-export zip or directory used to reconcile historical Approved Project conversations beyond the live discovery window.
_Avoid_: Backup database, incremental export

**Historical reconciliation**:
Processing an Official export snapshot through the same ChatGPT adapter and state contract to fill archive gaps without creating a second Markdown writer.
_Avoid_: Second import pipeline, legacy conversion

## Consumer boundary

**Archive metadata state**:
The raw-text-free per-conversation ledger that binds each Markdown file to Project ID, thread ID, selected-branch fingerprint, message IDs, roles, timestamps, and content hashes.
_Avoid_: Ingest state, raw snapshot cache

**Archive consumer**:
A downstream process that reads ChatGPT Markdown only after verifying its turns against Archive metadata state. Project ingestion is an Archive consumer and never parses the live or official snapshot directly.
_Avoid_: Snapshot consumer, second parser

## Retention

**Retained archive**:
A ChatGPT Markdown file that remains durable after its conversation is no longer discovered or no longer belongs to an Approved Project. Its absence stops future updates but never authorizes automatic deletion.
_Avoid_: Stale file, orphaned export

**Explicit prune**:
A separately invoked, human-confirmed operation that removes Markdown only after an authoritative Official export snapshot proves the conversation is outside the Project allowlist.
_Avoid_: Garbage collection, automatic cleanup

## Conversation identity

**Current branch**:
The single user-visible message path selected by ChatGPT for a conversation. One Markdown file represents only this branch and is rewritten in place when the branch or an existing message changes.
_Avoid_: Complete branch history, conversation version

**Branch quarantine**:
The Project Ingest state for previously consumed messages that disappear from the Current branch. It blocks automatic derived-memory deletion and waits for explicit review.
_Avoid_: Branch cleanup, automatic rollback
