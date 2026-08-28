from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .chatgpt_incremental import (
    ChatGPTIncrementalError,
    activate_live_authority,
    apply_incremental_payload,
    discovery_saturation,
    discovery_report,
    load_incremental_scope_allowlist,
    plan_historical_backfill,
    plan_incremental_threads,
    plan_window_threads,
)
from .cli import BASE_DIR, STATE_FILE, _read_sensitive_stdin, _state_write_lock
from .state import load_state, save_state


def _payload() -> dict[str, Any]:
    raw = _read_sensitive_stdin()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ChatGPTIncrementalError(f"invalid incremental JSON: {error}") from error
    if not isinstance(value, dict):
        raise ChatGPTIncrementalError("incremental input must be an object")
    return value


def _summaries(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema_version") != 2:
        raise ChatGPTIncrementalError("unsupported incremental payload schema")
    summaries = value.get("summaries")
    if not isinstance(summaries, list) or not all(isinstance(item, dict) for item in summaries):
        raise ChatGPTIncrementalError("incremental summaries must be an object array")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or apply a metadata-gated ChatGPT incremental archive read."
    )
    parser.add_argument(
        "action",
        choices=(
            "plan",
            "apply",
            "backfill-plan",
            "backfill-apply",
            "window-plan",
            "window-apply",
            "activate-live",
        ),
    )
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--chatgpt-project-config", type=Path, required=True)
    parser.add_argument("--at", help="Timezone-aware activation timestamp for activate-live.")
    parser.add_argument("--since-at", help="Timezone-aware lower bound for window actions.")
    parser.add_argument(
        "--until-at",
        help="Timezone-aware exclusive upper bound for window actions.",
    )
    return parser.parse_args()


def _since_seconds(value: str | None) -> float:
    if not value:
        raise ChatGPTIncrementalError("window action requires --since-at")
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ChatGPTIncrementalError("window --since-at is invalid") from error
    if instant.tzinfo is None:
        raise ChatGPTIncrementalError("window --since-at must be timezone-aware")
    return instant.timestamp()


def _until_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ChatGPTIncrementalError("window --until-at is invalid") from error
    if instant.tzinfo is None:
        raise ChatGPTIncrementalError("window --until-at must be timezone-aware")
    return instant.timestamp()


def run(
    args: argparse.Namespace, payload: dict[str, Any] | None
) -> tuple[int, dict[str, Any]]:
    approved = load_incremental_scope_allowlist(args.chatgpt_project_config)
    if args.action in {"plan", "backfill-plan", "window-plan"}:
        if payload is None:
            raise ChatGPTIncrementalError("plan requires an input payload")
        state = load_state(args.state_file)
        if args.action == "window-plan":
            plan = plan_window_threads(
                _summaries(payload),
                approved,
                since_at=_since_seconds(args.since_at),
                until_at=_until_seconds(args.until_at),
            )
        else:
            planner = (
                plan_historical_backfill
                if args.action == "backfill-plan"
                else plan_incremental_threads
            )
            plan = planner(_summaries(payload), state["chatgpt"], approved)
        report = discovery_report(plan)
        report["discovery_saturated"] = discovery_saturation(payload)
        return 0, report

    with _state_write_lock(args.state_file):
        state = load_state(args.state_file)
        source_state = state["chatgpt"]
        if args.action == "activate-live":
            if not args.at:
                raise ChatGPTIncrementalError("activate-live requires --at")
            activate_live_authority(source_state, args.at)
            save_state(state, args.state_file)
            return 0, {
                "source": "chatgpt",
                "authority": "live_only",
                "live_authority_started_at": args.at,
                "bootstrap_complete": False,
            }
        if payload is None:
            raise ChatGPTIncrementalError("apply requires an input payload")

        def checkpoint(staged_source: dict[str, Any]) -> None:
            staged_state = deepcopy(state)
            staged_state["chatgpt"] = staged_source
            save_state(staged_state, args.state_file)

        report = apply_incremental_payload(
            args.base_dir / "chatgpt",
            source_state,
            approved,
            payload,
            historical_backfill=args.action == "backfill-apply",
            window_since_at=(
                _since_seconds(args.since_at)
                if args.action == "window-apply"
                else None
            ),
            window_until_at=(
                _until_seconds(args.until_at)
                if args.action == "window-apply"
                else None
            ),
            checkpoint_state=checkpoint,
        )
        return (1 if report["failed"] else 0), report


def main() -> None:
    args = parse_args()
    try:
        exit_code, report = run(
            args, None if args.action == "activate-live" else _payload()
        )
    except (ChatGPTIncrementalError, OSError, ValueError) as error:
        print(json.dumps({"source": "chatgpt", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(exit_code)
