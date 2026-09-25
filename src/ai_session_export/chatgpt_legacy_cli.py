from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .chatgpt_legacy import (
    ChatGPTLegacyImportError,
    apply_legacy_import,
    import_report,
    plan_legacy_import,
)
from .cli import BASE_DIR, STATE_FILE, _state_write_lock
from .state import assert_chatgpt_state_local, load_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-time conversion of legacy ChatGPT Markdown into verified archives."
    )
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument("--chatgpt-project-config", type=Path, required=True)
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    observed_at = datetime.now(ZoneInfo(args.timezone)).isoformat()
    if not args.apply:
        state = load_state(args.state_file, sources=("chatgpt",))
        assert_chatgpt_state_local(state)
        plan = plan_legacy_import(
            args.base_dir / "chatgpt",
            state,
            args.chatgpt_project_config,
            timezone_name=args.timezone,
            observed_at=observed_at,
        )
        return import_report(plan, applied=False)
    with _state_write_lock(args.state_file):
        state = load_state(args.state_file, sources=("chatgpt",))
        assert_chatgpt_state_local(state)
        plan = plan_legacy_import(
            args.base_dir / "chatgpt",
            state,
            args.chatgpt_project_config,
            timezone_name=args.timezone,
            observed_at=observed_at,
        )
        return apply_legacy_import(plan, args.state_file)


def main() -> None:
    args = parse_args()
    try:
        report = run(args)
    except (
        ChatGPTLegacyImportError,
        OSError,
        ValueError,
        ZoneInfoNotFoundError,
    ) as error:
        print(json.dumps({"source": "chatgpt", "error": str(error)}), file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
