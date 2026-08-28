# Use live snapshots with official-export reconciliation

The ChatGPT adapter accepts transient live snapshots from the authenticated Codex App automation for daily updates and untouched official export snapshots for historical reconciliation. Both paths share one parser, Project allowlist, incremental state, and Markdown writer: live discovery provides freshness but is globally windowed, while official exports provide user-triggered completeness but cannot support daily synchronization on their own.
