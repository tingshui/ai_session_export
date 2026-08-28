# Make the ChatGPT archive the ingestion boundary

ChatGPT Project Ingest consumes only Markdown plus the exporter’s raw-text-free per-conversation metadata state; it no longer accepts live or official snapshots directly. The authenticated automation reads ChatGPT once and `ai_session_export` owns the only raw parser and Markdown writer, while Project Ingest verifies archived turns by message ID and hash before treating user turns as reflection evidence.
