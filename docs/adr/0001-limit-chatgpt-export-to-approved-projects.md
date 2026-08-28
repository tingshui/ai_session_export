# Limit ChatGPT export to approved Projects

ChatGPT session export is intentionally Project-scoped: only conversations whose exact `g-p-...` Project ID appears in the user-approved allowlist become Markdown, and the same boundary governs downstream Project ingestion. This gives up a complete account-wide archive so Projectless and unapproved conversations never cross into the durable local archive by accidental discovery.
