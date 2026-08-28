# Store only the current ChatGPT branch

Each ChatGPT conversation has one stable Markdown path that is rewritten to the current user-visible branch when messages are edited or the branch changes; edits do not create versioned Markdown copies. The exporter records message IDs, hashes, and a branch fingerprint, Project Ingest quarantines messages that leave the current branch instead of deleting derived memory, and untouched official export snapshots remain the source for any later historical audit.
