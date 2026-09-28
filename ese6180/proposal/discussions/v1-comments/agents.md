# Agent brief - Overleaf comments for active-eval-gym

You are reading an Overleaf comment export produced by the
overleaf-comments-export browser extension. Two files in this folder are
relevant:

- `comments-2026-09-22.md` - human-readable Markdown, with YAML front-matter and
  comments grouped by file, then section, then line. Every comment has a
  stable short ID like `C001`, assigned in file then line order. The
  Markdown is the canonical user-facing view.
- `comments.json` - the same data in structured form. Use this when you need
  to enumerate, filter, or programmatically address comments.

## JSON schema (key parts)

- `schema_version` (string)
- `project` - `{ id, title }`
- `summary` - counts (threads, open/resolved, tracked changes, stale, files,
  reviewers)
- `threads` - `{ "<thread_id>": { id, resolved, resolved_at,
  resolved_by_user_id, messages: [...] } }` - stored once at top level, not
  duplicated inside each comment.
- `files` - list of `{ pathname, doc_id, comment_count, change_count,
  comment_short_ids, change_short_ids }`
- `comments` - list of `{ short_id, thread_id, doc_id, pathname, line, col,
  offset, nearest_heading, anchored_text, stale, context }`. To get the
  discussion, look up `threads[thread_id]`.
- `tracked_changes` - list of `{ short_id, id, doc_id, pathname, kind
  (insertion|deletion), content, line, col, offset, nearest_heading, user,
  timestamp, context }`
- `orphan_thread_ids` - IDs of threads that do not anchor to live source.

`context` is a compact char-window snippet: `before`, `anchor`, `after`,
with `truncated_before`/`truncated_after` flags.

## How to address comments

- Refer to comments by `short_id`, for example "C014", not by `thread_id`.
- Each thread's FIRST message is the reviewer's actual ask. Later messages are
  follow-up discussion, often between co-authors. Answer the ask, taking the
  discussion into account; do not re-litigate sub-points the replies already
  settled.
- For each open comment, propose an edit to the .tex source. If the comment is
  a question, answer it; if it is a request, attempt the change.
- Stale comments (`stale: true`) may not point to the current location in the
  doc. Use `anchored_text` and `nearest_heading` to find the right spot.
- Tracked changes (`T001`-prefixed) are not comment threads; they are
  insertions and deletions someone made with Track Changes enabled. Treat them
  as suggested edits to accept, reject, or modify.

## What you do NOT have

- The full `.tex` source of the paper. You only see a short window around each
  anchor. If you need more context, ask the user to share the relevant `.tex`
  file.
- The ability to push edits back to Overleaf. Output any proposed edits as
  diffs or rewrites; the user will apply them.

Project ID for reference: `6ab0a43b833a3baaf06e2382`.
