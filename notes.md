# Working notes

Rough log, written as things happen. This feeds `report/comparison.md`.

## Setup log

### 10 Oct 2026
- Mac (Apple Silicon), Python 3.13 from miniforge.
- `pip install sentence-transformers "psycopg[binary]" pgvector qdrant-client pymupdf` worked first time.
- Postgres + pgvector: one `docker run` command, image pulled and container started with no errors.
- Qdrant: one `docker run` command, image pulled and container started with no errors.
- Pasting commands together with `#` comment lines gave `zsh: command not found: #`. Harmless.
- Setup effort so far: a tie, one command each.

### 11 Oct 2026
- Moved both databases into `docker-compose.yml` (named volumes, database `atl`).
- Got the source file `propertyAndTenantCases.json` (20 records). Problems found:
  - 6 judgments cut off at exactly 32,765 characters (looks like a spreadsheet cell limit)
  - no case name field, only a file id
  - text split across two fields, sometimes mid-word
  - everything lowercased and brackets removed ("section 12 1  a")
  - no paragraph marking; headnotes mixed into the text
  - one record with unreadable junk in a field
- Rebuilt from the original text (Indian Kanoon). Result: 20 cases, 644 paragraphs, 871 search rows.

## Time taken

| Step | Postgres + pgvector | Qdrant |
|------|---------------------|--------|
| Install and start | | |
| Create table / collection | | |
| Load 871 rows | | |
| First working meaning search | | |
| First working exact-word search | | |

## Things that went wrong

| Date | Database | What happened | How it was fixed |
|------|----------|---------------|------------------|
| | | | |

## Speed observations

- Embedding 871 paragraphs with BGE-M3 on this Mac took:
- Typical search time, Postgres:
- Typical search time, Qdrant:

## Exact-word matching observations

- Postgres:
- Qdrant:

## Open questions for the team

- No banking cases in this set. Where do those come from?
- Is Indian Kanoon an acceptable source, or does the team have a licensed one?
