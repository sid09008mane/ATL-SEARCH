# Storing and searching: Qdrant vs Postgres + pgvector

Divyani and Siddharth, ATL project. Status: **draft, results not filled in yet.**

## 1. Recommendation

_One paragraph: which option ATL should use, and the main reason._

## 2. What we tested

- **Data:** 20 Supreme Court judgments on property and tenancy (1965 to 2021),
  cut into 871 paragraphs.
- **Embedding model:** BAAI/bge-m3, 1024-dimension vectors. Identical vectors
  were loaded into both databases.
- **Questions:** 12 test questions in `eval/questions.json`: 6 in plain
  language, 3 exact terms, 1 mixed, and 2 that have no answer in the corpus.
- **Machine:** MacBook Air (Apple Silicon), both databases in Docker.

## 3. Cleaning and cutting the text

### What the source file looked like
The file we received had 20 records with these problems:

| Problem | Effect |
|---------|--------|
| 6 judgments cut off at 32,765 characters | final reasoning and order missing |
| No case name field | results could not show a case name |
| Text split across two fields | had to be joined, some splits were mid-word |
| All lowercase, brackets removed | "Section 12(1)(a)" had become "section 12 1  a" |
| No paragraph marking | could not split on blank lines |
| Headnotes mixed in | search could return an editor's summary |

### What we did
- Took the original text of each judgment again and rebuilt the file.
- Removed page numbers, court headers, signatures and footnote lists from the searchable text.
- Kept headnotes in a separate field.
- Used the court's own paragraph numbers where they exist (10 of 20 judgments);
  generated numbers in order for the rest.
- Split paragraphs longer than about 320 words at sentence ends, about 200
  words per part, one sentence of overlap.

### What did not work, or is still weak
- _to fill in_
- Old judgments (1965 to 1979) still contain scanning typos from the source.

## 4. Comparison

| | Postgres + pgvector | Qdrant |
|---|---|---|
| Ease of setup | | |
| Time to first working search | | |
| How fast it feels (871 rows) | | |
| Average search time (ms) | | |
| Exact-word match (e.g. "Section 12(1)(a)") | | |
| Meaning + exact word in one query | | |
| Extra systems to run | none (ATL needs Postgres anyway) | one more service |

With only 871 rows both will feel instant, so speed here says little about
behaviour at lakhs of paragraphs.

## 5. Results by question

| Question | Type | Postgres: expected case in top 5? | Qdrant: expected case in top 5? |
|----------|------|-----------------------------------|---------------------------------|
| q01 | plain | | |
| q02 | plain | | |
| q03 | plain | | |
| q04 | plain | | |
| q05 | plain | | |
| q06 | plain | | |
| q07 | exact | | |
| q08 | exact | | |
| q09 | exact | | |
| q10 | mixed | | |
| q11 | no answer | | |
| q12 | no answer | | |

For q11 and q12 the right behaviour is to return nothing relevant. Note the
top score each database gives, since ATL must be able to say "no matching
case found".

## 6. Open problems

- No banking judgments in the test set.
- Text source is Indian Kanoon; licensing for a larger corpus is not settled.
- Scanned PDFs (needing OCR) were not tested.
- _to fill in_

## 7. Next steps

- _to fill in_
