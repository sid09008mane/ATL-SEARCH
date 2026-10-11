# ATL-SEARCH

Storing and searching court judgments for **Any Time Law (ATL)**, an MIT-WPU
seed grant project. This repo is the "storing and searching" experiment
(Divyani and Siddharth):

1. Clean 20 Supreme Court judgments and cut them into paragraphs, so a search
   returns the exact paragraph and not a whole judgment.
2. Compare two ways of storing and searching those paragraphs:
   **Qdrant** and **Postgres with pgvector**. For each: ease of setup, speed,
   and whether it can match exact words such as a section number.
3. Build a very small search: type a question, get the five best paragraphs
   with the case name beside each one.

## Status

| Step | File | State |
|------|------|-------|
| Clean and cut the judgments | `scripts/build_clean_json.py` | done |
| Turn paragraphs into vectors | `scripts/embedder.py` | done |
| Load into Postgres + pgvector | `scripts/load_postgres.py` | to write |
| Load into Qdrant | `scripts/load_qdrant.py` | to write |
| Search from either database | `scripts/search.py` | to write |
| Run all test questions on both | `scripts/compare.py` | to write |
| Final recommendation | `report/comparison.md` | draft |

## Quick start

You need **Python 3.10+**, **Git** and **Docker Desktop** (open and running).

### 1. Get the code

```bash
git clone https://github.com/sid09008mane/ATL-SEARCH.git
cd ATL-SEARCH
```

### 2. Install the Python libraries

```bash
python -m pip install -r requirements.txt
```

Use `python -m pip`, not plain `pip`, so the libraries go into the same Python
that runs the scripts. Otherwise you may get
`ModuleNotFoundError: No module named 'sentence_transformers'`.

### 3. Start the two databases

```bash
docker compose up -d
docker compose ps
```

Both `atl-pg` and `atl-qdrant` should show "Up".

| Service  | Address |
|----------|---------|
| Postgres | `postgresql://postgres:atl@localhost:5432/atl` |
| Qdrant   | `http://localhost:6333` (dashboard at `http://localhost:6333/dashboard`) |

### 4. Create the vectors and run a test search

```bash
python scripts/embedder.py "tenant did not pay rent, can the landlord evict him"
```

- The first run downloads the BGE-M3 model (about 2.3 GB, once only) and embeds
  all 871 paragraphs. It ends with `Saved (871, 1024) to data/vectors/bge_m3.npy`
  and prints the five best paragraphs.
- Later runs print `Loaded 871 vectors` and skip the embedding.
- If the download sits at 0% for several minutes, stop it with `Ctrl + C` and
  run it again; it continues where it stopped.

### Everyday commands

| What | Command |
|------|---------|
| Start the databases | `docker compose up -d` |
| Stop them (data is kept) | `docker compose stop` |
| See their logs | `docker compose logs -f` |
| Remove them and their data | `docker compose down -v` |
| Ask a different question | `python scripts/embedder.py "your question"` |
| Rebuild the vectors | delete `data/vectors/bge_m3.npy`, then run the embedder |
| Get the latest code | `git pull` |
| Save and share your work | `git add .` then `git commit -m "what changed"` then `git push` |

Always run commands from the project folder, because paths in the scripts are
relative to it.

## Folder layout

```
ATL-SEARCH/
├── README.md
├── requirements.txt          Python libraries
├── docker-compose.yml        Postgres + pgvector and Qdrant
├── notes.md                  running log of observations
├── data/
│   ├── raw/                  original judgment text, never edited
│   ├── clean/                cleaned text (atl_chunks.json is the file to load)
│   └── vectors/              embeddings, created by the embedder (not in Git)
├── scripts/
│   ├── build_clean_json.py   raw -> clean paragraphs
│   ├── embedder.py           paragraphs -> vectors, plus a test search
│   ├── embedder_reference.py a longer version with per-paragraph caching
│   ├── load_postgres.py      vectors -> Postgres + pgvector
│   ├── load_qdrant.py        vectors -> Qdrant
│   ├── search.py             question -> five best paragraphs
│   └── compare.py            runs eval/questions.json on both databases
├── eval/
│   ├── questions.json        12 test questions with expected cases
│   └── results.csv           one row per question per database
└── report/
    └── comparison.md         final recommendation for the team
```

## The data

| File | What it is |
|------|------------|
| `data/clean/atl_chunks.json` | 871 search-ready paragraphs, one row each: `id`, `chunk_id`, `case_id`, `case_name`, `court`, `year`, `category`, `para_no`, `part`, `parts_in_para`, `paragraph`, `word_count`, `source_url` |
| `data/clean/atl_judgments_clean.json` | The same text grouped by case (20 cases, 644 paragraphs) with bench, date, citations, headnote and footnotes |
| `data/raw/atl_raw_judgments.json` | Original text as downloaded |

- 20 Supreme Court of India judgments, 1965 to 2021: 7 property, 8 tenant, 5 both. No banking cases yet.
- Text was taken from Indian Kanoon, not from an official court site. Check any
  paragraph against the official report before quoting it in a paper, and check
  Indian Kanoon's terms of use before scaling beyond this experiment.
- 10 judgments use the court's own paragraph numbers; the other 10 have no
  numbering in the original, so theirs are generated in order
  (`para_numbering` in the per-case file says which).
- Paragraphs longer than about 320 words are split at sentence ends into parts
  of roughly 200 words, with one sentence of overlap.

## Embedding model

`BAAI/bge-m3` (1024-dimension vectors, multilingual). The same model must be
used for paragraphs and questions, and the same vectors must go into both
databases so the comparison is fair.

## Using the embedder from other scripts

```python
from embedder import load_chunks, embed_documents, embed_query

chunks = load_chunks()
vectors = embed_documents([c["paragraph"] for c in chunks])
q_vector = embed_query("my cheque bounced, what can I do")
```

`vectors[i]` always belongs to `chunks[i]`.
