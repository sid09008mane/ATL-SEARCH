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
| Load into Postgres + pgvector | `scripts/load_postgres.py` | done |
| Load into Qdrant | `scripts/load_qdrant.py` | done |
| Search engine (both databases, three modes) | `scripts/search.py` | done |
| Run all test questions on both | `scripts/compare.py` | done |
| Answer engine ("brain"): gpt-oss-20b on Groq | `scripts/answer.py` | done |
| Final recommendation | `report/comparison.md` | draft, waiting for real results |

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

### 4. Create the vectors

```bash
python scripts/embedder.py "tenant did not pay rent, can the landlord evict him"
```

- The first run downloads the BGE-M3 model (about 2.3 GB, once only) and embeds
  all 871 paragraphs. It ends with `Saved (871, 1024) to data/vectors/bge_m3.npy`
  and prints the five best paragraphs.
- Later runs print `Loaded 871 vectors` and skip the embedding.
- If the download sits at 0% for several minutes, stop it with `Ctrl + C` and
  run it again; it continues where it stopped.

### 5. Load both databases

```bash
python scripts/load_postgres.py
python scripts/load_qdrant.py
```

Both reuse the saved vectors, so the two databases hold identical data. Each
prints how long loading and indexing took; copy those numbers into `notes.md`.
Re-running either one rebuilds it from scratch.

### 6. Search

```bash
python scripts/search.py "tenant did not pay rent, can the landlord evict him"
```

This searches both databases and shows the five best paragraphs from each, with
the case name, year, paragraph number, score and a link to the judgment.

| What | Command |
|------|---------|
| Interactive (model loads once, ask many questions) | `python scripts/search.py` |
| One database only | `python scripts/search.py "your question" --db postgres` (or `--db qdrant`) |
| Choose how to search | `--mode semantic`, `--mode keyword` or `--mode hybrid` (default) |
| More results | `-k 10` |
| Only one kind of case | `--category property`, `tenant` or `property_and_tenant` |
| Every result must contain the section you named | `--strict` |
| Output for other programs | `--json` |

In interactive mode, change settings with `:db qdrant`, `:mode keyword`, `:k 10`,
`:category tenant`, `:strict on`, and quit with `:q`.

### 7. Compare the two databases

```bash
python scripts/compare.py
```

Runs all 12 questions in `eval/questions.json` on both databases in all three
modes, writes one row per run to `eval/results.csv`, and prints a summary:
how often the expected case was in the top 5, how often the exact section was
found, and the median search time. These numbers go into `report/comparison.md`.

### 8. Get an answer from the brain

The answer engine runs the search, gives the best paragraphs to **gpt-oss-20b**
(an open-weight model) through Groq's API, and prints a plain-language answer
with a citation after every point.

One-time setup:

1. Get a free API key at https://console.groq.com/keys (no credit card needed).
2. Copy the template and put your key in it:

```bash
cp .env.example .env
```

Then open `.env` and replace `gsk_your_key_here` with your key. `.env` is in
`.gitignore`, so the key never goes to GitHub.

Ask a question:

```bash
python scripts/answer.py "The tenant did not pay rent. Can the landlord evict him?"
```

| What | Command |
|------|---------|
| Interactive (ask many questions) | `python scripts/answer.py` |
| Use Qdrant instead of Postgres | `--db qdrant` |
| Give the model more paragraphs | `-k 8` |
| See every paragraph the model read | `--show-sources` |
| Only one kind of case | `--category tenant` |
| Output for other programs | `--json` |

**How the answer stays grounded** (ATL's rule: no claim without a matching precedent):

1. If the question names a section that no judgment contains (for example
   `Section 138`), it says so and does not call the model at all.
2. The model sees only the retrieved paragraphs, numbered `[1]`, `[2]`, ...,
   and is told to use nothing else and to cite after every point.
3. If those paragraphs do not answer the question, the model must reply
   `NO_MATCH`, and the engine prints "no answer is given" instead of guessing.
4. Every citation in the answer is checked. A citation to a source the model
   was not given, or an answer with no citations at all, is flagged as unverified.
5. It explains what courts held; it does not predict outcomes or give advice.

The model is set in `.env`: `ATL_LLM_MODEL=openai/gpt-oss-120b` gives stronger
answers, a little slower. Any OpenAI-compatible API works by changing
`ATL_LLM_BASE_URL` (for example Cerebras, OpenRouter, or a local Ollama server).
Your question and the retrieved paragraphs are sent to Groq; that is fine for
public judgments, but not for private client details.

## How the search works

| Mode | Postgres + pgvector | Qdrant |
|------|---------------------|--------|
| `semantic` (meaning) | nearest vectors through an HNSW index | nearest vectors through its HNSW index |
| `keyword` (words) | full-text search, ranked by `ts_rank_cd` | keeps paragraphs containing the words, orders them by meaning (Qdrant has no keyword scoring without extra sparse vectors) |
| `hybrid` (default) | both lists merged with Reciprocal Rank Fusion | same |

**Exact legal references.** References in a question such as `Section 12(1)(a)`,
`Section 38-E`, `Article 65`, `Order XXI` or `s. 70(b)`, and anything in double
quotes, are picked up automatically.

- Keyword search only returns paragraphs that contain them. Postgres matches
  with a regular expression over a trigram index; Qdrant matches the phrase
  through its full-text index.
- Each result says which references it contains.
- If no paragraph in the collection contains a reference, the search says so
  (for example, none of the 20 judgments mentions `Section 138`).
- `--strict` applies the same rule to meaning search too.

Timings shown by `search.py` separate the database search from the time taken
to embed the question, so the two databases are compared fairly.

## Everyday commands

| What | Command |
|------|---------|
| Start the databases | `docker compose up -d` |
| Stop them (data is kept) | `docker compose stop` |
| See their logs | `docker compose logs -f` |
| Remove them and their data | `docker compose down -v` |
| Rebuild the vectors | delete `data/vectors/bge_m3.npy`, then run the embedder |
| Get the latest code | `git pull` |
| Save and share your work | `git add .` then `git commit -m "what changed"` then `git push` |

All settings (file paths, models, database addresses, API key) live in
`scripts/config.py`, which also reads your `.env` file. The scripts work from any folder. To point at a different
database, set `ATL_PG_DSN` or `ATL_QDRANT_URL`. To run Qdrant without Docker,
set `ATL_QDRANT_PATH` to a folder and Qdrant runs inside Python instead
(handy for quick tests, but not for timing comparisons).

## Folder layout

```
ATL-SEARCH/
├── README.md
├── requirements.txt          Python libraries
├── docker-compose.yml        Postgres + pgvector and Qdrant
├── .env.example              template for your API key (copy to .env)
├── notes.md                  running log of observations
├── data/
│   ├── raw/                  original judgment text, never edited
│   ├── clean/                cleaned text (atl_chunks.json is the file to load)
│   └── vectors/              embeddings, created by the embedder (not in Git)
├── scripts/
│   ├── config.py             shared settings and database connections
│   ├── build_clean_json.py   raw -> clean paragraphs
│   ├── embedder.py           paragraphs -> vectors, question -> vector
│   ├── embedder_reference.py a longer version with per-paragraph caching (not used by the pipeline)
│   ├── load_postgres.py      vectors -> Postgres + pgvector
│   ├── load_qdrant.py        vectors -> Qdrant
│   ├── search.py             the search engine: question -> best paragraphs
│   ├── answer.py             the brain: question -> cited answer from gpt-oss-20b
│   └── compare.py            runs eval/questions.json on both databases
├── eval/
│   ├── questions.json        12 test questions with expected cases
│   └── results.csv           written by compare.py
└── report/
    └── comparison.md         final recommendation for the team
```

How the scripts connect:

```
config.py ──► embedder.py ──► load_postgres.py ──► Postgres + pgvector ─┐
                    │    └──► load_qdrant.py   ──► Qdrant ──────────────┤
                    └───────────────────────────────────────────────────┴──► search.py ──► compare.py
                                                                                 └──► answer.py ──► gpt-oss-20b (Groq)
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

## Using the search engine from other code

```python
from search import SearchEngine

engine = SearchEngine()                      # connects to both databases
response = engine.search("my cheque bounced, what can I do", database="postgres", mode="hybrid", k=5)
for hit in response.hits:
    print(hit.case_name, hit.para_no, hit.score, hit.source_url)
print(response.notes)                        # e.g. a reference no paragraph contains
engine.close()
```

Run it from the `scripts` folder, or add `scripts` to the Python path.
