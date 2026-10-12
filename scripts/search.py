"""ATL search engine: type a question, get the best paragraphs with the case name beside each.

    python scripts/search.py "tenant did not pay rent, can the landlord evict him"
    python scripts/search.py "Section 12(1)(a)" --mode keyword --db postgres
    python scripts/search.py "adverse possession" --category property -k 10
    python scripts/search.py                      # interactive: the model loads once, ask many questions

Three ways to search, on either database:
  semantic  by meaning: the question's vector against every paragraph's vector
  keyword   by words: Postgres ranks paragraphs with full-text search; Qdrant has no keyword
            scoring, so it keeps only paragraphs containing the words and orders them by meaning
  hybrid    both lists merged with Reciprocal Rank Fusion (the default)

Exact legal references in the question, such as "Section 12(1)(a)", "Section 38-E" or
"Article 65", and anything in double quotes, are detected automatically. Keyword search
only returns paragraphs that contain them; with --strict every mode does.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field

from config import PG_TABLE, QDRANT_COLLECTION, connect_postgres, connect_qdrant

DATABASES = ("postgres", "qdrant")
MODES = ("semantic", "keyword", "hybrid")
CATEGORIES = ("property", "tenant", "property_and_tenant")
DB_LABELS = {"postgres": "Postgres + pgvector", "qdrant": "Qdrant"}

CANDIDATES = 50  # results each method contributes before hybrid fusion
RRF_K = 60       # standard Reciprocal Rank Fusion constant

STOPWORDS = set(
    "a an and are as at be been but by can could did do does for from had has have he her his how "
    "i if in into is it its me my no not of on or our she should so than that the their them then "
    "there these they this to under upon was we were what when where which who whom why will with "
    "would you your".split()
)


# ---------------------------------------------------------------- exact terms

@dataclass(frozen=True)
class ExactTerm:
    """A phrase that must appear verbatim, with matchers for Python, Postgres and Qdrant."""
    text: str        # canonical form, also used as the Qdrant phrase
    python: re.Pattern
    postgres: str    # POSIX regex for Postgres' case-insensitive ~* operator


REFERENCE_KINDS = {
    "section": r"(?:sections?|sec\.?|s\.)",
    "article": r"(?:articles?|art\.?)",
    "order": r"orders?",
    "rule": r"rules?",
    "regulation": r"regulations?",
    "clause": r"clauses?",
    "schedule": r"schedules?",
}
KIND_ALIASES = {"sections": "section", "sec": "section", "sec.": "section", "s.": "section",
                "articles": "article", "art": "article", "art.": "article"}
NUMBER = r"(?:\d+[A-Za-z]{0,2}|[IVXLC]+\b)"
LEGAL_REFERENCE = re.compile(
    r"(?<![A-Za-z])(sections?|sec\.|s\.|articles?|art\.|orders?|rules?|regulations?|clauses?|schedules?)"
    rf"\s*({NUMBER}(?:\s*[-–]\s*[A-Za-z]{{1,2}}\b)?(?:\s*\(\s*[0-9A-Za-z]{{1,4}}\s*\))*)",
    re.IGNORECASE,
)
QUOTED = re.compile(r'"([^"]{2,})"|“([^”]{2,})”')


def _literal_pattern(text: str) -> str:
    """Regex for a phrase, tolerant of spacing around brackets and hyphens. Valid in Python and Postgres."""
    out = []
    for char in text.strip():
        if char.isalnum():
            out.append(char)
        elif char.isspace():
            if out and out[-1] != r"\s+":
                out.append(r"\s+")
        elif char == "(":
            out.append(r"\s*\(\s*")
        elif char == ")":
            out.append(r"\s*\)")
        elif char in "-–":
            out.append(r"\s*[-–]\s*")
        else:
            out.append("\\" + char)
    return "".join(out)


def make_term(phrase: str, kind: str | None = None, number: str | None = None) -> ExactTerm:
    if kind:
        canonical_kind = KIND_ALIASES.get(kind.lower()) or kind.lower().rstrip("s")
        number = re.sub(r"\s+", "", number)
        text = f"{canonical_kind.capitalize()} {number}"
        body = REFERENCE_KINDS[canonical_kind] + r"\s*" + _literal_pattern(number)
    else:
        text = re.sub(r"\s+", " ", phrase.strip())
        body = _literal_pattern(text)
    pattern = r"(?<![0-9A-Za-z])" + body + r"(?![0-9A-Za-z])"
    return ExactTerm(text=text, python=re.compile(pattern, re.IGNORECASE), postgres=pattern)


def extract_exact_terms(query: str) -> list[ExactTerm]:
    """Find legal references and quoted phrases in a question."""
    terms: dict[str, ExactTerm] = {}
    for match in QUOTED.finditer(query):
        term = make_term(match.group(1) or match.group(2))
        terms.setdefault(term.text.lower(), term)
    unquoted = QUOTED.sub(" ", query)
    for match in LEGAL_REFERENCE.finditer(unquoted):
        term = make_term(match.group(0), kind=match.group(1), number=match.group(2))
        terms.setdefault(term.text.lower(), term)
    return list(terms.values())


def content_words(query: str) -> list[str]:
    words = re.findall(r"[0-9a-z]+", query.lower())
    return list(dict.fromkeys(w for w in words if w not in STOPWORDS and (len(w) > 2 or w.isdigit())))


# ---------------------------------------------------------------- results

@dataclass
class Hit:
    id: int
    case_name: str
    para_no: int | None
    part: int | None
    parts_in_para: int | None
    year: int | None
    category: str | None
    paragraph: str
    source_url: str | None
    score: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class SearchResponse:
    query: str
    database: str
    mode: str
    hits: list[Hit]
    exact_terms: list[str]
    embed_ms: float
    search_ms: float
    score_kind: str
    notes: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)  # exact terms no paragraph in the collection contains


class BackendUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- Postgres + pgvector

PG_COLUMNS = "id, case_name, para_no, part, parts_in_para, year, category, paragraph, source_url"


class PostgresBackend:
    name = "postgres"
    keyword_needs_vector = False

    def __init__(self):
        try:
            self.conn = connect_postgres()
        except Exception as exc:
            raise BackendUnavailable(
                f"Cannot connect to Postgres ({exc.__class__.__name__}). Is it running? Try: docker compose up -d"
            ) from exc
        if self.conn.execute("SELECT to_regclass(%s)", [PG_TABLE]).fetchone()[0] is None:
            raise BackendUnavailable(f"Postgres has no '{PG_TABLE}' table yet. Run: python scripts/load_postgres.py")
        for setting in ("SET hnsw.ef_search = 100", "SET hnsw.iterative_scan = relaxed_order"):
            try:
                self.conn.execute(setting)
            except Exception:
                pass  # older pgvector versions do not have every setting

    @staticmethod
    def _filters(terms, category):
        clauses, params = [], []
        for term in terms:
            clauses.append("paragraph ~* %s")
            params.append(term.postgres)
        if category:
            clauses.append("category = %s")
            params.append(category)
        return clauses, params

    @staticmethod
    def _hits(rows):
        return [Hit(id=r[0], case_name=r[1], para_no=r[2], part=r[3], parts_in_para=r[4], year=r[5],
                    category=r[6], paragraph=r[7], source_url=r[8], score=float(r[9])) for r in rows]

    def semantic(self, query_vector, k, terms=(), category=None):
        clauses, params = self._filters(terms, category)
        if clauses:
            # Filter first, then rank the survivors exactly, so a filter can never hide a match.
            sql = (f"WITH candidates AS MATERIALIZED (SELECT * FROM {PG_TABLE} WHERE {' AND '.join(clauses)}) "
                   f"SELECT {PG_COLUMNS}, 1 - (embedding <=> %s) AS score FROM candidates "
                   f"ORDER BY embedding <=> %s LIMIT %s")
            rows = self.conn.execute(sql, [*params, query_vector, query_vector, k]).fetchall()
        else:
            sql = (f"SELECT {PG_COLUMNS}, 1 - (embedding <=> %s) AS score FROM {PG_TABLE} "
                   f"ORDER BY embedding <=> %s LIMIT %s")
            rows = self.conn.execute(sql, [query_vector, query_vector, k]).fetchall()
        return self._hits(rows)

    def keyword(self, query, k, terms=(), category=None, query_vector=None):
        lexemes = self.conn.execute(
            "SELECT tsvector_to_array(to_tsvector('english', %s))", [query]
        ).fetchone()[0]
        words = sorted({re.sub(r"[^0-9a-z]", "", lexeme) for lexeme in lexemes} - {""})
        clauses, params = self._filters(terms, category)
        if words:
            where = " AND ".join(["tsv @@ q", *clauses])
            sql = (f"SELECT {PG_COLUMNS}, ts_rank_cd(tsv, q, 1) AS score "
                   f"FROM {PG_TABLE}, to_tsquery('simple', %s) AS q WHERE {where} "
                   f"ORDER BY score DESC, id LIMIT %s")
            rows = self.conn.execute(sql, [" | ".join(words), *params, k]).fetchall()
        elif clauses:
            sql = f"SELECT {PG_COLUMNS}, 1.0 AS score FROM {PG_TABLE} WHERE {' AND '.join(clauses)} ORDER BY id LIMIT %s"
            rows = self.conn.execute(sql, [*params, k]).fetchall()
        else:
            return []
        return self._hits(rows)

    def count_containing(self, term):
        return self.conn.execute(f"SELECT count(*) FROM {PG_TABLE} WHERE paragraph ~* %s", [term.postgres]).fetchone()[0]

    def close(self):
        self.conn.close()


# ---------------------------------------------------------------- Qdrant

class QdrantBackend:
    name = "qdrant"
    keyword_needs_vector = True  # Qdrant filters on words but needs the vector to order the results

    def __init__(self):
        from qdrant_client import models

        self.models = models
        try:
            self.client = connect_qdrant()
            exists = self.client.collection_exists(QDRANT_COLLECTION)
        except Exception as exc:
            raise BackendUnavailable(
                f"Cannot connect to Qdrant ({exc.__class__.__name__}). Is it running? Try: docker compose up -d"
            ) from exc
        if not exists:
            raise BackendUnavailable(f"Qdrant has no '{QDRANT_COLLECTION}' collection yet. Run: python scripts/load_qdrant.py")

    def _filter(self, terms=(), category=None, words=()):
        m = self.models
        must = [m.FieldCondition(key="paragraph", match=m.MatchPhrase(phrase=t.text)) for t in terms]
        if category:
            must.append(m.FieldCondition(key="category", match=m.MatchValue(value=category)))
        should = [m.FieldCondition(key="paragraph", match=m.MatchText(text=w)) for w in words]
        if not must and not should:
            return None
        return m.Filter(must=must or None, should=should or None)

    @staticmethod
    def _hit(point):
        p = point.payload
        return Hit(id=int(point.id), case_name=p["case_name"], para_no=p.get("para_no"), part=p.get("part"),
                   parts_in_para=p.get("parts_in_para"), year=p.get("year"),
                   category=p.get("category"), paragraph=p["paragraph"], source_url=p.get("source_url"),
                   score=float(getattr(point, "score", 0.0) or 0.0))

    def semantic(self, query_vector, k, terms=(), category=None):
        result = self.client.query_points(
            QDRANT_COLLECTION, query=query_vector.tolist(), query_filter=self._filter(terms, category),
            limit=k, with_payload=True,
        )
        return [self._hit(p) for p in result.points]

    def keyword(self, query, k, terms=(), category=None, query_vector=None):
        words = content_words(query)
        if not words and not terms:
            return []
        flt = self._filter(terms, category, words)
        if query_vector is None:
            points, _ = self.client.scroll(QDRANT_COLLECTION, scroll_filter=flt, limit=k, with_payload=True)
            return [self._hit(p) for p in points]
        result = self.client.query_points(
            QDRANT_COLLECTION, query=query_vector.tolist(), query_filter=flt, limit=k, with_payload=True,
        )
        return [self._hit(p) for p in result.points]

    def count_containing(self, term):
        return self.client.count(QDRANT_COLLECTION, count_filter=self._filter([term]), exact=True).count

    def close(self):
        self.client.close()


BACKENDS = {"postgres": PostgresBackend, "qdrant": QdrantBackend}


# ---------------------------------------------------------------- engine

def reciprocal_rank_fusion(*ranked_lists: list[Hit]) -> list[Hit]:
    fused: dict[int, Hit] = {}
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, 1):
            fused.setdefault(hit.id, hit)
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda i: (-scores[i], i))
    out = []
    for i in ordered:
        hit = fused[i]
        out.append(Hit(**{**asdict(hit), "score": scores[i]}))
    return out


class SearchEngine:
    """One entry point for every database and mode. Loads the embedding model only when needed."""

    def __init__(self, databases=DATABASES):
        self.backends = {}
        self.unavailable = {}
        for name in databases:
            try:
                self.backends[name] = BACKENDS[name]()
            except BackendUnavailable as exc:
                self.unavailable[name] = str(exc)
        self._last_vector = (None, None)

    def embed(self, query: str):
        """Return (vector, milliseconds). The same question is embedded only once."""
        if self._last_vector[0] == query:
            return self._last_vector[1], 0.0
        from embedder import embed_query

        started = time.perf_counter()
        vector = embed_query(query)
        elapsed = (time.perf_counter() - started) * 1000
        self._last_vector = (query, vector)
        return vector, elapsed

    def search(self, query, database, mode="hybrid", k=5, category=None, strict=False, query_vector=None):
        if database not in self.backends:
            raise BackendUnavailable(self.unavailable.get(database, f"Unknown database '{database}'"))
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        backend = self.backends[database]
        terms = extract_exact_terms(query)
        filter_terms = terms if strict else ()

        needs_vector = mode != "keyword" or backend.keyword_needs_vector
        embed_ms = 0.0
        if needs_vector and query_vector is None:
            query_vector, embed_ms = self.embed(query)

        started = time.perf_counter()
        if mode == "semantic":
            hits = backend.semantic(query_vector, k, filter_terms, category)
            score_kind = "cosine similarity"
        elif mode == "keyword":
            hits = backend.keyword(query, k, terms, category, query_vector)
            score_kind = "full-text rank" if database == "postgres" else "cosine similarity (after word filter)"
        else:
            semantic_hits = backend.semantic(query_vector, CANDIDATES, filter_terms, category)
            keyword_hits = backend.keyword(query, CANDIDATES, terms, category, query_vector)
            hits = reciprocal_rank_fusion(semantic_hits, keyword_hits)[:k]
            score_kind = "fusion score"
        search_ms = (time.perf_counter() - started) * 1000

        for hit in hits:
            hit.matched_terms = [t.text for t in terms if t.python.search(hit.paragraph)]

        notes, missing = [], []
        for term in terms:
            if any(term.text in hit.matched_terms for hit in hits):
                continue
            if backend.count_containing(term) == 0:
                missing.append(term.text)
                notes.append(f'No paragraph in the collection contains "{term.text}".')
            else:
                notes.append(f'None of these results contains "{term.text}". Try --mode keyword or --strict.')
        return SearchResponse(query=query, database=database, mode=mode, hits=hits,
                              exact_terms=[t.text for t in terms], embed_ms=embed_ms, search_ms=search_ms,
                              score_kind=score_kind, notes=notes, missing_terms=missing)

    def close(self):
        for backend in self.backends.values():
            backend.close()


# ---------------------------------------------------------------- display

USE_COLOUR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
BOLD, DIM, YELLOW, RESET = ("\033[1m", "\033[2m", "\033[33m", "\033[0m") if USE_COLOUR else ("", "", "", "")


def snippet(hit: Hit, terms: list[ExactTerm], width: int = 320) -> str:
    text = hit.paragraph
    start = 0
    first = min((m.start() for t in terms for m in [t.python.search(text)] if m), default=None)
    if first is not None and first > width // 3:
        start = max(0, first - width // 3)
    piece = text[start:start + width]
    piece = ("..." if start else "") + piece + ("..." if start + width < len(text) else "")
    for term in terms:
        piece = term.python.sub(lambda m: f"{BOLD}{YELLOW}{m.group(0)}{RESET}", piece)
    return piece


def print_response(response: SearchResponse):
    timing = f"{response.search_ms:.1f} ms"
    if response.embed_ms:
        timing += f" (+ {response.embed_ms:.0f} ms to embed the question)"
    print(f"\n{BOLD}{DB_LABELS[response.database]}{RESET} | {response.mode} | {timing} | scores: {response.score_kind}")
    if response.exact_terms:
        print("Exact terms: " + ", ".join(f'"{t}"' for t in response.exact_terms))
    if not response.hits:
        print("  No matching paragraphs.")
    terms = extract_exact_terms(response.query)
    for rank, hit in enumerate(response.hits, 1):
        year = f" ({hit.year})" if hit.year else ""
        para = f" | para {hit.para_no}" if hit.para_no is not None else ""
        if hit.parts_in_para and hit.parts_in_para > 1:
            para += f" (part {hit.part} of {hit.parts_in_para})"
        print(f"\n{rank:>2}. {BOLD}{hit.case_name}{RESET}{year}{para}    {DIM}score {hit.score:.4f}{RESET}")
        print("    " + snippet(hit, terms))
        tags = [f"contains {t}" for t in hit.matched_terms]
        if hit.source_url:
            tags.append(hit.source_url)
        if tags:
            print(f"    {DIM}{' | '.join(tags)}{RESET}")
    for note in response.notes:
        print(f"\n  Note: {note}")


# ---------------------------------------------------------------- command line

def run_query(engine, query, args):
    databases = list(engine.backends) if args.db == "both" else [args.db]
    responses = []
    for database in databases:
        try:
            responses.append(engine.search(query, database, args.mode, args.k, args.category, args.strict))
        except BackendUnavailable as exc:
            print(f"\n{DB_LABELS.get(database, database)}: {exc}")
    if args.json:
        print(json.dumps([asdict(r) for r in responses], ensure_ascii=False, indent=2))
    else:
        for response in responses:
            print_response(response)


def interactive(engine, args):
    print("ATL search. Type a question, or a command: :db postgres|qdrant|both  :mode semantic|keyword|hybrid"
          "  :k 10  :category property|tenant|property_and_tenant|all  :strict on|off  :q")
    while True:
        try:
            line = input(f"\n[{args.db} | {args.mode} | k={args.k}] question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in (":q", ":quit", "exit"):
            return
        if line.startswith(":"):
            command, _, value = line[1:].partition(" ")
            value = value.strip()
            if command == "db" and value in (*DATABASES, "both"):
                args.db = value
            elif command == "mode" and value in MODES:
                args.mode = value
            elif command == "k" and value.isdigit():
                args.k = int(value)
            elif command == "category" and value in (*CATEGORIES, "all"):
                args.category = None if value == "all" else value
            elif command == "strict" and value in ("on", "off"):
                args.strict = value == "on"
            else:
                print("Unknown command.")
            continue
        run_query(engine, line, args)


def main():
    parser = argparse.ArgumentParser(description="Search the ATL judgment paragraphs.")
    parser.add_argument("query", nargs="*", help="your question; leave empty for interactive mode")
    parser.add_argument("--db", choices=(*DATABASES, "both"), default="both")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    parser.add_argument("--category", choices=CATEGORIES, help="only property, tenant or property_and_tenant cases")
    parser.add_argument("--strict", action="store_true", help="every result must contain the exact legal references")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    args = parser.parse_args()

    databases = DATABASES if args.db == "both" else (args.db,)
    engine = SearchEngine(databases)
    for name, reason in engine.unavailable.items():
        print(f"{DB_LABELS[name]} unavailable: {reason}", file=sys.stderr)
    if not engine.backends:
        raise SystemExit(1)

    try:
        if args.query:
            run_query(engine, " ".join(args.query), args)
        else:
            interactive(engine, args)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
