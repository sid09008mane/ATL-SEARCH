"""Run every test question on both databases in every search mode, and record the results.

    python scripts/compare.py                 # writes eval/results.csv and prints a summary
    python scripts/compare.py --repeats 5     # time each search 5 times and keep the median

Each question is embedded once, so both databases get the identical question vector and the
timings measure only the database search. Before running, load both databases:
    python scripts/load_postgres.py
    python scripts/load_qdrant.py
"""
import argparse
import csv
import json
import statistics

from config import QUESTIONS_FILE, RESULTS_FILE
from search import DB_LABELS, MODES, SearchEngine, make_term

COLUMNS = [
    "question_id", "question_type", "database", "search_mode", "latency_ms", "results_returned",
    "top1_case", "top1_score", "expected_case_in_top5", "exact_term_in_top5", "notes",
]


def yes_no(value):
    return "yes" if value else "no"


def evaluate(question, response):
    cases = [hit.case_name for hit in response.hits]
    expected = question.get("expected_cases") or []
    exact = question.get("exact_term")
    if exact:
        term = make_term(exact)
        exact_found = yes_no(any(term.python.search(hit.paragraph) for hit in response.hits))
    else:
        exact_found = "n/a"
    notes = list(response.notes)
    if question["type"] == "no_answer" and response.hits:
        notes.insert(0, "should find nothing relevant; check top score")
    return {
        "results_returned": len(response.hits),
        "top1_case": cases[0] if cases else "",
        "top1_score": f"{response.hits[0].score:.4f}" if response.hits else "",
        "expected_case_in_top5": yes_no(any(c in expected for c in cases)) if expected else "n/a",
        "exact_term_in_top5": exact_found,
        "notes": " ".join(notes),
    }


def summarise(rows):
    print(f"\n{'Database':<22}{'Mode':<10}{'Expected case found':>21}{'Exact term found':>18}{'Median ms':>11}")
    for database in dict.fromkeys(r["database"] for r in rows):
        for mode in MODES:
            subset = [r for r in rows if r["database"] == database and r["search_mode"] == mode]
            if not subset:
                continue
            answerable = [r for r in subset if r["expected_case_in_top5"] != "n/a"]
            with_term = [r for r in subset if r["exact_term_in_top5"] != "n/a" and r["question_type"] != "no_answer"]
            found = sum(r["expected_case_in_top5"] == "yes" for r in answerable)
            term_found = sum(r["exact_term_in_top5"] == "yes" for r in with_term)
            median = statistics.median(float(r["latency_ms"]) for r in subset)
            print(f"{DB_LABELS[database]:<22}{mode:<10}{f'{found}/{len(answerable)}':>21}"
                  f"{f'{term_found}/{len(with_term)}':>18}{median:>11.2f}")
    no_answer = [r for r in rows if r["question_type"] == "no_answer" and r["search_mode"] == "semantic"]
    if no_answer:
        print("\nQuestions with no answer in the corpus (semantic top score; lower is better):")
        for r in no_answer:
            print(f"  {r['question_id']}  {DB_LABELS[r['database']]:<20} top score {r['top1_score'] or '-'}  ({r['top1_case'] or 'nothing'})")


def main():
    parser = argparse.ArgumentParser(description="Compare Postgres + pgvector and Qdrant on the test questions.")
    parser.add_argument("--repeats", type=int, default=3, help="time each search this many times (median kept)")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    engine = SearchEngine()
    for name, reason in engine.unavailable.items():
        print(f"Skipping {DB_LABELS[name]}: {reason}")
    if not engine.backends:
        raise SystemExit(1)

    print(f"Embedding {len(questions)} questions...")
    vectors = {q["id"]: engine.embed(q["question"])[0] for q in questions}

    first = questions[0]
    for database in engine.backends:  # warm-up, so the first timed search is not penalised
        for mode in MODES:
            engine.search(first["question"], database, mode, args.k, query_vector=vectors[first["id"]])

    rows = []
    for question in questions:
        for database in engine.backends:
            for mode in MODES:
                timings, response = [], None
                for _ in range(max(1, args.repeats)):
                    response = engine.search(question["question"], database, mode, args.k,
                                             query_vector=vectors[question["id"]])
                    timings.append(response.search_ms)
                rows.append({
                    "question_id": question["id"],
                    "question_type": question["type"],
                    "database": database,
                    "search_mode": mode,
                    "latency_ms": f"{statistics.median(timings):.2f}",
                    **evaluate(question, response),
                })
    engine.close()

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_FILE}")
    summarise(rows)


if __name__ == "__main__":
    main()
