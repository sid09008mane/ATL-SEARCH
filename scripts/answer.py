"""Ask a question, get a plain-language answer grounded in the judgments, with citations.

    python scripts/answer.py "The tenant did not pay rent. Can the landlord evict him?"
    python scripts/answer.py "Can someone become owner by living on my land?" --db qdrant -k 8
    python scripts/answer.py "..." --json
    python scripts/answer.py                         # interactive: loads the model once, ask many questions

The "brain" is gpt-oss-20b, an open-weight model, called through Groq's API. Put your key in a
.env file in the project folder:  GROQ_API_KEY=gsk_...   (free key: https://console.groq.com/keys)

How the answer stays grounded (ATL's "no claim without a matching precedent" rule):
  1. search.py finds the best paragraphs with hybrid search
  2. if the question names a section that no judgment contains, it stops there, with no model call
  3. the model sees only those paragraphs, numbered [1], [2], ..., and must return its answer as
     separate points, each listing the sources it comes from and the Act it applies
  4. the code attaches the citations itself; a point with no valid source is removed, so every
     sentence shown is tied to a judgment
  5. if the paragraphs do not answer the question, the model says so and nothing is invented
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
import time
from dataclasses import asdict, dataclass, field

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_REASONING_EFFORT
from search import BOLD, DB_LABELS, DIM, RESET, YELLOW, BackendUnavailable, SearchEngine

DISCLAIMER = "This explains how courts decided past cases. It is not legal advice for your situation."
STATE_LAW_NOTE = ("Rent and tenancy laws differ from state to state. These points come from the laws named "
                  "above; check which law applies where the property is.")
STATE_LAW_WORDS = re.compile(r"rent|tenan|lease|accommodation|land reforms|land revenue", re.IGNORECASE)

SYSTEM_PROMPT = """You are the legal research assistant of Any Time Law (ATL). You help ordinary people in India understand how the Supreme Court has decided disputes like theirs.

Answer using ONLY the numbered sources in the user's message. They are paragraphs from Indian Supreme Court judgments. Do not use any outside knowledge of law, cases or statutes.

Return JSON with:
- "status": "answered", or "no_match" if the sources do not answer the question (then "points" is an empty list).
- "points": 2 to 6 short statements in plain language for someone who is not a lawyer. The first point answers the question directly. Each statement makes ONE point, names the case it comes from (for example "In Chinnamma v. Gopalan, ..."), and lists in "sources" the number of every source that supports it. Never write a statement you cannot tie to a source. Do not put [1]-style numbers inside the text.
- "law" in each point: the Act the cited source is applying, with the state if it is a state law, written as the source names it (for example "Kerala Buildings (Lease and Rent Control) Act, 1965"). Use "" if the source does not name the Act.
- "not_covered": one sentence saying what part of the question the sources do not answer, or "" if they answer all of it.

Rules:
- Rent, tenancy and land laws differ from state to state. Never present one state's rule as if it applies everywhere.
- Do not predict who will win and do not tell the person what to do. Explain what the courts held.
- Write in the same language as the question."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "no_match"]},
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "integer"}},
                    "law": {"type": "string"},
                },
                "required": ["text", "sources", "law"],
                "additionalProperties": False,
            },
        },
        "not_covered": {"type": "string"},
    },
    "required": ["status", "points", "not_covered"],
    "additionalProperties": False,
}


@dataclass
class Source:
    number: int
    case_name: str
    year: int | None
    para_no: int | None
    source_url: str | None
    paragraph: str


@dataclass
class Point:
    text: str
    sources: list[int]
    law: str


@dataclass
class Answer:
    question: str
    status: str                      # "answered", "no_match" or "error"
    text: str                        # the whole answer as plain text, citations included
    sources: list[Source]
    points: list[Point] = field(default_factory=list)
    laws: list[str] = field(default_factory=list)
    not_covered: str = ""
    cited: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    database: str = ""
    model: str = LLM_MODEL
    search_ms: float = 0.0
    embed_ms: float = 0.0
    llm_ms: float = 0.0


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------- the model

class Brain:
    """A thin wrapper around any OpenAI-compatible chat API (Groq by default)."""

    def __init__(self, model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=LLM_API_KEY,
                 reasoning_effort=LLM_REASONING_EFFORT):
        if not api_key:
            raise LLMError(
                "No API key found. Get a free Groq key at https://console.groq.com/keys, then create a file "
                "called .env in the project folder containing:\n    GROQ_API_KEY=gsk_your_key_here"
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=2)
        self.model = model
        self.base_url = base_url
        self.extra = {}
        if reasoning_effort:
            self.extra["reasoning_effort"] = reasoning_effort
        if "groq.com" in base_url:
            self.extra["extra_body"] = {"include_reasoning": False}  # Groq: return only the final answer
        # Strict JSON schema first; fall back to plain JSON mode for APIs that do not support it.
        self.formats = [
            {"type": "json_schema", "json_schema": {"name": "atl_answer", "strict": True, "schema": ANSWER_SCHEMA}},
            {"type": "json_object"},
        ]

    def answer_json(self, messages) -> dict:
        """Return the model's answer as a dict following ANSWER_SCHEMA."""
        import openai

        last_error = None
        while self.formats:
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, response_format=self.formats[0],
                    temperature=0.2, max_completion_tokens=2048, **self.extra,
                )
                content = response.choices[0].message.content or ""
                try:
                    return json.loads(content)
                except json.JSONDecodeError as exc:
                    raise LLMError("The model did not return valid JSON. Try again.") from exc
            except openai.BadRequestError as exc:
                message = str(exc).lower()
                if len(self.formats) > 1 and any(w in message for w in ("response_format", "schema", "json")):
                    last_error = exc
                    self.formats.pop(0)  # this API does not accept strict schemas; use plain JSON mode
                    continue
                raise LLMError(f"The model API rejected the request: {exc}") from exc
            except openai.AuthenticationError as exc:
                raise LLMError("The API key was rejected. Check GROQ_API_KEY in your .env file.") from exc
            except openai.RateLimitError as exc:
                raise LLMError("Rate limit reached (Groq's free tier allows a set number of requests per "
                               "minute). Wait a minute and try again.") from exc
            except openai.NotFoundError as exc:
                raise LLMError(f"The model '{self.model}' was not found at {self.base_url}.") from exc
            except openai.APIConnectionError as exc:
                raise LLMError(f"Cannot reach {self.base_url}. Check your internet connection.") from exc
            except openai.APIError as exc:
                raise LLMError(f"The model API returned an error: {exc}") from exc
        raise LLMError(f"The model API rejected every answer format: {last_error}")


# ---------------------------------------------------------------- prompt and checks

def build_sources(hits) -> list[Source]:
    return [Source(number=i, case_name=h.case_name, year=h.year, para_no=h.para_no,
                   source_url=h.source_url, paragraph=h.paragraph) for i, h in enumerate(hits, 1)]


def source_label(s: Source) -> str:
    return s.case_name + (f" ({s.year})" if s.year else "") + (f", para {s.para_no}" if s.para_no is not None else "")


def build_messages(question, sources, notes=()):
    blocks = [f"[{s.number}] {source_label(s)}\n{s.paragraph}" for s in sources]
    user = f"Question: {question}\n\nSources:\n\n" + "\n\n".join(blocks)
    if notes:
        user += "\n\nSearch notes:\n" + "\n".join(f"- {n}" for n in notes)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def check_points(raw: dict, source_count: int):
    """Keep only points tied to sources the model was actually given."""
    points, warnings, dropped = [], [], 0
    invalid_numbers = set()
    for item in raw.get("points") or []:
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s*\[\d+\]", "", str(item.get("text", ""))).strip()
        numbers = []
        for n in item.get("sources") or []:
            try:
                n = int(n)
            except (TypeError, ValueError):
                continue
            if 1 <= n <= source_count:
                if n not in numbers:
                    numbers.append(n)
            else:
                invalid_numbers.add(n)
        if not text:
            continue
        if not numbers:
            dropped += 1
            continue
        points.append(Point(text=text, sources=sorted(numbers), law=str(item.get("law") or "").strip()))
    if invalid_numbers:
        warnings.append("The model referred to sources it was not given ("
                        + ", ".join(f"[{n}]" for n in sorted(invalid_numbers)) + "); those references were ignored.")
    if dropped:
        warnings.append(f"Removed {dropped} statement{'s' if dropped > 1 else ''} the model could not tie to any judgment.")
    return points, warnings


def cases_for(numbers, by_number) -> str:
    """'Chinnamma v. Gopalan (1995), paras 7, 8; ...' for the given source numbers, grouped by case."""
    grouped: dict[str, list] = {}
    for n in numbers:
        s = by_number[n]
        name = s.case_name + (f" ({s.year})" if s.year else "")
        grouped.setdefault(name, [])
        if s.para_no is not None and s.para_no not in grouped[name]:
            grouped[name].append(s.para_no)
    parts = []
    for name, paras in grouped.items():
        paras = sorted(paras)
        parts.append(name + (f", para{'s' if len(paras) > 1 else ''} " + ", ".join(map(str, paras)) if paras else ""))
    return "; ".join(parts)


def unique(items):
    return list(dict.fromkeys(i for i in items if i))


# ---------------------------------------------------------------- the answer engine

class AnswerEngine:
    def __init__(self, database="postgres", k=5, mode="hybrid", category=None, brain=None):
        self.database, self.k, self.mode, self.category = database, k, mode, category
        self.brain = brain or Brain()  # check the API key first, before connecting to a database
        self.search_engine = SearchEngine((database,))
        if database not in self.search_engine.backends:
            raise BackendUnavailable(self.search_engine.unavailable[database])

    def warm_up(self):
        """Load the embedding model now, so the first question is as fast as the rest."""
        self.search_engine.embed("warm up")

    def ask(self, question) -> Answer:
        found = self.search_engine.search(question, self.database, self.mode, self.k, self.category)
        sources = build_sources(found.hits)
        answer = Answer(question=question, status="answered", text="", sources=sources, database=self.database,
                        model=self.brain.model, search_ms=found.search_ms, embed_ms=found.embed_ms)

        if found.exact_terms and set(found.missing_terms) == set(found.exact_terms):
            names = ", ".join(f'"{t}"' for t in found.missing_terms)
            answer.status = "no_match"
            answer.text = f"None of the judgments in the collection mentions {names}, so there is no precedent to answer from."
            return answer
        if not sources:
            answer.status = "no_match"
            answer.text = "No matching judgment was found for this question."
            return answer

        started = time.perf_counter()
        raw = self.brain.answer_json(build_messages(question, sources, found.notes))
        answer.llm_ms = (time.perf_counter() - started) * 1000

        points, warnings = check_points(raw, len(sources))
        answer.warnings = warnings
        answer.not_covered = str(raw.get("not_covered") or "").strip()
        if raw.get("status") == "no_match" or not points:
            answer.status = "no_match"
            answer.text = ("The closest judgments found do not answer this question, so no answer is given."
                           if not warnings else
                           "The model's answer could not be tied to the judgments found, so it was withheld.")
            return answer

        answer.points = points
        answer.cited = sorted({n for p in points for n in p.sources})
        answer.laws = unique(p.law for p in points)
        answer.text = " ".join(f"{p.text} " + "".join(f"[{n}]" for n in p.sources) for p in points)
        if answer.not_covered:
            answer.text += f" Not covered: {answer.not_covered}"
        return answer

    def close(self):
        self.search_engine.close()


# ---------------------------------------------------------------- display and command line

def print_answer(answer: Answer, show_sources: bool):
    print()
    if answer.status != "answered":
        print(f"{YELLOW}{answer.text}{RESET}")
    else:
        by_number = {s.number: s for s in answer.sources}
        for i, point in enumerate(answer.points, 1):
            cites = "".join(f"[{n}]" for n in point.sources)
            print(textwrap.fill(f"{point.text} {BOLD}{cites}{RESET}", 96,
                                initial_indent=f"{i:>2}. ", subsequent_indent="    "))
            basis = cases_for(point.sources, by_number)
            if point.law:
                basis += f" | {point.law}"
            for line in textwrap.wrap(basis, 92):
                print(f"    {DIM}{line}{RESET}")
        if answer.not_covered:
            print(f"\n{BOLD}Not covered:{RESET} {answer.not_covered}")
        if answer.laws:
            print(f"\n{BOLD}Laws applied:{RESET} " + "; ".join(answer.laws))
            if any(STATE_LAW_WORDS.search(law) for law in answer.laws):
                print(f"{YELLOW}{STATE_LAW_NOTE}{RESET}")

    for warning in answer.warnings:
        print(f"{YELLOW}Warning: {warning}{RESET}")

    listed = answer.sources if show_sources else [s for s in answer.sources if s.number in answer.cited]
    if listed:
        print(f"\n{BOLD}Sources{RESET}")
        for s in listed:
            mark = "" if s.number in answer.cited or answer.status != "answered" else f" {DIM}(not used){RESET}"
            print(f"  [{s.number}] {source_label(s)}{mark}")
            if s.source_url:
                print(f"      {DIM}{s.source_url}{RESET}")
            if show_sources:
                print(textwrap.indent(textwrap.fill(s.paragraph[:400] + ("..." if len(s.paragraph) > 400 else ""), 92), "      "))

    timing = f"search {answer.search_ms:.0f} ms"
    if answer.embed_ms:
        timing += f" + embed {answer.embed_ms:.0f} ms"
        if answer.embed_ms > 3000:
            timing += " (includes loading the model once)"
    if answer.llm_ms:
        timing += f" + {answer.model} {answer.llm_ms / 1000:.1f} s"
    print(f"\n{DIM}{DB_LABELS[answer.database]} | {timing}{RESET}")
    print(f"{DIM}{DISCLAIMER}{RESET}")


def ask_and_print(engine: AnswerEngine, question: str, args):
    try:
        if not args.json:
            print(f"{DIM}Searching and asking {engine.brain.model}...{RESET}", flush=True)
        answer = engine.ask(question)
    except LLMError as exc:
        if args.json:
            print(json.dumps({"question": question, "status": "error", "text": str(exc)}, indent=2))
        else:
            print(f"{YELLOW}{exc}{RESET}")
        return
    if args.json:
        print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
    else:
        print_answer(answer, args.show_sources)


def interactive(engine: AnswerEngine, args):
    print(f"{DIM}Loading the embedding model (once per session)...{RESET}", flush=True)
    started = time.perf_counter()
    engine.warm_up()
    print(f"{DIM}Ready in {time.perf_counter() - started:.0f} s.{RESET}")
    print("ATL answers. Ask a question, or type :sources on|off, :k 8, :q to quit.")
    while True:
        try:
            line = input(f"\n[{engine.database} | k={engine.k}] question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in (":q", ":quit", "exit"):
            return
        if line.startswith(":"):
            command, _, value = line[1:].partition(" ")
            if command == "sources" and value.strip() in ("on", "off"):
                args.show_sources = value.strip() == "on"
            elif command == "k" and value.strip().isdigit():
                engine.k = int(value)
            else:
                print("Unknown command.")
            continue
        ask_and_print(engine, line, args)


def main():
    parser = argparse.ArgumentParser(description="Answer a question from the ATL judgments, with citations.")
    parser.add_argument("question", nargs="*", help="your question; leave empty for interactive mode")
    parser.add_argument("--db", choices=("postgres", "qdrant"), default="postgres")
    parser.add_argument("-k", type=int, default=5, help="how many paragraphs the model reads (default 5)")
    parser.add_argument("--category", choices=("property", "tenant", "property_and_tenant"))
    parser.add_argument("--show-sources", action="store_true", help="print every paragraph the model was given")
    parser.add_argument("--json", action="store_true", help="print the full answer as JSON")
    args = parser.parse_args()

    try:
        engine = AnswerEngine(database=args.db, k=args.k, category=args.category)
    except (LLMError, BackendUnavailable) as exc:
        raise SystemExit(str(exc))
    try:
        if args.question:
            ask_and_print(engine, " ".join(args.question), args)
        else:
            interactive(engine, args)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
