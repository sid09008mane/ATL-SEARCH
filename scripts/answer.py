"""Ask a question, get a plain-language answer grounded in the judgments, with citations.

    python scripts/answer.py "The tenant did not pay rent. Can the landlord evict him?"
    python scripts/answer.py "Can someone become owner by living on my land?" --db qdrant -k 8
    python scripts/answer.py "..." --json
    python scripts/answer.py                         # interactive: ask many questions

The "brain" is gpt-oss-20b, an open-weight model, called through Groq's API. Put your key in a
.env file in the project folder:  GROQ_API_KEY=gsk_...   (free key: https://console.groq.com/keys)

How the answer stays grounded (ATL's "no claim without a matching precedent" rule):
  1. search.py finds the best paragraphs with hybrid search
  2. if the question names a section that no judgment contains, it stops there, with no model call
  3. the model sees only those paragraphs, numbered [1], [2], ..., and must cite one after every point
  4. if the paragraphs do not answer the question, the model must reply NO_MATCH
  5. every citation in the answer is checked; citations to sources it was not given are flagged
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_REASONING_EFFORT
from search import BOLD, DB_LABELS, DIM, RESET, YELLOW, BackendUnavailable, SearchEngine

NO_MATCH = "NO_MATCH"
DISCLAIMER = "This explains how courts decided past cases. It is not legal advice for your situation."

SYSTEM_PROMPT = """You are the legal research assistant of Any Time Law (ATL). You help ordinary people in India understand how the Supreme Court has decided disputes like theirs.

Rules:
1. Use ONLY the numbered sources in the user's message. They are paragraphs from Indian Supreme Court judgments. Do not use any outside knowledge of law, cases or statutes.
2. After every sentence that states a legal point or a fact, cite its source in square brackets, like [1] or [2][4]. Never cite a number that is not in the sources.
3. If the sources do not answer the question, reply with exactly NO_MATCH and nothing else.
4. If the sources answer only part of the question, answer that part and say plainly what they do not cover.
5. Do not predict who will win and do not tell the person what to do. Explain what the courts held.
6. Write in plain, simple language for someone who is not a lawyer, in the same language as the question. Name a case the first time you rely on it. Keep the answer under 200 words."""


@dataclass
class Source:
    number: int
    case_name: str
    year: int | None
    para_no: int | None
    source_url: str | None
    paragraph: str


@dataclass
class Answer:
    question: str
    status: str                      # "answered", "no_match" or "error"
    text: str
    sources: list[Source]
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
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
        self.extra = {}
        if reasoning_effort:
            self.extra["reasoning_effort"] = reasoning_effort
        if "groq.com" in base_url:
            self.extra["extra_body"] = {"include_reasoning": False}  # Groq: return only the final answer

    def stream(self, messages):
        """Yield the answer text piece by piece."""
        import openai

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, stream=True,
                temperature=0.2, max_completion_tokens=2048, **self.extra,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except openai.AuthenticationError as exc:
            raise LLMError("The API key was rejected. Check GROQ_API_KEY in your .env file.") from exc
        except openai.RateLimitError as exc:
            raise LLMError("Rate limit reached (Groq's free tier allows a set number of requests per minute). "
                           "Wait a minute and try again.") from exc
        except openai.NotFoundError as exc:
            raise LLMError(f"The model '{self.model}' was not found at {LLM_BASE_URL}.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"Cannot reach {LLM_BASE_URL}. Check your internet connection.") from exc
        except openai.APIError as exc:
            raise LLMError(f"The model API returned an error: {exc}") from exc


# ---------------------------------------------------------------- building the prompt

def build_sources(hits) -> list[Source]:
    return [Source(number=i, case_name=h.case_name, year=h.year, para_no=h.para_no,
                   source_url=h.source_url, paragraph=h.paragraph) for i, h in enumerate(hits, 1)]


def build_messages(question, sources, notes=()):
    blocks = []
    for s in sources:
        label = s.case_name + (f" ({s.year})" if s.year else "") + (f", para {s.para_no}" if s.para_no is not None else "")
        blocks.append(f"[{s.number}] {label}\n{s.paragraph}")
    user = f"Question: {question}\n\nSources:\n\n" + "\n\n".join(blocks)
    if notes:
        user += "\n\nSearch notes:\n" + "\n".join(f"- {n}" for n in notes)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def check_citations(text, count):
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    valid = [n for n in cited if 1 <= n <= count]
    invalid = [n for n in cited if not 1 <= n <= count]
    return valid, invalid


# ---------------------------------------------------------------- the answer engine

class AnswerEngine:
    def __init__(self, database="postgres", k=5, mode="hybrid", category=None, brain=None):
        self.database, self.k, self.mode, self.category = database, k, mode, category
        self.brain = brain or Brain()  # check the API key first, before connecting to a database
        self.search_engine = SearchEngine((database,))
        if database not in self.search_engine.backends:
            raise BackendUnavailable(self.search_engine.unavailable[database])

    def ask(self, question, on_token=None) -> Answer:
        """Search, then ask the model. on_token(text) receives the answer as it is written."""
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

        messages = build_messages(question, sources, found.notes)
        started = time.perf_counter()
        pieces, shown = [], False
        for piece in self.brain.stream(messages):
            pieces.append(piece)
            so_far = "".join(pieces).lstrip()
            if on_token and not shown and not NO_MATCH.startswith(so_far[:len(NO_MATCH)]):
                shown = True
                on_token(so_far)
            elif on_token and shown:
                on_token(piece)
        answer.llm_ms = (time.perf_counter() - started) * 1000
        text = "".join(pieces).strip()
        if on_token and not shown and text and not text.upper().startswith(NO_MATCH):
            on_token(text)  # a very short answer that never got past the NO_MATCH check

        if not text or text.upper().startswith(NO_MATCH):
            answer.status = "no_match"
            answer.text = "The closest judgments found do not answer this question, so no answer is given."
            return answer

        answer.text = text
        answer.cited, answer.invalid_citations = check_citations(text, len(sources))
        if answer.invalid_citations:
            answer.warnings.append("The answer cites sources it was not given: "
                                   + ", ".join(f"[{n}]" for n in answer.invalid_citations) + ". Treat those parts as unverified.")
        if not answer.cited:
            answer.warnings.append("The answer has no citations, so it cannot be checked against the judgments.")
        return answer

    def close(self):
        self.search_engine.close()


# ---------------------------------------------------------------- display and command line

def print_answer_footer(answer: Answer, show_sources: bool):
    if answer.status == "no_match":
        print(f"{YELLOW}{answer.text}{RESET}")
    print()
    for warning in answer.warnings:
        print(f"{YELLOW}Warning: {warning}{RESET}")
    listed = answer.sources if show_sources else [s for s in answer.sources if s.number in answer.cited]
    if listed:
        print(f"{BOLD}Sources{RESET}")
        for s in listed:
            year = f" ({s.year})" if s.year else ""
            para = f", para {s.para_no}" if s.para_no is not None else ""
            mark = "" if s.number in answer.cited or answer.status != "answered" else f" {DIM}(not cited){RESET}"
            print(f"  [{s.number}] {s.case_name}{year}{para}{mark}")
            if s.source_url:
                print(f"      {DIM}{s.source_url}{RESET}")
            if show_sources:
                print(textwrap.indent(textwrap.fill(s.paragraph[:400] + ("..." if len(s.paragraph) > 400 else ""), 96), "      "))
    timing = f"search {answer.search_ms:.0f} ms"
    if answer.embed_ms:
        timing += f" + embed {answer.embed_ms:.0f} ms"
    if answer.llm_ms:
        timing += f" + {answer.model} {answer.llm_ms / 1000:.1f} s"
    print(f"\n{DIM}{DB_LABELS[answer.database]} | {timing}{RESET}")
    print(f"{DIM}{DISCLAIMER}{RESET}")


def ask_and_print(engine: AnswerEngine, question: str, args):
    if args.json:
        try:
            answer = engine.ask(question)
        except LLMError as exc:
            answer = Answer(question=question, status="error", text=str(exc), sources=[], database=engine.database)
        print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
        return
    print()
    try:
        answer = engine.ask(question, on_token=lambda t: print(t, end="", flush=True))
    except LLMError as exc:
        print(f"{YELLOW}{exc}{RESET}")
        return
    if answer.status == "answered":
        print()
    print_answer_footer(answer, args.show_sources)


def interactive(engine: AnswerEngine, args):
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
