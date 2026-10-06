"""Core retrieval (RAG) over the crawled Technyx facts corpus.

This is the entire "grounding" mechanism for this chatbot - there is no
fine-tuning here. The base instruction-tuned model never sees the facts
during training; instead, at answer time, we embed the user's question,
retrieve the most relevant facts by cosine similarity, and hand them to
the model as verified context in the prompt. The model looks the fact up
instead of relying on having memorized it.

Facts and their embeddings live in Postgres with the pgvector extension
(Supabase in production - see supabase/migrations and scripts/sync_facts.py).
Searching is a cosine nearest-neighbour query in SQL; the ranking boosts
below are then applied in Python to the candidate rows. Updating the
knowledge base is a database write (scripts/sync_facts.py) - no redeploy.

Embeddings come straight from `transformers` using an ungated
sentence-embedding checkpoint, selected via the EMBEDDING_MODEL env var
(pooling strategy and L2 normalization applied per-model - see
_MODEL_CONFIGS) - no extra `sentence-transformers` package needed for such
a simple use case.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# Which embedding checkpoint to use - chosen once at process startup, same
# convention as LLM_BACKEND/LLM_API_MODEL in api.py. Changing this requires
# re-running scripts/sync_facts.py: embeddings from different models live
# in different vector spaces and aren't comparable, and open_index() below
# refuses to open a database whose vectors were built with a different model
# than the one currently configured. The database column is vector(768),
# i.e. the bge-base default - see the migration for other dimensions.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-base-en-v1.5"

# Per-model pooling strategy and (optional) query-side instruction prefix.
# These aren't uniform across embedding checkpoints - assuming one strategy
# for every model silently produces poor-quality embeddings with no error.
# Add an entry here before pointing EMBEDDING_MODEL at a new checkpoint.
#   pooling: "mean" (mean-pool token embeddings, e.g. sentence-transformers
#     MiniLM family) or "cls" (use the [CLS] token, e.g. BGE family).
#   query_instruction: prefix added to queries only (never to indexed facts)
#     - BGE models are trained to expect this for retrieval-style queries.
#   min_score: RetrievalIndex.search()'s default relevance cutoff for this
#     model. Cosine-similarity scale is NOT comparable across embedding
#     models - BGE's off-topic-vs-on-topic scores both sit much higher than
#     MiniLM's (e.g. an unrelated query scored 0.30-0.45 against this
#     corpus with BGE, where MiniLM scored well under 0.3 - the guardrail
#     that treats "no results" as "off-topic question" silently broke when
#     BGE was first plugged in with MiniLM's threshold). Measured on this
#     corpus with the same 5 off-topic / 5 on-topic probe queries ("What is
#     the capital of France?" / "Where is Technyx headquartered?" etc.):
#     BGE-large topped out at 0.454 off-topic vs 0.718 on-topic; BGE-base
#     topped out at 0.452 off-topic vs 0.815 on-topic. 0.55 sits in both
#     gaps with margin on either side. Re-measure this the same way before
#     changing embedding models.
_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "pooling": "mean",
        "query_instruction": None,
        "min_score": 0.3,
        "dimensions": 384,
    },
    "BAAI/bge-large-en-v1.5": {
        "pooling": "cls",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "min_score": 0.55,
        "dimensions": 1024,
    },
    "BAAI/bge-base-en-v1.5": {
        "pooling": "cls",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "min_score": 0.55,
        "dimensions": 768,
    },
}


def _model_config() -> dict[str, Any]:
    config = _MODEL_CONFIGS.get(EMBEDDING_MODEL_NAME)
    if config is None:
        raise ValueError(
            f"No pooling config for EMBEDDING_MODEL={EMBEDDING_MODEL_NAME!r}. "
            f"Add an entry to _MODEL_CONFIGS in retrieval.py - defaulting silently "
            f"risks embedding with the wrong pooling strategy for this checkpoint."
        )
    return config


def _mean_pool(last_hidden_state, attention_mask):

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


_embedding_model_cache: dict[str, Any] = {}


def _get_embedding_model():
    """Load the embedding tokenizer/model once per process and reuse it.

    Measured necessary: embed_texts() is called on every single retrieval
    search (once per chat turn in scripts/chat.py, once per test case in
    scripts/evaluate.py), and re-running from_pretrained() each time cost
    1.5-10s of pure model-loading overhead on top of generation time, on
    every single turn, for a model that never changes within a process.
    """
    if "model" not in _embedding_model_cache:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        model.eval()
        _embedding_model_cache["tokenizer"] = tokenizer
        _embedding_model_cache["model"] = model
    return _embedding_model_cache["tokenizer"], _embedding_model_cache["model"]


def embed_texts(texts: list[str], batch_size: int = 32, is_query: bool = False) -> np.ndarray:
    """Embed a list of texts into L2-normalized vectors using the configured
    sentence-embedding model (EMBEDDING_MODEL_NAME). Returns an (N, dim)
    float32 array.

    is_query distinguishes a retrieval query from an indexed fact/passage:
    some models (e.g. BGE) expect a fixed instruction prefix on the query
    side only - applying it to indexed facts too would embed every fact
    with the same misleading prefix.
    """
    import torch

    tokenizer, model = _get_embedding_model()
    config = _model_config()

    if is_query and config["query_instruction"]:
        texts = [config["query_instruction"] + t for t in texts]

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # 512 (both BGE and MiniLM's max sequence length) is a second,
            # independent safety net against silent truncation, on top of
            # scripts/crawl_technyx.py splitting long paragraphs into
            # sub-800-char chunks before they ever reach here - covers any
            # already-indexed fact that predates that splitting logic.
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            output = model(**encoded)
            if config["pooling"] == "cls":
                pooled = output.last_hidden_state[:, 0]
            else:
                pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(normalized.numpy())

    return np.concatenate(all_embeddings, axis=0).astype(np.float32)


# Phrases that mark a fact as explicitly stating company-identity
# information (e.g. "our head office is in Dubai"), as opposed to a raw
# fragment that merely shares keywords with an identity question (e.g. a
# bare office address). Measured necessary: without a boost, "Where is
# Technyx based, and how long have they been operating?" ranked the one
# fact that actually names the head office (score 0.640) behind a
# same-topic address fragment that scores higher on raw keyword overlap
# alone (0.675) but doesn't say which office is the headquarters - the
# model then answered from whichever fact happened to rank first, naming
# the wrong city. Only one fact in this corpus matches any of these
# phrases (see the facts in data/knowledge/), so the risk of this boost
# distorting unrelated queries is minimal.
_AUTHORITATIVE_PHRASES = ("head office", "headquarters", "headquartered")
_AUTHORITATIVE_FACT_BOOST = 0.1


def is_authoritative_identity_fact(fact: dict[str, Any]) -> bool:
    text = fact["fact"].lower()
    return any(phrase in text for phrase in _AUTHORITATIVE_PHRASES)


# Testimonial quotes (fact_type "client") are real ("Technyx has been an
# incredible partner..." - Blue Barracuda, etc.) but embed poorly against
# the word "testimonials" itself, since the quotes never use that word.
# Measured: asking "Does Technyx have any client testimonials?" scored the
# best real testimonial at 0.527 - just below the rank-8 cutoff (0.528) -
# so retrieval returned zero of the eleven real testimonials in the corpus
# and the model incorrectly said none exist. Unlike the identity-fact
# boost above, this one is deliberately query-conditional (only applied
# when the question itself is about testimonials/reviews/feedback) rather
# than always-on, precisely because an always-on boost was measured to
# leak into unrelated answers elsewhere (see README's "Measured results").
_TESTIMONIAL_QUERY_KEYWORDS = ("testimonial", "review", "feedback", "clients say", "client say")
_TESTIMONIAL_FACT_BOOST = 0.1


def _is_testimonial_query(query: str) -> bool:
    normalized = query.lower()
    return any(keyword in normalized for keyword in _TESTIMONIAL_QUERY_KEYWORDS)


_ANSWER_CARD_BOOST = 0.35


def _is_answer_card_match(fact: dict[str, Any], query: str) -> bool:
    """True when an answer card explicitly covers the user's intent."""
    keywords = fact.get("answer_card_keywords")
    if not isinstance(keywords, list):
        return False
    normalized = query.lower()
    return any(isinstance(keyword, str) and keyword in normalized for keyword in keywords)


# Columns selected for every candidate fact. Together with ``extra`` (any
# additional source-JSONL fields) they rebuild the same dict the old
# in-memory index returned, so pipeline.py/api.py are unaffected.
_FACT_COLUMNS = (
    "fact_id, fact, fact_type, source_url, source_page_title, source_section, source_excerpt, "
    "confidence, crawled_at, retrieval_text, answer_card_keywords, extra"
)

# One round trip returns every fact that could end up in the top-k after the
# Python-side boosts: the nearest neighbours by raw similarity, plus every
# fact that can receive a boost (authoritative, testimonial, matching answer
# card). A fact outside this set has neither a boost nor a raw score above
# the candidates', so it can never outrank them - results match an exact scan.
_SEARCH_SQL = f"""
with q as (select %(vec)s::vector as v),
nearest as (
  select fact_id from public.facts, q order by embedding <=> q.v limit %(candidates)s
),
boostable as (
  select fact_id from public.facts where is_authoritative
  union
  select fact_id from public.facts where %(testimonial)s and fact_type = 'client'
  union
  select fact_id from public.facts
  where answer_card_keywords is not null
    and exists (select 1 from unnest(answer_card_keywords) k where position(k in %(query_lower)s::text) > 0)
)
select {_FACT_COLUMNS}, is_authoritative, 1 - (embedding <=> q.v) as score
from public.facts, q
where fact_id in (select fact_id from nearest union select fact_id from boostable)
"""


def _row_to_fact(row: dict[str, Any]) -> dict[str, Any]:
    fact = {k: v for k, v in row.items() if v is not None and k not in ("extra", "is_authoritative", "score")}
    fact.update(row.get("extra") or {})
    if crawled_at := row.get("crawled_at"):
        fact["crawled_at"] = crawled_at.isoformat()
    return fact


class RetrievalIndex:
    """Similarity search over the ``public.facts`` table (pgvector)."""

    def __init__(self, pool) -> None:
        self.pool = pool

    def count(self) -> int:
        with self.pool.connection() as conn:
            row = conn.execute("select count(*) from public.facts").fetchone()
        return int(row[0])

    def close(self) -> None:
        self.pool.close()

    def search(self, query: str, top_k: int = 8, min_score: float | None = None) -> list[dict[str, Any]]:
        """Return up to top_k facts whose embedding cosine-similarity to the
        query (plus ranking boosts) is at least min_score, deduplicated by
        fact text, highest score first. An empty result means "nothing
        relevant found" - a useful signal that a question may be off-topic
        (see build_context_block).

        min_score defaults to the configured embedding model's measured
        cutoff (_MODEL_CONFIGS) - cosine-similarity scale differs by model,
        so a value tuned for one model is meaningless for another."""
        from psycopg.rows import dict_row

        if min_score is None:
            min_score = _model_config()["min_score"]
        query_vec = embed_texts([query], is_query=True)[0]
        testimonial_query = _is_testimonial_query(query)
        params = {
            "vec": query_vec,
            "candidates": max(top_k * 8, 64),
            "testimonial": testimonial_query,
            "query_lower": query.lower(),
        }
        with self.pool.connection() as conn:
            with conn.transaction():
                # HNSW returns at most ef_search rows; keep it above `candidates`.
                conn.execute("set local hnsw.ef_search = 200")
                with conn.cursor(row_factory=dict_row) as cur:
                    rows = cur.execute(_SEARCH_SQL, params).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            fact = _row_to_fact(row)
            score = float(row["score"])
            if row["is_authoritative"]:
                score += _AUTHORITATIVE_FACT_BOOST
            if testimonial_query and fact.get("fact_type") == "client":
                score += _TESTIMONIAL_FACT_BOOST
            if _is_answer_card_match(fact, query):
                score += _ANSWER_CARD_BOOST
            scored.append((score, fact))
        scored.sort(key=lambda item: (-item[0], item[1]["fact_id"]))

        results: list[dict[str, Any]] = []
        seen_fact_text: set[str] = set()
        for score, fact in scored:
            if score < min_score or len(results) >= top_k:
                break
            if fact["fact"] in seen_fact_text:
                continue
            seen_fact_text.add(fact["fact"])
            results.append({**fact, "score": score})
        return results


def _looks_like_nav_fragment(text: str) -> bool:
    """True for leftover navigation/tab-list artifacts from crawling (e.g.
    "About Technyx Work With Us Services Industries We Serve Platforms &
    Technology Get in Touch") that occasionally survive fact extraction.
    These add no answerable information and, worse, can get echoed verbatim
    into a generated answer if retrieved.

    Deliberately narrow (>=8 words, ends with no sentence punctuation, and
    is mostly Title-Cased) so it doesn't also exclude legitimate short
    facts like service/product names ("Digital Experience Platforms &
    Integrations") or addresses ("USA: McKinney, Texas, 5701 Sidney Lane").
    """
    words = text.split()
    if not (8 <= len(words) <= 25):
        return False
    if text.rstrip().endswith((".", "!", "?", ")", ":")):
        return False
    capitalized = sum(1 for w in words if w[:1].isupper())
    return capitalized / len(words) > 0.75


def _embedding_text(fact: dict[str, Any]) -> str:
    """Text actually embedded for a fact - not just the bare fact string.

    Many facts are short, keyword-dense fragments (e.g. "UAE: Technyx
    Consulting IFZA Business Park, DDP Dubai.") that embed poorly against a
    full natural-language question with a generic sentence-embedding model.
    Prepending the page title and section heading gives the embedding more
    natural-language context to align against (e.g. adding the words
    "Contact Us" / "Our Locations" to a location fact).
    """
    if retrieval_text := fact.get("retrieval_text"):
        return str(retrieval_text)
    return f"{fact['source_page_title']} - {fact['source_section']}: {fact['fact']}"


def _extract_text(content: Any) -> str:
    """Pull plain text out of a message's ``content`` field (either a bare
    string, or this project's list-of-parts form
    ``[{"type": "text", "text": "..."}]``)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def build_retrieval_query(user_input: str, history: list[dict[str, Any]] | None = None) -> str:
    """Construct the text actually embedded for a retrieval search.

    A short follow-up like "Which of those is the head office?" has almost
    no standalone semantic content - measured directly against this
    corpus: on its own it scored 0.19 (rank 63/883) against the one fact
    that actually answers it, versus 0.65-0.76 for questions that state
    their own subject. That's low enough to look like noise, and a
    generated answer that treats noise as verified context is how the
    chatbot ended up confidently naming the wrong city as company
    headquarters in testing (see README's "Measured results").

    Folding in the immediately preceding user turn gives the embedding an
    antecedent to match against - measured to raise that same fact's score
    to 0.61 (top 3) for the exact case above. Only the retrieval query is
    affected; the literal question shown to the model
    (see prompting.build_messages) is unchanged.
    """
    if not history:
        return user_input
    last_user_turn = next((m for m in reversed(history) if m.get("role") == "user"), None)
    if last_user_turn is None:
        return user_input
    prior_text = _extract_text(last_user_turn.get("content"))
    return f"{prior_text} {user_input}" if prior_text else user_input


def load_facts(*paths: str | Path) -> list[dict[str, Any]]:
    """Read facts from one or more JSONL files (crawled facts, then
    hand-curated answer cards), dropping navigation fragments and rejecting
    duplicate fact_ids across files."""
    facts: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fact = json.loads(line)
            if fact["fact_id"] in seen:
                raise ValueError(f"Duplicate fact_id {fact['fact_id']!r} (in {path})")
            seen.add(fact["fact_id"])
            if _looks_like_nav_fragment(fact["fact"]):
                skipped += 1
                continue
            facts.append(fact)
    if skipped:
        print(f"Skipping {skipped} nav-fragment-like fact(s).")
    return facts


def embedding_hash(fact: dict[str, Any]) -> str:
    """Identity of a fact's vector: changes when the embedded text or the
    embedding model changes, so a sync re-embeds exactly those facts."""
    payload = f"{EMBEDDING_MODEL_NAME}\n{_embedding_text(fact)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def open_index(database_url: str | None = None) -> RetrievalIndex:
    """Connect to the knowledge base, refusing to use one whose vectors were
    built with a different embedding model than the one currently configured
    (EMBEDDING_MODEL). Vectors from different models live in different,
    incomparable spaces - mismatched search would silently return nonsense
    scores instead of failing loudly."""
    from .db import create_pool

    pool = create_pool(database_url)
    try:
        with pool.connection() as conn:
            row = conn.execute("select value from public.index_meta where key = 'embedding_model'").fetchone()
        built_with = row[0] if row else None
        if built_with is None:
            raise RuntimeError("The knowledge base is empty - run `python scripts/sync_facts.py` first.")
        if built_with != EMBEDDING_MODEL_NAME:
            raise ValueError(
                f"The database vectors were built with embedding model {built_with!r} but EMBEDDING_MODEL is "
                f"currently {EMBEDDING_MODEL_NAME!r}. Run `python scripts/sync_facts.py` to re-embed for the "
                f"configured model, or set EMBEDDING_MODEL={built_with!r} to match the database."
            )
    except Exception:
        pool.close()
        raise
    return RetrievalIndex(pool)


def build_context_block(results: list[dict[str, Any]]) -> str | None:
    """Render retrieved facts into the text block injected into the user
    turn. Returns None if there's nothing to inject (caller should fall
    back to asking the question with no retrieved context - the "found
    nothing" signal is meaningful: it should make the model more likely to
    say it doesn't have verified information rather than guessing).

    Deliberately omits source URLs from the injected text: they cost extra
    tokens on every request and the model doesn't need them to answer.
    Callers that need the source for logging/citation should read it from
    the search() result dicts directly (each has 'source_url'), not from
    this rendered block.
    """
    if not results:
        return None
    lines = ["Relevant information:"]
    for r in results:
        lines.append(f"- {r['fact']}")
    return "\n".join(lines)
