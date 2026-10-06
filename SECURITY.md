# Security review

A review done as part of this project's production-readiness work. Findings
below are what was actually checked and found - not a certification that
nothing else exists. Re-run the checks (they're all cheap) whenever a
dependency changes or before an actual deployment.

## Secrets handling - clean

- `.env` (holds `HF_TOKEN`) is gitignored and was never committed - verified
  with `git log` and `git grep` across the full history, not just the
  current tree.
- `utils.get_hf_token()` never logs, prints, or returns the token in a
  truncated/partial form - checked its full implementation.
- `git grep` across all tracked `*.py` files for common secret patterns
  (`api_key`, `secret`, `password`, inline `token = "..."`) found nothing
  beyond the legitimate `HF_TOKEN`-passing code.

## Dependency vulnerabilities

`pip-audit` against `requirements.txt`: **no known vulnerabilities.**

`pip-audit` against the fully resolved environment (including the optional
`llama-cpp-python` GGUF path): one finding, **not a practical risk for how
this project uses it**:

- `diskcache 5.6.3` (PYSEC-2026-2447, unsafe pickle deserialization,
  CVSS 5.2/Medium) - a transitive dependency of `llama-cpp-python`, not of
  anything in this project's own `requirements.txt`. Traced through
  `llama_cpp`'s source: the vulnerable code path is `LlamaDiskCache`, an
  **opt-in** response-cache class that has to be explicitly constructed and
  passed to `Llama.set_cache()`. `gguf_llm.py` never calls `set_cache()` -
  the vulnerable feature is present in the dependency tree but dormant.
  **Do not add response caching to `gguf_llm.py` via `LlamaDiskCache`
  without addressing this first** (or wait for an upstream fix and pin to
  it).

## Input validation - a real gap, fixed

Found: nothing bounded `user_input` length before it reached
tokenization/embedding/generation - a pathologically long input cost real
CPU time proportional to its length at every layer, with no cap. Fixed:
`guardrails.is_input_too_long()` (checked first in `pipeline.answer()`,
before retrieval or generation ever run) rejects anything over 4000
characters - generous for any real question (every question in
`evaluation/test_cases.jsonl` is under 150 characters).

## Public widget and visitor limits

The embeddable widget deliberately ships **no API secret**. `API_KEY` remains
for trusted service-to-service calls to `/v1/chat`; browser visitors use
`/v1/chat/stream` and are rate-limited in-process by source IP (default
10/60s). This reduces a single-browser burst but is not authentication, does
not stop a distributed attack or NAT-shared users, and is per process. Use a
WAF/shared limiter before multi-replica or hostile public deployment.

`X-Forwarded-For` is ignored unless `TRUST_PROXY=1`. Only set that behind a
proxy that overwrites the header; otherwise clients can choose their own limit
key. `CORS_ORIGINS` is an explicit allowlist and defaults empty, so configure
`https://technyxsystems.com` before embedding the widget.

Conversation JSONL logs contain visitor messages and may therefore contain
PII. They are gitignored, retained locally by deployment policy, and need
restricted filesystem access. In-process conversation history is capped at 40
messages, 1,000 sessions, and one hour by default; Redis is the appropriate
replacement if restart survival or cross-replica sharing is required.

## Prompt injection - partially mitigated, not eliminated (disclosed)

No prompting-only defense against injection is complete for any LLM, and
this project doesn't claim otherwise. What actually helps here:

- `fact_check.py` checks the *generated answer* against retrieved context
  regardless of how the model was steered into producing it - an injection
  attempt that tries to make the model assert a fabricated named entity
  (a fake partner, a fake platform) gets caught the same way an ordinary
  hallucination does, because the check doesn't care why the claim appeared.
- `guardrails.py`'s three query-shape guardrails are structural (based on
  retrieval results and `user_input` keyword matching), not "trust the
  model's own judgment about whether to refuse" - so they can't be talked
  out of firing by injected text in the question.

What's **not** covered: an injection attempt aimed at tone/persona (getting
the model to be rude, off-brand, or to claim it has no restrictions) rather
than at asserting a fabricated named fact. Neither `guardrails.py` nor
`fact_check.py` checks for that, and nothing else in this project does
either. A production deployment that's actually exposed to adversarial
users should treat this as an open item, not a solved one.

## Testimonial content (real names, real quotes)

The chatbot surfaces real people's names, roles, companies, and quotes
(`data/knowledge/facts.jsonl`'s `client`-type facts). This is **not new
exposure**: every one of these is a testimonial Technyx already published
on its own public marketing site (`technyxsystems.com`), which is where
this corpus was crawled from - the chatbot echoes already-public content,
it doesn't surface anything that wasn't already public. Re-evaluate this
if the corpus is ever extended with non-public data.

## Error handling / information disclosure - deferred to the serving layer

CLI scripts (`chat.py`, `evaluate.py`, `chat_gguf.py`) print raw exceptions
to the terminal, which is fine for a local operator running their own
tooling. **This must not carry over to any API/service wrapper** - a caller
should get a generic error response, with the real exception (stack trace,
internal paths, dependency versions) logged server-side only, never
returned in the response body. Enforced in the serving layer - see its own
documentation for what it actually does here.

## Concurrency safety - a real, load-tested crash, fixed

Found by load-testing `api.py`, not by inspection - and worth recording
here because it's exactly the kind of bug static review wouldn't catch:
`api.py`'s `/v1/chat` handler was made non-blocking (dispatched to a
thread via `asyncio.to_thread`) so the event loop wouldn't stall for an
entire generation - correct on its own, and verified directly (a
`/health` check kept responding in 1-4ms throughout a 13-second
generation). But this also meant multiple threads could call the single
shared model instance concurrently. Load-tested with 4 concurrent
`/v1/chat` requests: **every one hung and timed out**, and the server log
showed `GGML_ASSERT(i1 >= 0 && i1 < ne1) failed` - llama.cpp's internal
KV-cache indexing, corrupted by concurrent access to the same context.
The GGUF backend's `Llama` instance is not thread-safe for concurrent
generation calls, and this project makes no assumption that the plain
PyTorch backend is either.

Fixed with `app.state.answer_lock` (an `asyncio.Lock`), acquired only
around the actual model-touching call - `/health` and rate-limit
rejections never wait on it, only real generation does. Re-tested the
same 4-concurrent-request scenario after the fix: all four completed
successfully (serialized, ~7-8s apart, no crash). A permanent regression
test (`test_concurrent_requests_do_not_crash_or_corrupt_state`) now
covers this in `tests/test_api.py`.

**Capacity implication, not a bug**: because generation is serialized
per process, latency grows linearly with concurrent request volume - one
process handles one generation at a time, full stop. This is the correct
tradeoff for a CPU-bound model that isn't safe to call concurrently, not
something to "fix" with more threads. Scale by running multiple process
replicas behind a load balancer (see the Dockerfile/deployment docs),
each with its own model instance and its own lock, not by increasing
concurrency within one process.

## Not reviewed (out of scope for this pass)

- Third-party crawled content isn't validated for embedded prompt-injection
  text (e.g., a fact scraped from a webpage that itself contains
  instruction-like text). The corpus is small and was reviewed once during
  `_looks_like_nav_fragment()`'s development, but not specifically for this.
- No penetration test or adversarial red-teaming was performed - this is a
  code/dependency/design review, not an attack simulation.
