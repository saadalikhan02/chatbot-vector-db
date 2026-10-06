#!/usr/bin/env python3
"""Refresh the public Technyx knowledge base from its declared sitemap.

Uses only the standard library: a transparent crawler is easier to audit
than a scraping service. It deliberately excludes legal pages and
navigation/footer text, retaining headings, paragraphs, and list items
with their page/section provenance. Run build_index.py afterwards.

Hardened for crawling a much larger site than today's 32 pages (sitemap
index files, robots.txt compliance, one bad page not aborting the whole
run, long paragraphs split into embedding-sized chunks instead of silently
truncated later) - see README's "Updating the dataset" for the operational
story.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import xml.etree.ElementTree as element_tree
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

SITEMAP_URL = "https://technyxsystems.com/sitemap.xml"
EXCLUDED_PATH_PARTS = ("cookie-policy", "privacy-policy", "terms-conditions")
TEXT_TAGS = {"h1", "h2", "h3", "p", "li"}
USER_AGENT = "TechnyxRAGKBUpdater/1.0 (+https://technyxsystems.com/)"

# Character budget for a single extracted text block before it's split into
# multiple sub-chunks (see _split_long_text). Chosen to comfortably clear
# the embedding tokenizer's max_length (512 tokens, ~4 chars/token in
# English - see retrieval.py's embed_texts) with headroom for the page
# title/section prefix _embedding_text() adds on top. A no-op today: the
# largest fact in the current 32-page corpus is 557 chars.
MAX_FACT_CHARS = 800
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_long_text(text: str, max_chars: int = MAX_FACT_CHARS) -> list[str]:
    """Split a text block into sentence-boundary chunks no longer than
    max_chars each. A no-op (returns [text]) when already short enough -
    the common case today, so existing short facts are unaffected. Falls
    back to a hard word-boundary split for a single sentence that alone
    exceeds max_chars (rare, but must never lose text or emit an
    oversized chunk)."""
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        # A single sentence longer than the budget: hard-split on word
        # boundaries instead.
        piece = ""
        for word in sentence.split(" "):
            candidate_piece = f"{piece} {word}".strip() if piece else word
            if len(candidate_piece) <= max_chars:
                piece = candidate_piece
            else:
                if piece:
                    chunks.append(piece)
                piece = word
        current = piece
    if current:
        chunks.append(current)
    return chunks


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._tag: str | None = None
        self._parts: list[str] = []
        self.title = ""
        self.section = ""
        self.items: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in TEXT_TAGS or tag == "title":
            self._tag = tag
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._tag:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self._tag:
            return
        text = normalize("".join(self._parts))
        self._tag = None
        if not text:
            return
        if tag == "title":
            self.title = text
        elif tag in {"h1", "h2", "h3"}:
            self.section = text
            self.items.append((text, text))
        elif len(text) >= 30:
            section = self.section or "Overview"
            for chunk in _split_long_text(text):
                self.items.append((section, chunk))


def fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def load_robots(reference_url: str) -> RobotFileParser:
    """Fetch and parse robots.txt for reference_url's host via our own
    fetch() (so it goes through the same User-Agent and is mockable in
    tests, unlike RobotFileParser's built-in .read()). A missing or
    unreachable robots.txt means "allow everything" - the standard
    convention - not a hard failure."""
    parsed = urlparse(reference_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    robots = RobotFileParser()
    try:
        text = fetch(robots_url)
    except (URLError, OSError):
        robots.parse([])
        return robots
    robots.parse(text.splitlines())
    return robots


def sitemap_urls(sitemap_url: str) -> list[str]:
    """Resolve a sitemap URL into page URLs. Handles both a flat
    <urlset> (what the site uses today) and a <sitemapindex> of child
    <sitemap><loc> files (what a much larger site commonly splits into) -
    recursing one level so either shape resolves to real page URLs."""
    root = element_tree.fromstring(fetch(sitemap_url))
    root_tag = root.tag.rsplit("}", 1)[-1]
    if root_tag == "sitemapindex":
        urls: list[str] = []
        for child in root.findall("{*}sitemap/{*}loc"):
            if child.text:
                urls.extend(sitemap_urls(child.text))
        return urls
    urls = [element.text for element in root.findall("{*}url/{*}loc") if element.text]
    return [url for url in urls if not any(part in url for part in EXCLUDED_PATH_PARTS)]


def fact_type(url: str) -> str:
    if "/case-studies/" in url:
        return "case_study"
    if "/services" in url:
        return "service"
    if "/platforms" in url:
        return "platform"
    if "/industries" in url:
        return "industry"
    if url.endswith("/company"):
        return "company"
    if url.endswith("/contact"):
        return "contact"
    if url.endswith("/faq"):
        return "faq"
    return "other"


def crawl(sitemap_url: str, delay_seconds: float) -> list[dict[str, str]]:
    """Crawl every sitemap-listed page and extract facts.

    One bad page (network error, unparseable response, robots.txt
    disallow) is skipped and logged rather than aborting the whole run -
    necessary once the sitemap lists hundreds of pages instead of a
    handful, since a single transient failure becomes near-certain over a
    long sequential crawl."""
    crawled_at = datetime.now(UTC).isoformat()
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    robots = load_robots(sitemap_url)
    skipped: list[str] = []
    for url in sitemap_urls(sitemap_url):
        if not robots.can_fetch(USER_AGENT, url):
            skipped.append(f"{url} (robots.txt disallow)")
            continue
        try:
            parser = VisibleTextParser()
            parser.feed(fetch(url))
        except (URLError, OSError) as exc:
            skipped.append(f"{url} ({exc})")
            continue
        for section, text in parser.items:
            if text in seen:
                continue
            seen.add(text)
            digest = hashlib.sha256(f"{url}\n{text}".encode()).hexdigest()[:16]
            records.append(
                {
                    "fact_id": f"fact_{digest}",
                    "fact": text,
                    "fact_type": fact_type(url),
                    "source_url": url,
                    "source_page_title": parser.title or url,
                    "source_section": section,
                    "source_excerpt": text,
                    "confidence": "explicit",
                    "crawled_at": crawled_at,
                }
            )
        time.sleep(delay_seconds)
    if skipped:
        print(f"Skipped {len(skipped)} URL(s):")
        for entry in skipped:
            print(f"  - {entry}")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sitemap", default=SITEMAP_URL)
    parser.add_argument("--output", type=Path, default=Path("data/knowledge/facts.jsonl"))
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    args = parser.parse_args()
    facts = crawl(args.sitemap, args.delay_seconds)
    if not facts:
        raise RuntimeError("Crawler extracted no facts; leaving the existing knowledge base unchanged.")
    args.output.write_text("".join(json.dumps(fact, ensure_ascii=False) + "\n" for fact in facts), encoding="utf-8")
    print(f"Wrote {len(facts)} explicit facts from {args.sitemap} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
