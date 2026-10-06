import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "crawl_technyx.py"
SPEC = importlib.util.spec_from_file_location("crawl_technyx", SCRIPT)
assert SPEC and SPEC.loader
crawl_technyx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(crawl_technyx)


def test_parser_retains_content_and_section_but_ignores_short_fragments():
    parser = crawl_technyx.VisibleTextParser()
    parser.feed(
        "<title>Page</title><h1>Heading</h1>"
        "<p>This is a meaningful sentence that belongs in the knowledge base.</p><p>Skip</p>"
    )

    assert parser.title == "Page"
    assert parser.items == [
        ("Heading", "Heading"),
        ("Heading", "This is a meaningful sentence that belongs in the knowledge base."),
    ]


def test_fact_type_identifies_case_studies():
    assert crawl_technyx.fact_type("https://technyxsystems.com/case-studies/practical") == "case_study"


FLAT_URLSET = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page1</loc></url>
  <url><loc>https://example.com/cookie-policy</loc></url>
</urlset>"""

SITEMAP_INDEX = """<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap-a.xml</loc></sitemap>
  <sitemap><loc>https://example.com/sitemap-b.xml</loc></sitemap>
</sitemapindex>"""

URLSET_A = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/a1</loc></url>
</urlset>"""

URLSET_B = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/b1</loc></url>
</urlset>"""


class TestSitemapUrls:
    def test_flat_urlset_excludes_legal_pages(self, monkeypatch):
        monkeypatch.setattr(crawl_technyx, "fetch", lambda url: FLAT_URLSET)
        assert crawl_technyx.sitemap_urls("https://example.com/sitemap.xml") == ["https://example.com/page1"]

    def test_resolves_a_sitemap_index_into_page_urls(self, monkeypatch):
        pages = {
            "https://example.com/sitemap.xml": SITEMAP_INDEX,
            "https://example.com/sitemap-a.xml": URLSET_A,
            "https://example.com/sitemap-b.xml": URLSET_B,
        }
        monkeypatch.setattr(crawl_technyx, "fetch", lambda url: pages[url])
        urls = crawl_technyx.sitemap_urls("https://example.com/sitemap.xml")
        assert urls == ["https://example.com/a1", "https://example.com/b1"]


class TestLoadRobots:
    def test_blocks_disallowed_paths_and_allows_the_rest(self, monkeypatch):
        monkeypatch.setattr(crawl_technyx, "fetch", lambda url: "User-agent: *\nDisallow: /private/\n")
        robots = crawl_technyx.load_robots("https://example.com/sitemap.xml")
        assert robots.can_fetch(crawl_technyx.USER_AGENT, "https://example.com/private/page") is False
        assert robots.can_fetch(crawl_technyx.USER_AGENT, "https://example.com/public/page") is True

    def test_missing_robots_txt_allows_everything(self, monkeypatch):
        def raise_unreachable(url):
            raise crawl_technyx.URLError("not found")

        monkeypatch.setattr(crawl_technyx, "fetch", raise_unreachable)
        robots = crawl_technyx.load_robots("https://example.com/sitemap.xml")
        assert robots.can_fetch(crawl_technyx.USER_AGENT, "https://example.com/anything") is True


_PAGE_HTML = "<title>T</title><h1>H</h1><p>A meaningful sentence that is long enough to count as a fact.</p>"


class TestCrawlResilience:
    def test_skips_a_failing_page_and_keeps_the_rest(self, monkeypatch):
        sitemap_xml = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/good1</loc></url>
  <url><loc>https://example.com/bad</loc></url>
  <url><loc>https://example.com/good2</loc></url>
</urlset>"""

        def fake_fetch(url):
            if url == "https://example.com/sitemap.xml":
                return sitemap_xml
            if url == "https://example.com/robots.txt":
                raise crawl_technyx.URLError("no robots.txt")
            if url == "https://example.com/bad":
                raise crawl_technyx.URLError("network blip")
            if url == "https://example.com/good1":
                return "<title>T</title><h1>H</h1><p>A meaningful sentence that is long enough to count.</p>"
            return "<title>T</title><h1>H</h1><p>A different meaningful sentence that is long enough too.</p>"

        monkeypatch.setattr(crawl_technyx, "fetch", fake_fetch)
        records = crawl_technyx.crawl("https://example.com/sitemap.xml", delay_seconds=0)
        assert {r["source_url"] for r in records} == {"https://example.com/good1", "https://example.com/good2"}

    def test_skips_robots_disallowed_pages(self, monkeypatch):
        sitemap_xml = """<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/allowed</loc></url>
  <url><loc>https://example.com/blocked</loc></url>
</urlset>"""

        def fake_fetch(url):
            if url == "https://example.com/sitemap.xml":
                return sitemap_xml
            if url == "https://example.com/robots.txt":
                return "User-agent: *\nDisallow: /blocked\n"
            return _PAGE_HTML

        monkeypatch.setattr(crawl_technyx, "fetch", fake_fetch)
        records = crawl_technyx.crawl("https://example.com/sitemap.xml", delay_seconds=0)
        assert {r["source_url"] for r in records} == {"https://example.com/allowed"}


class TestSplitLongText:
    def test_short_text_returned_unchanged(self):
        assert crawl_technyx._split_long_text("Short text.", max_chars=800) == ["Short text."]

    def test_splits_at_sentence_boundaries_under_budget(self):
        text = "Sentence one is here. Sentence two is here. Sentence three is here."
        chunks = crawl_technyx._split_long_text(text, max_chars=40)
        assert len(chunks) > 1
        assert all(len(c) <= 40 for c in chunks)
        assert " ".join(chunks) == text

    def test_hard_splits_a_single_sentence_longer_than_budget(self):
        text = ("word " * 50).strip() + "."
        chunks = crawl_technyx._split_long_text(text, max_chars=20)
        assert len(chunks) > 1
        assert all(len(c) <= 20 for c in chunks)
        assert sum(len(c.split()) for c in chunks) == len(text.split())
