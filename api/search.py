"""
api/search.py — Web search grounding for opensense.

Mirrors Google Gemini's "grounding" feature: before the LLM responds, the
user's query is searched against the web and the top snippets from trusted
sources are injected into the prompt as additional context.  This lets the
model answer with up-to-date, source-backed information rather than relying
solely on its training weights or local memory.

Supported providers (configured via config.yaml  web_search.provider):
  - "duckduckgo"  (default — no API key needed)
  - "brave"       (Brave Search API — free tier available, needs api_key)
  - "serpapi"     (SerpAPI — needs api_key)

Result flow
-----------
1. Client sends a chat message.
2. If web_search.enabled is true AND (mode=="always" OR the query is
   detected as a factual/current-events question), perform a search.
3. Fetch the top ``max_results`` snippets.
4. Inject them into the prompt as a [Web search results] block so the LLM
   can cite and use them.
5. Return the source URLs alongside the response for full transparency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import httpx

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class SearchOutcome:
    results: list[SearchResult] = field(default_factory=list)
    provider_used: str = ""
    query_run: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Heuristic: should this query be grounded?
# ---------------------------------------------------------------------------

# Keywords that strongly signal a need for fresh / factual web info
_FACTUAL_SIGNALS = re.compile(
    r"\b("
    r"latest|recent|current|today|now|news|update|price|cost|"
    r"when did|when was|what is|who is|who are|how many|how much|"
    r"release|announce|launch|available|version|changelog|status|"
    r"weather|stock|rate|score|result|winner|election|crisis|war|"
    r"2024|2025|2026"
    r")\b",
    re.IGNORECASE,
)


def should_ground(query: str, mode: str) -> bool:
    """
    Decide whether to trigger a web search for this query.

    Parameters
    ----------
    query : str
        The user's message.
    mode : str
        "always"   — ground every query unconditionally
        "auto"     — ground only when the query looks factual / time-sensitive
        "never"    — grounding disabled (same as web_search.enabled = false)
    """
    if mode == "always":
        return True
    if mode == "never":
        return False
    # "auto" — heuristic based on signal keywords
    return bool(_FACTUAL_SIGNALS.search(query))


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

async def _search_duckduckgo(query: str, max_results: int) -> list[SearchResult]:
    """
    DuckDuckGo Instant Answer API + HTML scrape fallback.

    DDG's public JSON endpoint returns only the top summary; the ``lite``
    HTML endpoint is parsed for the richer organic results.
    """
    results: list[SearchResult] = []

    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
        # DDG Lite HTML — simplest reliable organic result source
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
            headers={"User-Agent": "opensense/0.1 (web-grounding; +https://github.com/opensense)"},
        )
        resp.raise_for_status()
        html = resp.text

    # Parse result blocks  — DDG HTML structure: <div class="result__body">
    # We do a lightweight regex extraction to avoid a full HTML parser dep.
    title_re = re.compile(r'class="result__a"[^>]*>(.*?)</a>', re.DOTALL)
    url_re = re.compile(r'class="result__url"[^>]*>(.*?)</span>', re.DOTALL)
    snippet_re = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)

    titles = [re.sub(r"<[^>]+>", "", t).strip() for t in title_re.findall(html)]
    urls = [u.strip() for u in url_re.findall(html)]
    snippets = [re.sub(r"<[^>]+>", "", s).strip() for s in snippet_re.findall(html)]

    for i in range(min(max_results, len(titles), len(urls), len(snippets))):
        url = urls[i]
        # DDG sometimes returns relative ad URLs — skip them
        if not url.startswith("http"):
            url = "https://" + url
        results.append(SearchResult(title=titles[i], url=url, snippet=snippets[i]))

    return results


async def _search_brave(
    query: str, max_results: int, api_key: str
) -> list[SearchResult]:
    """Brave Search API — https://api.search.brave.com/res/v1/web/search"""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results: list[SearchResult] = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            )
        )
    return results


async def _search_serpapi(
    query: str, max_results: int, api_key: str
) -> list[SearchResult]:
    """SerpAPI Google Search — https://serpapi.com/search"""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "num": max_results,
                "api_key": api_key,
                "engine": "google",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results: list[SearchResult] = []
    for item in data.get("organic_results", [])[:max_results]:
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

async def web_search(
    query: str,
    *,
    provider: Literal["duckduckgo", "brave", "serpapi"] = "duckduckgo",
    api_key: str = "",
    max_results: int = 5,
) -> SearchOutcome:
    """
    Run a web search and return structured results.

    Parameters
    ----------
    query       : The search query derived from the user's message.
    provider    : Which search backend to use.
    api_key     : API key (required for brave / serpapi, ignored for duckduckgo).
    max_results : Maximum number of results to retrieve.
    """
    outcome = SearchOutcome(provider_used=provider, query_run=query)
    try:
        if provider == "brave":
            outcome.results = await _search_brave(query, max_results, api_key)
        elif provider == "serpapi":
            outcome.results = await _search_serpapi(query, max_results, api_key)
        else:
            outcome.results = await _search_duckduckgo(query, max_results)
    except Exception as exc:  # noqa: BLE001
        outcome.error = str(exc)
    return outcome


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_search_context_block(outcome: SearchOutcome) -> str:
    """
    Format search results into a prompt block the LLM can read.

    Example output injected before the user message:

        [Web search results — DuckDuckGo | query: "latest Python version"]
        1. Python 3.13 released — python.org
           Python 3.13 is the latest stable release...
           Source: https://www.python.org/downloads/
        ...
    """
    if not outcome.results:
        return ""

    lines = [
        f"[Web search results — {outcome.provider_used} | query: \"{outcome.query_run}\"]"
    ]
    for idx, r in enumerate(outcome.results, 1):
        lines.append(f"{idx}. {r.title} — {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
    return "\n".join(lines)
