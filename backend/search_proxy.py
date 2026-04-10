"""
VaultMind Search Proxy
Phase 2 -- privacy-safe web search through self-hosted SearXNG or Brave API.

Search chain:
  1. Privacy Firewall sanitizes the query (strips PII)
  2. Search Proxy sends ONLY the sanitized query to the search engine
  3. Results come back as raw text + URLs
  4. Context Fusion merges with local vault results

Supported backends (in priority order):
  1. SearXNG (self-hosted, no logs, no profiles -- ideal)
  2. Brave Search API (no-log agreement, privacy-focused -- good fallback)
  3. DuckDuckGo (via ddgs library -- existing VaultMind fallback)

The proxy NEVER sends the original query. Only the sanitized version.
"""

import os
import json
import time
import requests
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from privacy_firewall import sanitize_for_search, FirewallResult
from search_quality import filter_results, QualityResult


# ── Configuration ──────────────────────────────────────────────

@dataclass
class SearchConfig:
    """Search proxy configuration."""
    # SearXNG (preferred -- self-hosted)
    searxng_url: str = ""  # e.g., "http://192.168.1.100:8080"
    searxng_enabled: bool = False

    # Brave Search API (fallback)
    brave_api_key: str = ""
    brave_enabled: bool = False

    # DuckDuckGo (last resort, already in VaultMind)
    ddg_enabled: bool = True

    # General
    max_results: int = 8
    timeout: int = 10
    safe_search: bool = True


def load_search_config() -> SearchConfig:
    """Load search configuration from environment or config file."""
    config_path = os.path.expanduser("~/.vaultmind/search_config.json")

    cfg = SearchConfig()

    # Environment variables take priority
    cfg.searxng_url = os.environ.get("VAULTMIND_SEARXNG_URL", "")
    cfg.brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip().replace("\xa0", "")

    if cfg.searxng_url:
        cfg.searxng_enabled = True
    if cfg.brave_api_key:
        cfg.brave_enabled = True

    # Also check config file
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                data = json.load(f)
                if not cfg.searxng_url and data.get("searxng_url"):
                    cfg.searxng_url = data["searxng_url"]
                    cfg.searxng_enabled = True
                if not cfg.brave_api_key and data.get("brave_api_key"):
                    cfg.brave_api_key = data["brave_api_key"].strip().replace("\xa0", "")
                    cfg.brave_enabled = True
                cfg.max_results = data.get("max_results", 8)
                cfg.timeout = data.get("timeout", 10)
        except (json.JSONDecodeError, IOError):
            pass

    return cfg


# ── Search Result Type ─────────────────────────────────────────

@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    source_engine: str  # "searxng", "brave", "ddg"
    quality_tier: str = ""  # Set by quality filter
    trust_score: float = 0.0  # Set by quality filter


@dataclass
class SearchResponse:
    """Full response from the search proxy."""
    query_original: str
    query_sanitized: str
    results: list = field(default_factory=list)
    firewall_result: Optional[FirewallResult] = None
    engine_used: str = ""
    search_time_ms: int = 0
    entities_stripped: int = 0
    quality_filtered: bool = False
    error: str = ""


# ── Main Search Function ──────────────────────────────────────

def privacy_search(
    query: str,
    config: Optional[SearchConfig] = None,
    apply_quality_filter: bool = True,
) -> SearchResponse:
    """Run a privacy-safe web search.

    This is the main entry point. It:
      1. Sanitizes the query through the Privacy Firewall
      2. Sends ONLY the sanitized query to the search backend
      3. Applies quality filtering to results
      4. Returns tagged results ready for Context Fusion

    Args:
        query: The raw user query (may contain PII)
        config: Search configuration (loads from env/file if None)
        apply_quality_filter: Whether to filter low-quality results

    Returns:
        SearchResponse with sanitized query, results, and metadata
    """
    if config is None:
        config = load_search_config()

    start_time = time.time()

    # Step 1: Privacy Firewall
    sanitized_query, firewall = sanitize_for_search(query)
    print(f"[SearchProxy] Firewall: {firewall.entity_count} entities stripped")
    if firewall.was_modified:
        print(f"[SearchProxy] Sanitized: \"{sanitized_query}\"")

    response = SearchResponse(
        query_original=query,
        query_sanitized=sanitized_query,
        firewall_result=firewall,
        entities_stripped=firewall.entity_count,
    )

    # Step 2: Try search backends in priority order
    results = []
    engine = ""

    if config.searxng_enabled and config.searxng_url:
        results, engine = _search_searxng(sanitized_query, config)

    if not results and config.brave_enabled and config.brave_api_key:
        results, engine = _search_brave(sanitized_query, config)

    if not results and config.ddg_enabled:
        results, engine = _search_ddg(sanitized_query, config)

    if not results:
        response.error = "All search backends failed or returned no results"
        response.search_time_ms = int((time.time() - start_time) * 1000)
        return response

    response.results = results
    response.engine_used = engine

    # Step 3: Quality filter
    if apply_quality_filter and results:
        try:
            quality_result = filter_results(results)
            response.results = quality_result.filtered_results
            response.quality_filtered = True
            print(f"[SearchProxy] Quality filter: {quality_result.kept}/{quality_result.total} results kept")
        except Exception as e:
            print(f"[SearchProxy] Quality filter failed (non-fatal): {e}")

    response.search_time_ms = int((time.time() - start_time) * 1000)
    print(f"[SearchProxy] {len(response.results)} results via {engine} in {response.search_time_ms}ms")

    return response


# ── Search Backends ───────────────────────────────────────────

def _search_searxng(query: str, config: SearchConfig) -> tuple[list[SearchResult], str]:
    """Search via self-hosted SearXNG instance."""
    try:
        url = config.searxng_url.rstrip("/") + "/search"
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "en",
            "safesearch": 1 if config.safe_search else 0,
        }
        r = requests.get(url, params=params, timeout=config.timeout)
        r.raise_for_status()
        data = r.json()

        results = []
        for item in data.get("results", [])[:config.max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source_engine="searxng",
            ))

        return results, "searxng"

    except Exception as e:
        print(f"[SearchProxy] SearXNG failed: {e}")
        return [], ""


def _search_brave(query: str, config: SearchConfig) -> tuple[list[SearchResult], str]:
    """Search via Brave Search API (privacy-focused, no-log)."""
    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": config.brave_api_key,
        }
        params = {
            "q": query,
            "count": config.max_results,
            "safesearch": "moderate" if config.safe_search else "off",
        }
        r = requests.get(url, headers=headers, params=params, timeout=config.timeout)
        r.raise_for_status()
        data = r.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:config.max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source_engine="brave",
            ))

        return results, "brave"

    except Exception as e:
        print(f"[SearchProxy] Brave Search failed: {e}")
        return [], ""


def _search_ddg(query: str, config: SearchConfig) -> tuple[list[SearchResult], str]:
    """Search via DuckDuckGo (existing VaultMind fallback)."""
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=config.max_results):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    source_engine="ddg",
                ))

        return results, "ddg"

    except Exception as e:
        print(f"[SearchProxy] DuckDuckGo failed: {e}")
        return [], ""
