import os
import time
import requests
from typing import Any, Dict, List, Optional


BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchClient:
    """
    Brave Search API wrapper.
       """

    def __init__(self, api_key: str, timeout_s: int = 20, min_delay_s: float = 0.35):
        if not api_key:
            raise ValueError("BRAVE_API_KEY is missing.")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.min_delay_s = min_delay_s
        self._last_call = 0.0

    def _rate_limit(self) -> None:
        # Keep calls polite + reduce chance of HTTP 429
        elapsed = time.time() - self._last_call
        if elapsed < self.min_delay_s:
            time.sleep(self.min_delay_s - elapsed)
        self._last_call = time.time()

    def search(
        self,
        query: str,
        count: int = 10,
        country: str = "US",
        safesearch: str = "moderate",
        freshness: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of normalized results:
        [{"url":..., "title":..., "snippet":..., "source": "brave"}]
        """
        self._rate_limit()

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": max(1, min(count, 20)),
            "country": country,
            "safesearch": safesearch,
        }
        if freshness:
            # e.g. "past_week", "past_month", depends on Brave
            params["freshness"] = freshness

        resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=self.timeout_s)

        if resp.status_code != 200:
            # Print enough to debug without leaking secrets
            raise RuntimeError(f"Brave API error {resp.status_code}: {resp.text[:500]}")

        data = resp.json()
        web = data.get("web", {})
        results = web.get("results", []) or []

        out: List[Dict[str, Any]] = []
        for r in results:
            url = r.get("url") or ""
            title = r.get("title") or ""
            snippet = r.get("description") or r.get("snippet") or ""
            if url:
                out.append(
                    {
                        "url": url,
                        "title": title.strip(),
                        "snippet": snippet.strip(),
                        "source": "brave",
                    }
                )
        return out