import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urlunparse


LINKEDIN_PROFILE_PREFIXES = (
    "https://www.linkedin.com/in/",
    "http://www.linkedin.com/in/",
    "https://linkedin.com/in/",
    "http://linkedin.com/in/",
)

REJECT_SUBPATHS = (
    "/company/",
    "/jobs/",
    "/posts/",
    "/feed/",
    "/groups/",
    "/learning/",
    "/pub/",
)


@dataclass
class Candidate:
    company: str
    part: str  # "part1" or "part2"
    url: str
    name: str
    title_snippet: str
    raw_title: str
    raw_snippet: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def canonicalize_url(url: str) -> str:
    """
    Strip query params/fragments to improve dedup.
    """
    try:
        p = urlparse(url)
        clean = p._replace(query="", fragment="")
        return urlunparse(clean)
    except Exception:
        return url


def is_valid_profile_url(url: str) -> bool:
    u = url.strip()
    if not u.startswith(LINKEDIN_PROFILE_PREFIXES):
        return False
    for bad in REJECT_SUBPATHS:
        if bad in u:
            return False
    return True


def extract_name_from_title(title: str) -> str:
    """
    Typical: "Jane Doe - Technical Recruiter - Company | LinkedIn"
    We'll take the left-most chunk before " - " or " | ".
    """
    t = (title or "").strip()
    if not t:
        return "Unknown"

    # Split on common separators
    first = re.split(r"\s-\s|\s\|\s", t, maxsplit=1)[0].strip()
    # Avoid garbage like "LinkedIn"
    if len(first) < 2 or first.lower() in {"linkedin", "log in", "sign in"}:
        return "Unknown"
    return first


def normalize_results_to_candidates(
    results: List[Dict[str, Any]],
    company: str,
    part: str,
) -> List[Candidate]:
    out: List[Candidate] = []
    for r in results:
        url = canonicalize_url(r.get("url", ""))
        title = (r.get("title") or "").strip()
        snippet = (r.get("snippet") or "").strip()

        if not url or not is_valid_profile_url(url):
            continue

        name = extract_name_from_title(title)
        title_snippet = " ".join([x for x in [title, snippet] if x])[:600]

        out.append(
            Candidate(
                company=company,
                part=part,
                url=url,
                name=name,
                title_snippet=title_snippet,
                raw_title=title,
                raw_snippet=snippet,
                source=r.get("source", "unknown"),
            )
        )
    return out


def dedup_candidates(cands: List[Candidate]) -> List[Candidate]:
    seen = set()
    out: List[Candidate] = []
    for c in cands:
        key = c.url.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out