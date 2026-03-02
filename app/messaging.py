import re
from typing import Optional

from .scoring import ScoredCandidate


def _first_name(full: str) -> str:
    if not full or full == "Unknown":
        return "there"
    return full.split()[0]


def infer_role_focus(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(ai|ml|machine learning)\b", t):
        return "AI/ML engineering"
    if re.search(r"\bfull[- ]stack\b", t):
        return "full stack"
    if re.search(r"\bbackend\b", t):
        return "backend"
    if re.search(r"\bfrontend\b", t):
        return "frontend"
    if re.search(r"\bplatform\b", t):
        return "platform"
    return "software engineering"


def shorten_300(s: str, limit: int = 320) -> str:
    s = " ".join(s.split())
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def draft_message(sc: ScoredCandidate, resume_blurb: str, role_focus: Optional[str] = None) -> str:
    c = sc.candidate
    first = _first_name(c.name)
    title_hint = c.raw_title or "your role"
    focus = role_focus.strip() if role_focus else infer_role_focus(c.title_snippet)

    if c.part == "part1":
        msg = (
            f"Hi {first} — saw you’re {title_hint}. {resume_blurb} "
            f"Are you the right contact for {focus} roles in the US? If not, who’s best to reach out to?"
        )
        return shorten_300(msg)

    # part2
    msg = (
        f"Hi {first} — noticed you’re {title_hint}. {resume_blurb} "
        f"I’m exploring {focus} roles in the US and would value 1–2 pointers on the team/hiring process. Open to a quick chat?"
    )
    return shorten_300(msg)