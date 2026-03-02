import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

from .normalize import Candidate


def _contains(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


# Weighted keyword packs
PART1_POS = {
    r"\btalent acquisition\b": 4.0,
    r"\btechnical recruiter\b": 4.0,
    r"\buniversity recruiter\b": 3.5,
    r"\brecruiter\b": 3.0,
    r"\bsourcer\b": 3.0,
    r"\btalent partner\b": 2.5,
    r"\bhr business partner\b|\bhrbp\b": 1.2,
    r"\bpeople ops\b|\bpeople operations\b": 1.2,
    r"\bhuman resources\b|\bhr\b": 0.8,
    r"\bhiring manager\b": 1.2,  # only mild; often noisy
}

PART2_POS = {
    r"\bstaff software engineer\b": 4.0,
    r"\bprincipal\b": 3.8,
    r"\bsenior software engineer\b": 3.5,
    r"\bengineering manager\b": 3.3,
    r"\btech lead\b|\btechnical lead\b": 3.0,
    r"\blead software engineer\b": 2.8,
    r"\bsoftware architect\b|\barchitect\b": 1.5,
    r"\bsenior\b": 1.2,   # mild because "senior recruiter" exists
    r"\bstaff\b": 1.0,    # mild alone
}

ROLE_ALIGNMENT = {
    r"\bsoftware\b": 1.2,
    r"\bengineer\b": 1.2,
    r"\bfull[- ]stack\b": 1.2,
    r"\bbackend\b": 1.0,
    r"\bfrontend\b": 1.0,
    r"\bplatform\b": 1.0,
    r"\bcloud\b": 0.8,
    r"\bai\b|\bml\b|\bmachine learning\b": 1.1,
    r"\bdata\b": 0.5,
}

NEG_SIGNALS = {
    r"\bex[- ]": -4.0,
    r"\bformer\b": -4.0,
    r"\bretired\b": -4.0,
    r"\bintern\b": -2.5,
    r"\bstudent\b": -2.0,
}


@dataclass
class ScoredCandidate:
    candidate: Candidate
    confidence: float
    why_matched: List[str]
    raw_score: float

    def to_output_dict(self) -> Dict[str, Any]:
        c = self.candidate
        return {
            "name": c.name,
            "title_snippet": c.title_snippet,
            "profile_url": c.url,
            "why_matched": self.why_matched,
            "confidence": round(self.confidence, 3),
        }


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def score_candidate(c: Candidate, location: str = "United States") -> Tuple[float, List[str]]:
    text = f"{c.raw_title}\n{c.raw_snippet}\n{c.title_snippet}".lower()

    pos_pack = PART1_POS if c.part == "part1" else PART2_POS

    score = 0.0
    why: List[str] = []

    # Title/role keywords
    for pat, w in pos_pack.items():
        if _contains(text, pat):
            score += w
            why.append(f"matched: {pat}")

    # Role alignment
    for pat, w in ROLE_ALIGNMENT.items():
        if _contains(text, pat):
            score += w
            why.append(f"aligned: {pat}")

    # Company mention heuristic
    if c.company and c.company.lower() in text:
        score += 1.4
        why.append("matched: company mention")

    # Location hint (best effort)
    if location.lower() in text or _contains(text, r"\bunited states\b|\bus\b|\busa\b"):
        score += 0.5
        why.append("matched: US location")

    # Negative signals
    for pat, w in NEG_SIGNALS.items():
        if _contains(text, pat):
            score += w
            why.append(f"penalty: {pat}")

    return score, why


def to_confidence(raw_score: float) -> float:
    """
    Map raw score -> 0..1.
    Tuned so typical good matches land ~0.7-0.9.
    """
    # shift/scale controls
    x = (raw_score - 3.0) / 2.2
    conf = _sigmoid(x)
    # clamp
    return max(0.0, min(1.0, conf))


def rank_candidates(cands: List[Candidate], location: str = "United States") -> List[ScoredCandidate]:
    scored: List[ScoredCandidate] = []
    for c in cands:
        raw, why = score_candidate(c, location=location)
        conf = to_confidence(raw)
        scored.append(ScoredCandidate(candidate=c, confidence=conf, why_matched=why, raw_score=raw))

    scored.sort(key=lambda s: (s.confidence, s.raw_score), reverse=True)
    return scored


def diversify_top(scored: List[ScoredCandidate], k: int = 6) -> List[ScoredCandidate]:
    """
    Avoid returning clones. Simple heuristic: prefer varied title keywords.
    """
    if not scored:
        return []

    chosen: List[ScoredCandidate] = []
    seen_signatures = set()

    def signature(s: ScoredCandidate) -> str:
        t = (s.candidate.raw_title or "").lower()
        # coarse title bucket
        if "technical recruiter" in t:
            return "technical recruiter"
        if "university" in t and "recruit" in t:
            return "university recruiter"
        if "talent acquisition" in t:
            return "talent acquisition"
        if "sourcer" in t:
            return "sourcer"
        if "engineering manager" in t:
            return "engineering manager"
        if "staff" in t:
            return "staff"
        if "principal" in t:
            return "principal"
        if "senior" in t:
            return "senior"
        if "recruit" in t:
            return "recruiter"
        if "engineer" in t:
            return "engineer"
        return "other"

    for s in scored:
        sig = signature(s)
        # allow at most 2 per signature bucket
        count_sig = sum(1 for x in chosen if signature(x) == sig)
        if count_sig >= 2:
            continue
        if s.confidence < 0.25:
            continue
        chosen.append(s)
        if len(chosen) >= k:
            break

    # If we didn't reach k, fill remaining ignoring diversity
    if len(chosen) < k:
        for s in scored:
            if s in chosen:
                continue
            if s.confidence < 0.25:
                continue
            chosen.append(s)
            if len(chosen) >= k:
                break

    return chosen