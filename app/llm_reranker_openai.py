from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Literal, Tuple

from openai import OpenAI

Part = Literal["recruiting", "senior_engineer"]

# Hard signals (model must treat these as NOT US or NOT current)
NON_US_RE = re.compile(r"\b(uk|united kingdom|england|london|emea|india|apac|singapore|europe)\b", re.I)
EX_EMP_RE = re.compile(r"\b(ex[-\s]|former|previously|past|until\s+\d{4})\b", re.I)

# “Brand / influencer recruiter” signals (often not actually at the target company)
BRAND_RECRUITER_RE = re.compile(
    r"\b(founder|ceo|owner|influencer|content|coach|agency|staffing\s+agency|recruitment\s+agency|self[-\s]?employed)\b",
    re.I,
)

# Company mismatch cues (weak heuristic, model still decides, but we push it)
MISMATCH_HINT_RE = re.compile(r"\b(at|@)\s+([A-Z][\w&.\- ]{1,40})\b")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _coverage(input_urls: List[str], returned: List[Dict[str, Any]]) -> float:
    got = {r.get("url") for r in returned if r.get("url")}
    if not input_urls:
        return 1.0
    return len([u for u in input_urls if u in got]) / len(input_urls)


def _default_row(url: str) -> Dict[str, Any]:
    return {
        "url": url,
        "is_match": False,
        "match_type": "other",
        "current_employee": "uncertain",
        "confidence": 0.0,
        "why_matched": ["not_returned_by_llm"],
        "clean_title": "",
        "team_hint": "",
        "message_300": "",
    }


def _call_llm_json(
    client: OpenAI,
    model: str,
    system: str,
    user_obj: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Try Responses API with json_schema, then fallback to chat.completions json_object.
    Returns parsed JSON object.
    """
    # Strict schema: MUST output results array of same length and each url must be present.
    schema = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "is_match": {"type": "boolean"},
                        "match_type": {"type": "string"},
                        "current_employee": {"type": "string"},
                        "confidence": {"type": "number"},
                        "why_matched": {"type": "array", "items": {"type": "string"}},
                        "clean_title": {"type": "string"},
                        "team_hint": {"type": "string"},
                        "message_300": {"type": "string"},
                    },
                    "required": [
                        "url",
                        "is_match",
                        "match_type",
                        "current_employee",
                        "confidence",
                        "why_matched",
                        "clean_title",
                        "team_hint",
                        "message_300",
                    ],
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }

    # Attempt 1: Responses API (some SDK versions don’t support json_schema consistently)
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_obj)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "rerank_schema", "schema": schema},
            },
        )
        data = resp.output_parsed
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Attempt 2: Chat Completions JSON object
    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj)},
        ],
        response_format={"type": "json_object"},
    )
    content = chat.choices[0].message.content or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        return {"results": []}
    return parsed


def rerank_with_openai(
    company: str,
    part: Part,
    candidates: List[Dict[str, Any]],
    resume_blurb: str,
    model: str = "gpt-4o-mini",
    max_items: int = 25,
) -> List[Dict[str, Any]]:
    """
    Input candidates: list of dicts with: url, title, snippet
    Output: ALL candidates enriched with llm_* fields, sorted by llm_confidence desc.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

    client = OpenAI(api_key=api_key)

    payload = candidates[:max_items]
    input_urls = [c["url"] for c in payload]

    # IMPORTANT: we force “one row per input URL” by telling the model:
    # - results length must equal input length
    # - each url must appear exactly once and must match the input url string
    system = f"""
You classify LinkedIn search results. Use ONLY provided (url,title,snippet). No browsing.

You MUST return EXACTLY one decision for EACH input candidate, in the SAME ORDER.
- Output JSON: {{ "results": [ ... ] }}
- results length == number of input candidates
- For i-th result, url MUST equal the i-th input candidate url EXACTLY (copy/paste).
- Do NOT add new URLs. Do NOT drop any.

Target company: {company}
Target location: United States

Part definitions:
- recruiting: Talent Acquisition / Recruiter / Sourcer / University Recruiter / HR (SWE/Full Stack/AI)
- senior_engineer: Senior/Staff/Principal/Lead SWE or Engineering Manager/Tech Lead

Hard rules (must enforce):
1) If title/snippet shows NON-US (UK/United Kingdom/London/EMEA/India/APAC/etc) -> is_match=false, confidence <= 0.10
2) If title/snippet shows ex-employee (ex-, former, previously, past) -> is_match=false, current_employee="no", confidence <= 0.10
3) If company match is NOT explicit (the company name is not clearly tied to the person) -> current_employee="uncertain", confidence <= 0.70
4) If the person looks like a “brand recruiter / agency / influencer / founder / coach” and NOT clearly "{company}" -> is_match=false, confidence <= 0.30

Positive signals:
- explicit “at {company}” or “{company} | LinkedIn” style text -> current_employee="yes"
- explicit US/USA/United States or US city/state -> boosts confidence
- correct title keywords for the requested part -> boosts confidence

Confidence rubric (strict):
- 0.85–1.0 ONLY if: current_employee="yes" AND US is explicit AND role matches the requested part
- 0.60–0.84 if: likely match but missing either explicit US or explicit current employment
- <=0.30 if: weak/uncertain/company mismatch/brand recruiter vibe
- <=0.10 if: non-US OR ex-employee

Fields:
- match_type: "recruiting" | "senior_engineer" | "other"
- current_employee: "yes" | "no" | "uncertain"
- why_matched: 2–5 short reasons (keywords found)
- clean_title: cleaned role/title if possible
- team_hint: e.g., "University Recruiting", "Platform", "AI/ML", "Infrastructure"
- message_300: ~300 chars, concise, professional, no fluff, uses this blurb: "{resume_blurb}"
""".strip()

    user_obj = {"company": company, "part": part, "candidates": payload}

    def postprocess(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Ensure 1:1 mapping by position. If missing/bad, fill defaults.
        fixed: List[Dict[str, Any]] = []
        for i, url in enumerate(input_urls):
            r = results[i] if i < len(results) and isinstance(results[i], dict) else _default_row(url)
            if r.get("url") != url:
                # Force correct url if model messed up
                r = dict(r)
                r["url"] = url
                # If it mismatched URL, treat as unreliable
                r["is_match"] = False
                r["current_employee"] = "uncertain"
                r["confidence"] = 0.0
                r["why_matched"] = ["bad_url_mapping_from_llm"]
                r["message_300"] = ""
            # clamp
            try:
                r["confidence"] = _clamp01(float(r.get("confidence", 0.0)))
            except Exception:
                r["confidence"] = 0.0
            # Safety gates (deterministic) — enforce the same rules the prompt claims
            text = f"{payload[i].get('title','')} {payload[i].get('snippet','')}"
            if NON_US_RE.search(text):
                r["is_match"] = False
                r["current_employee"] = r.get("current_employee", "uncertain")
                r["confidence"] = min(r["confidence"], 0.10)
                r.setdefault("why_matched", [])
                r["why_matched"] = list(r["why_matched"]) + ["non_us_signal"]
                r["message_300"] = ""
            if EX_EMP_RE.search(text):
                r["is_match"] = False
                r["current_employee"] = "no"
                r["confidence"] = min(r["confidence"], 0.10)
                r.setdefault("why_matched", [])
                r["why_matched"] = list(r["why_matched"]) + ["ex_employee_signal"]
                r["message_300"] = ""
            if BRAND_RECRUITER_RE.search(text) and company.lower() not in text.lower():
                r["is_match"] = False
                r["confidence"] = min(r["confidence"], 0.30)
                r.setdefault("why_matched", [])
                r["why_matched"] = list(r["why_matched"]) + ["brand_or_agency_recruiter_signal"]
                r["message_300"] = ""

            fixed.append(r)
        return fixed

    # Call once
    parsed = _call_llm_json(client, model, system, user_obj)
    results = parsed.get("results", [])
    if not isinstance(results, list):
        results = []

    cov = _coverage(input_urls, results)
    if cov < 0.95:
        # Retry once with an even more explicit instruction that it must echo urls by index
        system_retry = system + "\n\nFINAL REMINDER: results[i].url MUST equal candidates[i].url. Do not omit any."
        parsed2 = _call_llm_json(client, model, system_retry, user_obj)
        results2 = parsed2.get("results", [])
        if isinstance(results2, list) and _coverage(input_urls, results2) > cov:
            results = results2

    fixed = postprocess(results)

    # Enrich originals (ALL candidates, not just payload)
    by_url = {r["url"]: r for r in fixed}
    enriched: List[Dict[str, Any]] = []
    for c in candidates:
        r = by_url.get(c["url"], _default_row(c["url"]))
        c2 = dict(c)
        c2.update(
            llm_is_match=bool(r.get("is_match", False)),
            llm_confidence=float(r.get("confidence", 0.0)),
            llm_why_matched=list(r.get("why_matched", []) or []),
            llm_message_300=str(r.get("message_300", "") or ""),
            llm_current_employee=str(r.get("current_employee", "uncertain") or "uncertain"),
            llm_match_type=str(r.get("match_type", "other") or "other"),
            llm_clean_title=str(r.get("clean_title", "") or ""),
            llm_team_hint=str(r.get("team_hint", "") or ""),
        )
        enriched.append(c2)

    enriched.sort(key=lambda x: x.get("llm_confidence", 0.0), reverse=True)
    return enriched