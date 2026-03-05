import json
import csv
import re
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from .search_provider_brave import BraveSearchClient
from .query_builder import build_queries
from .normalize import normalize_results_to_candidates, dedup_candidates
from .scoring import rank_candidates, diversify_top
from .store import Store
from .llm_reranker_openai import rerank_with_openai
from .llm_message_personalizer import personalize_message_with_llm


NON_US_RE = re.compile(
    r"\b(uk|united kingdom|england|london|great britain|britain|emea|india|apac|singapore|europe)\b",
    re.IGNORECASE,
)


def _guess_name_from_title(title: str) -> str:
    if not title:
        return ""
    t = re.sub(r"\s*\|\s*LinkedIn.*$", "", title).strip()
    parts = [p.strip() for p in t.split(" - ") if p.strip()]
    return parts[0] if parts else ""

def _first_name(name: str) -> str:
    """
    Extract first name only.
    Examples:
    'John Smith' -> 'John'
    'Anjali Srivastava' -> 'Anjali'
    'John F.' -> 'John'
    """
    if not name:
        return "there"

    name = name.strip()
    parts = name.split()

    if len(parts) == 0:
        return "there"

    return parts[0].capitalize()

def _compact_title_snippet(title: str, snippet: str, max_len: int = 600) -> str:
    s = (title or "") + " " + (snippet or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def _pad_bucket(items: List[Dict[str, Any]], k: int, bucket_name: str) -> List[Dict[str, Any]]:
    """
    Ensure exactly k items exist for consistent A/B output.
    """
    out = list(items)
    while len(out) < k:
        out.append(
            {
                "name": "",
                "title_snippet": "",
                "profile_url": "",
                "why_matched": [f"no_result_for_{bucket_name}"],
                "confidence": 0.0,
                "message_300": "",
                "source": "pad",
                "llm_is_match": False,
            }
        )
    return out[:k]


def run_for_company(
    company: str,
    client: BraveSearchClient,
    store: Store,
    location: str,
    resume_blurb: str,
    role_focus: Optional[str] = None,
    per_query_count: int = 10,
    seed_n: int = 25,
    rules_k: int = 2,
    llm_k: int = 2,
) -> Dict[str, Any]:
    qp = build_queries(company=company, location=location, role_focus=role_focus)
    # TODO: replace these later with your real details
    MY_NAME = "ABC"
    MY_EMAIL = "abc@gmail.com"
    ME_BLURB = "ABC, a software developer with X years of experience at XYZ and currently pursuing my MS in Computer Science at Rutgers University"
    ME_BLURB_LONG = "I’m pursuing my MS in Computer Science at Rutgers University and previously worked for X years at XYZ, building full-stack systems and applied AI solutions, including recent work evaluating LLM reasoning frameworks."
        # Search
    part1_results: List[Dict[str, Any]] = []
    for q in qp.part1_queries:
        part1_results.extend(client.search(q, count=per_query_count, country="US"))

    part2_results: List[Dict[str, Any]] = []
    for q in qp.part2_queries:
        part2_results.extend(client.search(q, count=per_query_count, country="US"))

    # Normalize + dedup
    part1_cands = dedup_candidates(normalize_results_to_candidates(part1_results, company=company, part="part1"))
    part2_cands = dedup_candidates(normalize_results_to_candidates(part2_results, company=company, part="part2"))

    # Rules rank
    ranked1 = rank_candidates(part1_cands, location=location)
    ranked2 = rank_candidates(part2_cands, location=location)

    old1 = diversify_top(ranked1, k=rules_k)
    old2 = diversify_top(ranked2, k=rules_k)

    # Seed for LLM
    seed1 = [{"url": sc.candidate.url, "title": sc.candidate.raw_title, "snippet": sc.candidate.raw_snippet} for sc in ranked1[:seed_n]]
    seed2 = [{"url": sc.candidate.url, "title": sc.candidate.raw_title, "snippet": sc.candidate.raw_snippet} for sc in ranked2[:seed_n]]

    # LLM rerank (returns ALL enriched, sorted by confidence; not filtered)
    llm1_all = rerank_with_openai(company, "recruiting", seed1, resume_blurb)
    llm2_all = rerank_with_openai(company, "senior_engineer", seed2, resume_blurb)

    # Hard non-US block based on title/snippet text
    def is_non_us(item: Dict[str, Any]) -> bool:
        text = f"{item.get('title','')} {item.get('snippet','')}"
        return bool(NON_US_RE.search(text))

    llm1_all = [x for x in llm1_all if not is_non_us(x)]
    llm2_all = [x for x in llm2_all if not is_non_us(x)]

    # Take top 2 LLM opinions regardless of is_match, so bucket always shows something useful
    llm1 = llm1_all[:llm_k]
    llm2 = llm2_all[:llm_k]

    store.upsert_company(company)

    part1_rules_out: List[Dict[str, Any]] = []
    for sc in old1:
        cand = sc.candidate
        part1_rules_out.append(
            {
                "name": _guess_name_from_title(cand.raw_title),
                "title_snippet": _compact_title_snippet(cand.raw_title, cand.raw_snippet),
                "profile_url": cand.url,
                "why_matched": sc.why_matched,
                "confidence": round(float(sc.confidence), 3),
                "message_300": personalize_message_with_llm(
                "recruiter",
                name=_first_name(_guess_name_from_title(cand.raw_title)),
                company=company,
                me_blurb=ME_BLURB,
                me_blurb_long=ME_BLURB_LONG,
                my_name=MY_NAME,
                my_email=MY_EMAIL,
            ),
                "source": "rules",
                "llm_is_match": None,
            }
        )

    part2_rules_out: List[Dict[str, Any]] = []
    for sc in old2:
        cand = sc.candidate
        part2_rules_out.append(
            {
                "name": _guess_name_from_title(cand.raw_title),
                "title_snippet": _compact_title_snippet(cand.raw_title, cand.raw_snippet),
                "profile_url": cand.url,
                "why_matched": sc.why_matched,
                "confidence": round(float(sc.confidence), 3),
                "message_300": personalize_message_with_llm(
                "senior",
                name=_first_name(_guess_name_from_title(cand.raw_title)),
                company=company,
                me_blurb=ME_BLURB,
                me_blurb_long=ME_BLURB_LONG,
                my_name=MY_NAME,
                my_email=MY_EMAIL,
            ),
                "source": "rules",
                "llm_is_match": None,
            }
        )

    part1_llm_out: List[Dict[str, Any]] = []
    for item in llm1:
        part1_llm_out.append(
            {
                "name": _guess_name_from_title(item.get("title", "")),
                "title_snippet": _compact_title_snippet(item.get("title", ""), item.get("snippet", "")),
                "profile_url": item["url"],
                "why_matched": item.get("llm_why_matched", []),
                "confidence": round(float(item.get("llm_confidence", 0.0)), 3),
                "message_300": personalize_message_with_llm(
                "recruiter",
                name=_first_name(_guess_name_from_title(item.get("title", ""))),
                company=company,
                me_blurb=ME_BLURB,
                me_blurb_long=ME_BLURB_LONG,
                my_name=MY_NAME,
                my_email=MY_EMAIL,
            ),
                "source": "llm",
                "llm_is_match": bool(item.get("llm_is_match", False)),
            }
        )

    part2_llm_out: List[Dict[str, Any]] = []
    for item in llm2:
        part2_llm_out.append(
            {
                "name": _guess_name_from_title(item.get("title", "")),
                "title_snippet": _compact_title_snippet(item.get("title", ""), item.get("snippet", "")),
                "profile_url": item["url"],
                "why_matched": item.get("llm_why_matched", []),
                "confidence": round(float(item.get("llm_confidence", 0.0)), 3),
                "message_300": personalize_message_with_llm(
                "senior",
                name=_first_name(_guess_name_from_title(item.get("title", ""))),
                company=company,
                me_blurb=ME_BLURB,
                me_blurb_long=ME_BLURB_LONG,
                my_name=MY_NAME,
                my_email=MY_EMAIL,
            ),
                "source": "llm",
                "llm_is_match": bool(item.get("llm_is_match", False)),
            }
        )

    # PAD to guarantee 2 rows each in console/JSON/CSV
    part1_rules_out = _pad_bucket(part1_rules_out, rules_k, "part1_recruiting_rules")
    part1_llm_out = _pad_bucket(part1_llm_out, llm_k, "part1_recruiting_llm")
    part2_rules_out = _pad_bucket(part2_rules_out, rules_k, "part2_senior_engineers_rules")
    part2_llm_out = _pad_bucket(part2_llm_out, llm_k, "part2_senior_engineers_llm")

    return {
        "company": company,
        "location": location,
        "part1_recruiting_rules": part1_rules_out,
        "part1_recruiting_llm": part1_llm_out,
        "part2_senior_engineers_rules": part2_rules_out,
        "part2_senior_engineers_llm": part2_llm_out,
    }


def print_company_report(console: Console, report: Dict[str, Any]) -> None:
    console.rule(f"[bold]{report['company']}[/bold] — {report['location']}")

    def _print_table(title: str, items: List[Dict[str, Any]]) -> None:
        t = Table(title=title)
        t.add_column("#", justify="right", width=3)
        t.add_column("Name", overflow="fold")
        t.add_column("Confidence", justify="right", width=10)
        t.add_column("Profile", overflow="fold")
        t.add_column("Why", overflow="fold")

        for i, p in enumerate(items, start=1):
            t.add_row(
                str(i),
                p.get("name", ""),
                str(p.get("confidence", "")),
                p.get("profile_url", ""),
                ", ".join((p.get("why_matched") or [])[:3]),
            )
        console.print(t)

    _print_table("Part 1 — Recruiting (Rules-only) [2]", report["part1_recruiting_rules"])
    _print_table("Part 1 — Recruiting (LLM-on-top) [2]", report["part1_recruiting_llm"])
    _print_table("Part 2 — Senior Engineers (Rules-only) [2]", report["part2_senior_engineers_rules"])
    _print_table("Part 2 — Senior Engineers (LLM-on-top) [2]", report["part2_senior_engineers_llm"])

    console.print("\n[bold]Messages (copy/paste)[/bold]")
    for section_name, items in [
        ("Part 1 (LLM)", report["part1_recruiting_llm"]),
        ("Part 2 (LLM)", report["part2_senior_engineers_llm"]),
    ]:
        console.print(f"\n[underline]{section_name}[/underline]")
        for i, p in enumerate(items, start=1):
            console.print(f"{i}. {p.get('name','')} — {p.get('profile_url','')}")
            msg = (p.get("message_300") or "").strip()
            if msg:
                console.print(f"   {msg}")


def export_json(path: str, reports: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def export_csv(path: str, reports: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []

    def add_rows(rep: Dict[str, Any], bucket: str, items: List[Dict[str, Any]]) -> None:
        for p in items:
            rows.append(
                {
                    "company": rep["company"],
                    "location": rep["location"],
                    "bucket": bucket,
                    "source": p.get("source", ""),
                    "name": p.get("name", ""),
                    "confidence": p.get("confidence", ""),
                    "profile_url": p.get("profile_url", ""),
                    "title_snippet": p.get("title_snippet", ""),
                    "why_matched": " | ".join(p.get("why_matched", []) or []),
                    "llm_is_match": p.get("llm_is_match", ""),
                    "message_300": p.get("message_300", ""),
                }
            )

    for rep in reports:
        add_rows(rep, "part1_recruiting_rules", rep["part1_recruiting_rules"])
        add_rows(rep, "part1_recruiting_llm", rep["part1_recruiting_llm"])
        add_rows(rep, "part2_senior_engineers_rules", rep["part2_senior_engineers_rules"])
        add_rows(rep, "part2_senior_engineers_llm", rep["part2_senior_engineers_llm"])

    fieldnames = [
        "company",
        "location",
        "bucket",
        "source",
        "name",
        "confidence",
        "profile_url",
        "title_snippet",
        "why_matched",
        "llm_is_match",
        "message_300",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)