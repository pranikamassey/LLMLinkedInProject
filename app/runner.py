import json
import csv
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from .search_provider_brave import BraveSearchClient
from .query_builder import build_queries
from .normalize import normalize_results_to_candidates, dedup_candidates
from .scoring import rank_candidates, diversify_top
from .messaging import draft_message, infer_role_focus
from .store import Store


def run_for_company(
    company: str,
    client: BraveSearchClient,
    store: Store,
    location: str,
    resume_blurb: str,
    role_focus: Optional[str] = None,
    per_query_count: int = 10,
    top_k: int = 6,
) -> Dict[str, Any]:
    qp = build_queries(company=company, location=location, role_focus=role_focus)

    # Search
    part1_results: List[Dict[str, Any]] = []
    for q in qp.part1_queries:
        part1_results.extend(client.search(q, count=per_query_count, country="US"))

    part2_results: List[Dict[str, Any]] = []
    for q in qp.part2_queries:
        part2_results.extend(client.search(q, count=per_query_count, country="US"))

    # Normalize + filter + dedup
    part1_cands = dedup_candidates(normalize_results_to_candidates(part1_results, company=company, part="part1"))
    part2_cands = dedup_candidates(normalize_results_to_candidates(part2_results, company=company, part="part2"))

    # Score
    ranked1 = rank_candidates(part1_cands, location=location)
    ranked2 = rank_candidates(part2_cands, location=location)

    top1 = diversify_top(ranked1, k=top_k)
    top2 = diversify_top(ranked2, k=top_k)

    # Messages + persist
    company_id = store.upsert_company(company)

    part1_out: List[Dict[str, Any]] = []
    for sc in top1:
        focus = role_focus.strip() if role_focus else infer_role_focus(sc.candidate.title_snippet)
        msg = draft_message(sc, resume_blurb=resume_blurb, role_focus=focus)
        store.upsert_candidate(company_id, sc, msg)
        d = sc.to_output_dict()
        d["message_300"] = msg
        part1_out.append(d)

    part2_out: List[Dict[str, Any]] = []
    for sc in top2:
        focus = role_focus.strip() if role_focus else infer_role_focus(sc.candidate.title_snippet)
        msg = draft_message(sc, resume_blurb=resume_blurb, role_focus=focus)
        store.upsert_candidate(company_id, sc, msg)
        d = sc.to_output_dict()
        d["message_300"] = msg
        part2_out.append(d)

    return {
        "company": company,
        "location": location,
        "part1_recruiting": part1_out,
        "part2_senior_engineers": part2_out,
    }


def print_company_report(console: Console, report: Dict[str, Any]) -> None:
    console.rule(f"[bold]{report['company']}[/bold] — {report['location']}")

    t1 = Table(title="Part 1 — Recruiting / TA (Top matches)")
    t1.add_column("#", justify="right", width=3)
    t1.add_column("Name", overflow="fold")
    t1.add_column("Confidence", justify="right", width=10)
    t1.add_column("Profile", overflow="fold")
    t1.add_column("Why", overflow="fold")

    for i, p in enumerate(report["part1_recruiting"], start=1):
        t1.add_row(
            str(i),
            p["name"],
            str(p["confidence"]),
            p["profile_url"],
            ", ".join(p["why_matched"][:3]),
        )
    console.print(t1)

    t2 = Table(title="Part 2 — Senior Engineers (Top matches)")
    t2.add_column("#", justify="right", width=3)
    t2.add_column("Name", overflow="fold")
    t2.add_column("Confidence", justify="right", width=10)
    t2.add_column("Profile", overflow="fold")
    t2.add_column("Why", overflow="fold")

    for i, p in enumerate(report["part2_senior_engineers"], start=1):
        t2.add_row(
            str(i),
            p["name"],
            str(p["confidence"]),
            p["profile_url"],
            ", ".join(p["why_matched"][:3]),
        )
    console.print(t2)

    # Show messages separately so it’s copy/paste friendly
    console.print("\n[bold]Messages (copy/paste)[/bold]")
    for section_name, items in [
        ("Part 1", report["part1_recruiting"]),
        ("Part 2", report["part2_senior_engineers"]),
    ]:
        console.print(f"\n[underline]{section_name}[/underline]")
        for i, p in enumerate(items, start=1):
            console.print(f"{i}. {p['name']} — {p['profile_url']}")
            console.print(f"   {p['message_300']}")


def export_json(path: str, reports: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def export_csv(path: str, reports: List[Dict[str, Any]]) -> None:
    """
    Flatten output into a single CSV.
    """
    rows: List[Dict[str, Any]] = []
    for rep in reports:
        company = rep["company"]
        location = rep["location"]

        for part_label, items in [
            ("part1_recruiting", rep["part1_recruiting"]),
            ("part2_senior_engineers", rep["part2_senior_engineers"]),
        ]:
            for p in items:
                rows.append(
                    {
                        "company": company,
                        "location": location,
                        "part": part_label,
                        "name": p.get("name", ""),
                        "confidence": p.get("confidence", ""),
                        "profile_url": p.get("profile_url", ""),
                        "title_snippet": p.get("title_snippet", ""),
                        "why_matched": " | ".join(p.get("why_matched", [])),
                        "message_300": p.get("message_300", ""),
                    }
                )

    fieldnames = [
        "company",
        "location",
        "part",
        "name",
        "confidence",
        "profile_url",
        "title_snippet",
        "why_matched",
        "message_300",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)