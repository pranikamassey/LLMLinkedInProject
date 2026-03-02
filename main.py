import argparse
import os
from typing import List, Optional

from dotenv import load_dotenv
from rich.console import Console

from app.search_provider_brave import BraveSearchClient
from app.runner import run_for_company, print_company_report, export_json, export_csv
from app.store import Store


def read_companies_file(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Find LinkedIn contacts via Brave Search (ToS-friendly).")
    parser.add_argument("--company", type=str, help="Single company name (e.g., Stripe)")
    parser.add_argument("--companies", type=str, help="Path to companies.txt (one per line)")
    parser.add_argument("--location", type=str, default=os.getenv("DEFAULT_LOCATION", "United States"))
    parser.add_argument("--role-focus", type=str, default=None, help='Optional focus like "AI engineer" or "full stack"')
    parser.add_argument("--per-query-count", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out-json", type=str, default="output.json")
    parser.add_argument("--out-csv", type=str, default="output.csv")
    parser.add_argument("--db", type=str, default="linkedin_finder.db")
    args = parser.parse_args()

    api_key = os.getenv("BRAVE_API_KEY", "").strip()
    resume_blurb = os.getenv("RESUME_BLURB", "").strip()
    if not resume_blurb:
        resume_blurb = "I’m ABC with XYZ experience at PQR, currently pursuing an MSCS at DEF."

    if not api_key:
        raise SystemExit("Missing BRAVE_API_KEY in .env")

    companies: List[str] = []
    if args.company:
        companies = [args.company.strip()]
    elif args.companies:
        companies = read_companies_file(args.companies)
    else:
        raise SystemExit("Provide either --company or --companies")

    console = Console()
    client = BraveSearchClient(api_key=api_key)
    store = Store(path=args.db)

    reports = []
    try:
        for comp in companies:
            rep = run_for_company(
                company=comp,
                client=client,
                store=store,
                location=args.location,
                resume_blurb=resume_blurb,
                role_focus=args.role_focus,
                per_query_count=args.per_query_count,
                top_k=args.top_k,
            )
            reports.append(rep)
            print_company_report(console, rep)

        export_json(args.out_json, reports)
        export_csv(args.out_csv, reports)

        console.print(f"\n[bold green]Saved[/bold green] {args.out_json} and {args.out_csv}")
        console.print(f"[dim]SQLite tracking DB:[/dim] {args.db}")

    finally:
        store.close()


if __name__ == "__main__":
    main()