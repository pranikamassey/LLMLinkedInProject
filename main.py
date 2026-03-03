import argparse
import os
from rich.console import Console

from dotenv import load_dotenv

from app.search_provider_brave import BraveSearchClient
from app.store import Store
from app.runner import run_for_company, print_company_report, export_json, export_csv


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--company", type=str, help="Single company name", default=None)
    parser.add_argument("--companies", type=str, help="Path to a file with company names (one per line)", default=None)
    parser.add_argument("--location", type=str, default="United States")
    parser.add_argument("--resume_blurb", type=str, default="I’m ABC with XYZ experience at PQR, currently pursuing an MSCS at DEF.")
    parser.add_argument("--per_query_count", type=int, default=10)
    parser.add_argument("--seed_n", type=int, default=25)
    parser.add_argument("--rules_k", type=int, default=2)
    parser.add_argument("--llm_k", type=int, default=2)
    parser.add_argument("--out_json", type=str, default="out.json")
    parser.add_argument("--out_csv", type=str, default="out.csv")
    args = parser.parse_args()

    companies = []
    if args.company:
        companies = [args.company.strip()]
    elif args.companies:
        with open(args.companies, "r", encoding="utf-8") as f:
            companies = [line.strip() for line in f if line.strip()]
    else:
        raise SystemExit("Provide --company or --companies")

    brave_key = os.getenv("BRAVE_API_KEY")
    if not brave_key:
        raise SystemExit("Missing BRAVE_API_KEY in .env")

    # OpenAI is optional for this run; if missing, LLM buckets will fail at runtime.
    # Keep it explicit so you notice quickly:
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. LLM buckets will fail.")

    client = BraveSearchClient(api_key=brave_key)
    store = Store("store.db")
    console = Console()

    reports = []
    for c in companies:
        rep = run_for_company(
            company=c,
            client=client,
            store=store,
            location=args.location,
            resume_blurb=args.resume_blurb,
            per_query_count=args.per_query_count,
            seed_n=args.seed_n,
            rules_k=args.rules_k,
            llm_k=args.llm_k,
        )
        reports.append(rep)
        print_company_report(console, rep)

    export_json(args.out_json, reports)
    export_csv(args.out_csv, reports)

    console.print(f"\n[bold]Wrote:[/bold] {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()