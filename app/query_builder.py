from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class QueryPack:
    part1_queries: List[str]
    part2_queries: List[str]


def build_queries(company: str, location: str = "United States", role_focus: Optional[str] = None) -> QueryPack:
    """
    Creates a small number of high-signal queries for each part.
    We avoid scraping by using search-engine queries only.
    """
    c = company.strip()
    loc = location.strip()

    focus = role_focus.strip() if role_focus else ""
    # Keep focus terms small + generic, to avoid overfitting.
    focus_terms = focus if focus else '"software engineer" OR "full stack" OR "AI" OR "ML" OR backend OR frontend OR platform'

    # Part 1 (recruiting/TA)
    part1 = [
        f'site:linkedin.com/in ("talent acquisition" OR recruiter OR "technical recruiter" OR sourcer OR "university recruiter" OR "talent partner") "{c}" "{loc}" ({focus_terms})',
        f'site:linkedin.com/in (recruiter OR sourcer OR "talent acquisition") "{c}" ("software" OR engineer OR "full stack" OR AI OR ML) "{loc}"',
        f'site:linkedin.com/in ("people operations" OR HRBP OR "human resources") "{c}" ("engineering" OR "software") "{loc}"',
    ]

    # Part 2 (senior engineers / managers)
    part2 = [
        f'site:linkedin.com/in ("Senior Software Engineer" OR "Staff Software Engineer" OR "Principal Engineer" OR "Engineering Manager" OR "Tech Lead") "{c}" "{loc}"',
        f'site:linkedin.com/in (Senior OR Staff OR Principal OR "Engineering Manager" OR "Tech Lead") "{c}" (backend OR frontend OR "full stack" OR platform OR AI OR ML) "{loc}"',
        f'site:linkedin.com/in ("software engineer" OR "engineering manager") "{c}" ("Staff" OR "Principal" OR "Senior" OR "Lead") "{loc}"',
    ]

    return QueryPack(part1_queries=part1, part2_queries=part2)