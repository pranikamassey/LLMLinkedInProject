from dotenv import load_dotenv
load_dotenv()

from app.search_provider_brave import brave_search

query = 'site:linkedin.com/in ("talent acquisition" OR recruiter OR "technical recruiter") "Stripe" "United States"'

results = brave_search(query, count=10)

for r in results:
    print(r["url"], "-", r["title"])