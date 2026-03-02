import os
from dotenv import load_dotenv
from app.search_provider_brave import BraveSearchClient

load_dotenv()

key = os.getenv("BRAVE_API_KEY", "").strip()
client = BraveSearchClient(key)

q = 'site:linkedin.com/in recruiter "Stripe" "United States"'
results = client.search(q, count=5, country="US")
print("Got:", len(results))
for r in results[:3]:
    print("-", r["title"])
    print("  ", r["url"])