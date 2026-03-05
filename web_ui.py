import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from app.search_provider_brave import BraveSearchClient
from app.store import Store
from app.runner import run_for_company

load_dotenv()

app = FastAPI(title="LinkedIn Finder UI")

# static files
app.mount("/static", StaticFiles(directory="static"), name="static")


class RunRequest(BaseModel):
    companies_text: str
    location: str = "United States"
    role_focus: Optional[str] = None
    per_query_count: int = 10
    seed_n: int = 25
    rules_k: int = 2
    llm_k: int = 2


def _parse_companies(text: str) -> List[str]:
    # allow newline or comma separated
    raw = []
    for line in text.splitlines():
        raw.extend([x.strip() for x in line.split(",") if x.strip()])
    # dedupe while preserving order
    seen = set()
    out = []
    for c in raw:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


# Initialize shared clients once
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
if not BRAVE_API_KEY:
    raise RuntimeError("Missing BRAVE_API_KEY in .env")

client = BraveSearchClient(api_key=BRAVE_API_KEY)
#store = Store("store.db")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LinkedIn Finder</title>
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
  <div class="wrap">
    <h1>LinkedIn Finder</h1>
    <p class="muted">Enter one company per line (or comma-separated). Output shows 4 buckets: Recruiters (rules/LLM) + Senior Engineers (rules/LLM).</p>

    <div class="card">
      <label for="companies">Companies</label>
      <textarea id="companies" placeholder="Google&#10;Stripe&#10;Amazon"></textarea>

      <div class="row">
        <div class="col">
          <label for="location">Location</label>
          <input id="location" value="United States" />
        </div>
        <div class="col">
          <label for="role_focus">Role focus (optional)</label>
          <input id="role_focus" placeholder="software engineer / full stack / AI" />
        </div>
      </div>

      <div class="buttons">
        <button id="runBtn">Run</button>
        <button id="clearBtn" class="secondary">Clear</button>
      </div>

      <div id="status" class="status"></div>
    </div>

    <div id="results"></div>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>"""
    return HTMLResponse(html)


@app.post("/api/run")
def api_run(req: RunRequest) -> Dict[str, Any]:
    companies = _parse_companies(req.companies_text)
    if not companies:
        return {"reports": []}

    # IMPORTANT: create a new Store per request (avoids SQLite cross-thread issue)
    store = Store("store.db")

    reports: List[Dict[str, Any]] = []
    for c in companies:
        rep = run_for_company(
            company=c,
            client=client,
            store=store,
            location=req.location,
            resume_blurb="placeholder",
            role_focus=req.role_focus,
            per_query_count=req.per_query_count,
            seed_n=req.seed_n,
            rules_k=req.rules_k,
            llm_k=req.llm_k,
        )
        reports.append(rep)

    return {"reports": reports}