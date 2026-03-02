import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .scoring import ScoredCandidate


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


class Store:
    def __init__(self, path: str = "linkedin_finder.db"):
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS companies (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT UNIQUE,
              created_at TEXT
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              company_id INTEGER,
              part TEXT,
              name TEXT,
              url TEXT UNIQUE,
              snippet TEXT,
              confidence REAL,
              why_json TEXT,
              message TEXT,
              last_seen_at TEXT,
              FOREIGN KEY(company_id) REFERENCES companies(id)
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outreach (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              candidate_id INTEGER,
              status TEXT,
              note TEXT,
              created_at TEXT,
              FOREIGN KEY(candidate_id) REFERENCES candidates(id)
            );
            """
        )
        self.conn.commit()

    def upsert_company(self, name: str) -> int:
        name = name.strip()
        self.conn.execute(
            "INSERT OR IGNORE INTO companies(name, created_at) VALUES(?, ?)",
            (name, _now()),
        )
        self.conn.commit()
        cur = self.conn.execute("SELECT id FROM companies WHERE name=?", (name,))
        row = cur.fetchone()
        return int(row[0])

    def upsert_candidate(
        self,
        company_id: int,
        sc: ScoredCandidate,
        message: str,
    ) -> int:
        c = sc.candidate
        why_json = json.dumps(sc.why_matched, ensure_ascii=False)
        self.conn.execute(
            """
            INSERT INTO candidates(company_id, part, name, url, snippet, confidence, why_json, message, last_seen_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
              company_id=excluded.company_id,
              part=excluded.part,
              name=excluded.name,
              snippet=excluded.snippet,
              confidence=excluded.confidence,
              why_json=excluded.why_json,
              message=excluded.message,
              last_seen_at=excluded.last_seen_at
            """,
            (
                company_id,
                c.part,
                c.name,
                c.url,
                c.title_snippet,
                float(sc.confidence),
                why_json,
                message,
                _now(),
            ),
        )
        self.conn.commit()
        cur = self.conn.execute("SELECT id FROM candidates WHERE url=?", (c.url,))
        row = cur.fetchone()
        return int(row[0])

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass