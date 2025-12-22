from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_request(payload: dict[str, Any]) -> str:
    """
    Hash a request payload (excluding idempotency_key) for idempotency validation.
    """
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IdempotencyRecord:
    request_hash: str
    response_json: str


class IdempotencyStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._ensure_parent_dir()
        self._init_db()

    def _ensure_parent_dir(self) -> None:
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency (
                  key TEXT PRIMARY KEY,
                  request_hash TEXT NOT NULL,
                  response_json TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                """
            )

    def get(self, key: str) -> IdempotencyRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT request_hash, response_json FROM idempotency WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return IdempotencyRecord(request_hash=row[0], response_json=row[1])

    def put_if_absent(self, key: str, request_hash: str, response_json: str) -> bool:
        """Insert a new record if the key does not already exist."""
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO idempotency(key, request_hash, response_json) VALUES(?,?,?)",
                    (key, request_hash, response_json),
                )
                return True
            except sqlite3.IntegrityError:
                return False


