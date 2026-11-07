"""database.py - the database connection and the base class for table classes.

Every table gets its own class (see students.py) that inherits from Repository.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "whc.db"
MIN_SQLITE_VERSION = (3, 37, 0)  # STRICT tables need this


class Database:
    """One SQLite connection, opened safely. Use it in a `with` block so it always closes."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
            raise RuntimeError(f"SQLite {sqlite3.sqlite_version} is too old; need 3.37 or newer")
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignores links unless this is on

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All-or-nothing block: saved if it finishes, undone if anything raises."""
        with self._conn:
            yield self._conn

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Run a SELECT and return all rows. Always pass values through `params`."""
        return self._conn.execute(sql, params).fetchall()

    def run_script(self, script: str) -> None:
        """Run several SQL statements at once (used to create tables)."""
        self._conn.executescript(script)


class Repository(ABC):
    """Base class for the class that owns ONE table.

    A subclass supplies `schema_sql`; the table is created if it is missing.
    Adding a new table later = writing a new subclass. Nothing existing changes.
    """

    def __init__(self, db: Database) -> None:
        self._db = db  # handed in, not created here, so tests can pass an in-memory database
        self._db.run_script(self.schema_sql)

    @property
    @abstractmethod
    def schema_sql(self) -> str:
        """SQL that creates this table (must be safe to run repeatedly)."""
