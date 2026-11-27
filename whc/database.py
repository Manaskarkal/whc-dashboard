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
MIN_SQLITE_VERSION = (3, 37, 0)  # STRICT tables + generated columns need this
SCHEMA_VERSION = 2  # v2: normalized test_results, DB-driven lookups, legacy import lock

_SYSTEM_CONFIG_SQL = """
CREATE TABLE IF NOT EXISTS system_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

INSERT OR IGNORE INTO system_config (key, value) VALUES ('allow_legacy_import', '0');
"""


class Database:
    """One SQLite connection, opened safely. Use it in a `with` block so it always closes."""

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
            raise RuntimeError(f"SQLite {sqlite3.sqlite_version} is too old; need 3.37 or newer")
        is_memory = str(path) == ":memory:"
        if not is_memory:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._tx_depth = 0
        self._conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignores links unless this is on
        self._conn.execute("PRAGMA busy_timeout = 5000")  # wait up to 5s on concurrent writes
        if not is_memory:
            self._conn.execute("PRAGMA journal_mode = WAL")  # concurrent readers during writes
            self._conn.execute("PRAGMA synchronous = NORMAL")  # safe + fast with WAL
        self._conn.executescript(_SYSTEM_CONFIG_SQL)
        if self.schema_version == 0:
            self.set_schema_version(SCHEMA_VERSION)

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @property
    def schema_version(self) -> int:
        (version,) = self._conn.execute("PRAGMA user_version").fetchone()
        return int(version)

    def set_schema_version(self, version: int) -> None:
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("schema version must be a non-negative integer")
        self._conn.execute(f"PRAGMA user_version = {version}")

    def is_legacy_import_open(self) -> bool:
        rows = self.query("SELECT value FROM system_config WHERE key = 'allow_legacy_import'")
        return bool(rows and rows[0][0] == "1")

    @contextmanager
    def legacy_import_mode(self) -> Iterator[None]:
        """Temporarily allow legacy rows (NULL dob / NULL test_date) during one-time import.

        Always re-locks on exit, even if an exception occurs.
        """
        with self.transaction() as conn:
            conn.execute("UPDATE system_config SET value = '1' WHERE key = 'allow_legacy_import'")
        try:
            yield
        finally:
            with self.transaction() as conn:
                conn.execute("UPDATE system_config SET value = '0' WHERE key = 'allow_legacy_import'")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """All-or-nothing block (supports nested calls via SQLite SAVEPOINTs)."""
        if self._tx_depth == 0:
            self._tx_depth += 1
            try:
                with self._conn:
                    yield self._conn
            finally:
                self._tx_depth -= 1
            return

        sp = f"whc_sp_{self._tx_depth}"
        self._tx_depth += 1
        self._conn.execute(f"SAVEPOINT {sp}")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            self._conn.execute(f"RELEASE SAVEPOINT {sp}")
            raise
        else:
            self._conn.execute(f"RELEASE SAVEPOINT {sp}")
        finally:
            self._tx_depth -= 1

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Run a SELECT and return all rows. Always pass values through `params`."""
        return self._conn.execute(sql, params).fetchall()

    def run_script(self, script: str) -> None:
        """Run several SQL statements at once (used to create tables)."""
        self._conn.executescript(script)

    def backup_to(self, destination: sqlite3.Connection) -> None:
        """Safely snapshot this database into `destination` using SQLite's online backup API."""
        self._conn.backup(destination)


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
