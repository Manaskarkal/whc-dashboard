"""audit.py - Phase 2: Cryptographically tamper-evident audit trail & WAL-safe SQLite backups.

    AuditEntry:       one immutable audit log record
    AuditRepository:  append-only, SHA-256 hash-chained `audit_log` table with `verify_chain()`
    BackupInfo:       metadata for one backup snapshot
    BackupManager:    WAL-safe online SQLite backups (`sqlite3.Connection.backup()`),
                      post-backup integrity verification, separate per-type retention,
                      and verified restore helper.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from whc.database import Database, Repository
from whc.exceptions import ValidationError

DEFAULT_BACKUP_DIR = Path(__file__).resolve().parent.parent / "backups"
GENESIS_HASH = "0" * 64

# Separate retention caps so pre-bulk / manual backups never crowd out daily history
RETENTION_BY_REASON: dict[str, int] = {
    "daily": 30,
    "pre_bulk": 15,
    "manual": 15,
}
DEFAULT_RETENTION = 15

_REASON_RE = re.compile(r"[a-z0-9_-]{1,24}")
_BACKUP_FILE_RE = re.compile(
    r"^whc_(\d{4}-\d{2}-\d{2})_(\d{6})_([a-z0-9_-]+?)(?:_\d{2})?\.db$"
)
_REDACTED_KEYS = frozenset({
    "password", "password_hash", "new_password", "current_password",
    "temporary_password", "token", "token_hash", "session_token", "secret",
})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _scrub_details(obj: Any) -> Any:
    """Recursively redact sensitive keys so passwords or tokens can NEVER reach audit_log."""
    if isinstance(obj, dict):
        return {
            str(k): ("[REDACTED]" if str(k).lower() in _REDACTED_KEYS else _scrub_details(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_scrub_details(item) for item in obj]
    return obj


def _compute_row_hash(
    prev_hash: str,
    occurred_at: str,
    actor_user_id: int | None,
    actor_username: str,
    action: str,
    target_table: str | None,
    target_id: str | None,
    details_json: str,
) -> str:
    payload = json.dumps(
        [
            prev_hash,
            occurred_at,
            actor_user_id,
            actor_username,
            action,
            target_table,
            target_id,
            details_json,
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---- AuditEntry & AuditRepository --------------------------------------------------
@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One immutable, hash-chained audit record."""

    audit_id: int
    occurred_at: str
    actor_user_id: int | None
    actor_username: str
    action: str
    target_table: str | None
    target_id: str | None
    details_json: str
    prev_hash: str
    row_hash: str

    @property
    def details(self) -> dict[str, Any]:
        return json.loads(self.details_json)

    @classmethod
    def from_row(cls, row: tuple) -> AuditEntry:
        return cls(*row)


_AUDIT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at    TEXT NOT NULL,
    actor_user_id  INTEGER,
    actor_username TEXT NOT NULL CHECK (length(actor_username) BETWEEN 1 AND 64),
    action         TEXT NOT NULL CHECK (length(action) BETWEEN 1 AND 64),
    target_table   TEXT,
    target_id      TEXT,
    details_json   TEXT NOT NULL DEFAULT '{}',
    prev_hash      TEXT NOT NULL CHECK (length(prev_hash) = 64),
    row_hash       TEXT NOT NULL CHECK (length(row_hash) = 64)
) STRICT;

CREATE INDEX IF NOT EXISTS ix_audit_log_occurred_at ON audit_log (occurred_at);
CREATE INDEX IF NOT EXISTS ix_audit_log_actor ON audit_log (actor_username);
CREATE INDEX IF NOT EXISTS ix_audit_log_action ON audit_log (action);

DROP TRIGGER IF EXISTS audit_log_no_update;
CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is immutable and cannot be updated'); END;

DROP TRIGGER IF EXISTS audit_log_no_delete;
CREATE TRIGGER audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is immutable and cannot be deleted'); END;
"""

_AUDIT_SELECT = (
    "SELECT audit_id, occurred_at, actor_user_id, actor_username, action, "
    "target_table, target_id, details_json, prev_hash, row_hash FROM audit_log"
)

_AUDIT_DF_SCHEMA = {
    "audit_id": pl.Int64,
    "occurred_at": pl.String,
    "actor_user_id": pl.Int64,
    "actor_username": pl.String,
    "action": pl.String,
    "target_table": pl.String,
    "target_id": pl.String,
    "details_json": pl.String,
    "prev_hash": pl.String,
    "row_hash": pl.String,
}


class AuditRepository(Repository):
    """Append-only, SHA-256 hash-chained audit trail."""

    def __init__(self, db: Database, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock
        super().__init__(db)

    @property
    def schema_sql(self) -> str:
        return _AUDIT_SCHEMA_SQL

    def record(
        self,
        action: str,
        *,
        actor_user_id: int | None = None,
        actor_username: str = "system",
        target_table: str | None = None,
        target_id: str | int | None = None,
        details: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> AuditEntry:
        """Append one hash-chained audit event.

        Pass an active `conn` to commit atomically inside an existing transaction.
        """
        action_clean = action.strip().upper()
        if not action_clean or len(action_clean) > 64:
            raise ValidationError("audit action must be 1-64 characters")
        actor_clean = actor_username.strip() or "system"
        if len(actor_clean) > 64:
            actor_clean = actor_clean[:64]
        tid_str = str(target_id) if target_id is not None else None
        scrubbed = _scrub_details(details) if details is not None else {}
        details_json = json.dumps(scrubbed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        occurred_at = _iso_utc(self._clock())

        def _insert(c: sqlite3.Connection) -> AuditEntry:
            last = c.execute(
                "SELECT row_hash FROM audit_log ORDER BY audit_id DESC LIMIT 1"
            ).fetchone()
            prev_hash = last[0] if last else GENESIS_HASH
            row_hash = _compute_row_hash(
                prev_hash, occurred_at, actor_user_id, actor_clean,
                action_clean, target_table, tid_str, details_json,
            )
            cur = c.execute(
                "INSERT INTO audit_log "
                "(occurred_at, actor_user_id, actor_username, action, "
                "target_table, target_id, details_json, prev_hash, row_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    occurred_at, actor_user_id, actor_clean, action_clean,
                    target_table, tid_str, details_json, prev_hash, row_hash,
                ),
            )
            assert cur.lastrowid is not None
            return AuditEntry(
                audit_id=cur.lastrowid,
                occurred_at=occurred_at,
                actor_user_id=actor_user_id,
                actor_username=actor_clean,
                action=action_clean,
                target_table=target_table,
                target_id=tid_str,
                details_json=details_json,
                prev_hash=prev_hash,
                row_hash=row_hash,
            )

        if conn is not None:
            return _insert(conn)
        with self._db.transaction() as tx_conn:
            return _insert(tx_conn)

    def verify_chain(self) -> tuple[bool, str]:
        """Verify the cryptographic SHA-256 hash chain across every row in `audit_log`."""
        rows = self._db.query(_AUDIT_SELECT + " ORDER BY audit_id ASC")
        expected_prev = GENESIS_HASH
        for row in rows:
            entry = AuditEntry.from_row(row)
            if entry.prev_hash != expected_prev:
                return False, f"broken prev_hash link at audit_id={entry.audit_id}"
            recomputed = _compute_row_hash(
                entry.prev_hash,
                entry.occurred_at,
                entry.actor_user_id,
                entry.actor_username,
                entry.action,
                entry.target_table,
                entry.target_id,
                entry.details_json,
            )
            if entry.row_hash != recomputed:
                return False, f"tampered row_hash at audit_id={entry.audit_id}"
            expected_prev = entry.row_hash
        return True, f"ok ({len(rows)} entries verified)"

    def latest_hash(self) -> str:
        """Return the latest chain head hash (for off-DB verification anchors)."""
        rows = self._db.query("SELECT row_hash FROM audit_log ORDER BY audit_id DESC LIMIT 1")
        return rows[0][0] if rows else GENESIS_HASH

    def recent(
        self,
        limit: int = 100,
        *,
        actor_username: str | None = None,
        action: str | None = None,
    ) -> list[AuditEntry]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValidationError("limit must be a positive whole number")
        conditions: list[str] = []
        params: list[object] = []
        if actor_username is not None:
            conditions.append("lower(actor_username) = lower(?)")
            params.append(actor_username.strip())
        if action is not None:
            conditions.append("action = ?")
            params.append(action.strip().upper())
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)
        rows = self._db.query(
            _AUDIT_SELECT + where + " ORDER BY audit_id DESC LIMIT ?", tuple(params)
        )
        return [AuditEntry.from_row(r) for r in rows]

    def to_dataframe(
        self,
        limit: int = 500,
        *,
        actor_username: str | None = None,
        action: str | None = None,
    ) -> pl.DataFrame:
        entries = self.recent(limit=limit, actor_username=actor_username, action=action)
        rows = [
            (
                e.audit_id, e.occurred_at, e.actor_user_id, e.actor_username, e.action,
                e.target_table, e.target_id, e.details_json, e.prev_hash, e.row_hash,
            )
            for e in entries
        ]
        df = pl.DataFrame(rows, schema=_AUDIT_DF_SCHEMA, orient="row")
        return df.with_columns(pl.col("occurred_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"))


# ---- BackupManager -----------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BackupInfo:
    path: Path
    filename: str
    reason: str
    created_date: str
    size_bytes: int


def _verify_sqlite_file(path: Path) -> None:
    """Verify a SQLite snapshot passes both integrity_check and foreign_key_check."""
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        integrity = conn.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise RuntimeError(f"backup failed integrity_check: {integrity}")
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if fk_errors:
            raise RuntimeError(f"backup failed foreign_key_check ({len(fk_errors)} errors)")
    finally:
        conn.close()


class BackupManager:
    """WAL-safe online SQLite backups with post-backup verification & per-type retention."""

    def __init__(
        self,
        db: Database,
        backup_dir: Path | str = DEFAULT_BACKUP_DIR,
        *,
        retention: dict[str, int] | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._db = db
        self._dir = Path(backup_dir)
        self._retention = dict(RETENTION_BY_REASON if retention is None else retention)
        self._clock = clock

    @property
    def backup_dir(self) -> Path:
        return self._dir

    def create_backup(
        self,
        reason: str = "manual",
        *,
        audit: AuditRepository | None = None,
        actor_user_id: int | None = None,
        actor_username: str = "system",
    ) -> Path:
        """Create an online SQLite snapshot, verify its integrity, and prune old snapshots of this type."""
        clean_reason = reason.strip().lower()
        if not _REASON_RE.fullmatch(clean_reason):
            raise ValidationError("backup reason must be 1-24 lowercase letters, digits, _ or -")

        self._dir.mkdir(parents=True, exist_ok=True)
        stamp = self._clock().astimezone(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        target = self._dir / f"whc_{stamp}_{clean_reason}.db"
        counter = 1
        while target.exists():
            target = self._dir / f"whc_{stamp}_{clean_reason}_{counter:02d}.db"
            counter += 1

        dest_conn = sqlite3.connect(target)
        try:
            self._db.backup_to(dest_conn)
        finally:
            dest_conn.close()

        try:
            _verify_sqlite_file(target)
        except Exception:
            target.unlink(missing_ok=True)
            raise

        self._prune_reason(clean_reason)

        if audit is not None:
            audit.record(
                "BACKUP_CREATED",
                actor_user_id=actor_user_id,
                actor_username=actor_username,
                target_table="system",
                target_id=target.name,
                details={"reason": clean_reason, "size_bytes": target.stat().st_size},
            )
        return target

    def ensure_daily_backup(
        self,
        *,
        audit: AuditRepository | None = None,
        actor_user_id: int | None = None,
        actor_username: str = "system",
    ) -> Path | None:
        """Create a daily snapshot if one does not already exist for today's UTC date."""
        today_str = self._clock().astimezone(timezone.utc).strftime("%Y-%m-%d")
        if self._dir.exists():
            for existing in self._dir.glob(f"whc_{today_str}_*_daily*.db"):
                if existing.is_file():
                    return None
        return self.create_backup(
            "daily",
            audit=audit,
            actor_user_id=actor_user_id,
            actor_username=actor_username,
        )

    def list_backups(self) -> list[BackupInfo]:
        """List all valid backup snapshots newest-first."""
        if not self._dir.exists():
            return []
        items: list[BackupInfo] = []
        for p in sorted(self._dir.glob("whc_*.db"), key=lambda x: x.name, reverse=True):
            m = _BACKUP_FILE_RE.fullmatch(p.name)
            if not m or not p.is_file():
                continue
            items.append(
                BackupInfo(
                    path=p,
                    filename=p.name,
                    reason=m.group(3),
                    created_date=m.group(1),
                    size_bytes=p.stat().st_size,
                )
            )
        return items

    @staticmethod
    def restore_to(backup_path: Path | str, destination: Path | str) -> Path:
        """Verify `backup_path` and restore it safely into `destination` via SQLite backup API."""
        src = Path(backup_path)
        if not src.exists():
            raise FileNotFoundError(f"backup file not found: {src}")
        _verify_sqlite_file(src)

        dst = Path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_conn = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
        dst_conn = sqlite3.connect(dst)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
            src_conn.close()
        _verify_sqlite_file(dst)
        return dst

    def _prune_reason(self, reason: str) -> None:
        keep = self._retention.get(reason, DEFAULT_RETENTION)
        matching = [b.path for b in self.list_backups() if b.reason == reason]
        for old_path in matching[keep:]:
            old_path.unlink(missing_ok=True)
