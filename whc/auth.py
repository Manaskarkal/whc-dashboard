"""auth.py - Phase 2: Argon2id authentication, sessions, lockout, RBAC roles, and break-glass CLI.

    Role:            Lead Coach | Coach
    User:            validated, immutable user view (never exposes password hashes)
    AuthRepository:  manages `users` and `sessions` tables with OWASP / NIST best practices

Security Controls Implemented:
    * Argon2id via `argon2-cffi` (64 MiB memory, 3 iterations, 4 parallelism, RFC 9106)
      with transparent `check_needs_rehash()` upgrades on login.
    * 12-128 character password policy with Unicode NFKC normalization and offline
      common/breached password blocklist (+ rejects username & school/program names).
    * No username enumeration: identical generic error message on unknown username,
      wrong password, inactive user, or locked account + dummy-hash timing defense.
    * Atomic failed-attempt counter: 5 consecutive failures -> 15-minute lockout
      (Lead Coach can unlock anytime).
    * Cryptographically random 256-bit session tokens (`secrets.token_urlsafe(32)`),
      stored SHA-256 hashed in SQLite (`sessions`). 2-hour idle timeout + 12-hour max lifetime.
    * Expiring temporary passwords (72 hours) that force immediate password change on login
      and cannot be reused as the new password.
    * Session revocation on password change, password reset, deactivation, and role change.
    * SQLite triggers block user deletion and prevent deactivating/demoting the last active Lead Coach.
    * Break-glass host CLI: `uv run python -m whc.auth reset-lead-coach <username>`.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum

import polars as pl
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from whc.audit import AuditRepository
from whc.database import DEFAULT_DB_PATH, Database, Repository
from whc.exceptions import (
    AuthenticationError,
    DuplicateError,
    NotFoundError,
    ValidationError,
)
from whc.students import NAME_MAX_LENGTH, NAME_PATTERN, parse_enum, require_text

# ---- Security Settings -------------------------------------------------------------
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128
USERNAME_PATTERN = re.compile(r"[a-z][a-z0-9._-]{2,31}")  # 3-32 chars, starts with letter

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION = timedelta(minutes=15)
IDLE_TIMEOUT = timedelta(hours=2)
MAX_SESSION_LIFETIME = timedelta(hours=12)
TEMP_PASSWORD_TTL = timedelta(hours=72)

GENERIC_LOGIN_ERROR = "Invalid username or password"

# OWASP Argon2id parameters: 64 MiB RAM, 3 iterations, 4 lanes
_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)
# Precomputed dummy hash for constant-time rejection when username does not exist
_DUMMY_HASH = _HASHER.hash("whc-dummy-constant-time-verification-placeholder!")

# Curated offline common/breached password blocklist (lowercase, NFKC-normalized)
_BLOCKED_PASSWORDS: frozenset[str] = frozenset({
    "password1234", "password12345", "password123456", "123456789012",
    "1234567890123", "12345678901234", "qwertyuiop12", "qwerty123456",
    "letmein12345", "welcome12345", "admin1234567", "football1234",
    "basketball12", "westernheights", "westernheightscollege", "whcdashboard",
    "specialistsports", "iloveyou1234", "changeme1234", "abc123456789",
    "000000000000", "111111111111", "aaaaaaaaaaaa", "zzzzzzzzzzzz",
})


class Role(StrEnum):
    LEAD_COACH = "Lead Coach"
    COACH = "Coach"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _clean_username(value: object) -> str:
    cleaned = require_text(value, "username").strip().lower()
    if not USERNAME_PATTERN.fullmatch(cleaned):
        raise ValidationError(
            "username must be 3-32 characters starting with a letter (letters, digits, . _ - only)"
        )
    return cleaned


def _clean_full_name(value: object) -> str:
    cleaned = unicodedata.normalize("NFC", " ".join(require_text(value, "full_name").split()))
    if not cleaned or len(cleaned) > NAME_MAX_LENGTH or not NAME_PATTERN.fullmatch(cleaned):
        raise ValidationError(
            f"full_name must be letters only (single spaces, - and ' allowed), max {NAME_MAX_LENGTH} characters"
        )
    return cleaned


def _normalize_password(password: object) -> str:
    raw = require_text(password, "password")
    return unicodedata.normalize("NFKC", raw)


def validate_new_password(password: object, *, username: str | None = None) -> str:
    """Validate a new password against our 12-128 char NFKC + blocklist policy."""
    norm = _normalize_password(password)
    if len(norm) < MIN_PASSWORD_LENGTH:
        raise ValidationError(f"password must be at least {MIN_PASSWORD_LENGTH} characters long")
    if len(norm) > MAX_PASSWORD_LENGTH:
        raise ValidationError(f"password must be at most {MAX_PASSWORD_LENGTH} characters long")
    if len(set(norm)) < 4:
        raise ValidationError("password is too repetitive; choose a stronger passphrase")
    folded = re.sub(r"\s+", "", norm).casefold()
    if folded in _BLOCKED_PASSWORDS:
        raise ValidationError("password is too common; please choose a stronger passphrase")
    if username and username.casefold() in folded:
        raise ValidationError("password cannot contain your username")
    return norm


def generate_temporary_password() -> str:
    """Generate a high-entropy, readable one-time temporary password (16+ chars)."""
    return f"WHC-{secrets.token_urlsafe(12)}"


# ---- User value object -------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class User:
    """Safe, immutable user view (never carries password hashes)."""

    user_id: int
    username: str
    full_name: str
    role: Role
    is_active: bool
    must_change_password: bool
    temp_password_expires_at: str | None
    failed_attempts: int
    locked_until: str | None
    created_at: str

    @property
    def is_lead_coach(self) -> bool:
        return self.role is Role.LEAD_COACH

    @classmethod
    def from_row(cls, row: tuple) -> User:
        (
            user_id, username, full_name, role, is_active, must_change_password,
            temp_password_expires_at, failed_attempts, locked_until, created_at,
        ) = row
        return cls(
            user_id=int(user_id),
            username=username,
            full_name=full_name,
            role=Role(role),
            is_active=bool(is_active),
            must_change_password=bool(must_change_password),
            temp_password_expires_at=temp_password_expires_at,
            failed_attempts=int(failed_attempts),
            locked_until=locked_until,
            created_at=created_at,
        )


# ---- SQL Schema & Triggers ---------------------------------------------------------
_AUTH_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS users (
    user_id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    username                 TEXT NOT NULL
                             CHECK (username = lower(trim(username))
                                    AND length(username) BETWEEN 3 AND 32),
    full_name                TEXT NOT NULL
                             CHECK (full_name = trim(full_name, char(32, 9, 10, 13))
                                    AND length(full_name) BETWEEN 1 AND {NAME_MAX_LENGTH}),
    role                     TEXT NOT NULL CHECK (role IN ('Lead Coach', 'Coach')),
    password_hash            TEXT NOT NULL CHECK (length(password_hash) >= 32),
    is_active                INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    must_change_password     INTEGER NOT NULL DEFAULT 0 CHECK (must_change_password IN (0, 1)),
    temp_password_expires_at TEXT,
    failed_attempts          INTEGER NOT NULL DEFAULT 0 CHECK (failed_attempts >= 0),
    locked_until             TEXT,
    created_at               TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username ON users (lower(username));

CREATE TABLE IF NOT EXISTS sessions (
    token_hash   TEXT PRIMARY KEY CHECK (length(token_hash) = 64),
    user_id      INTEGER NOT NULL REFERENCES users(user_id),
    created_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    expires_at   TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS ix_sessions_user_id ON sessions (user_id);

DROP TRIGGER IF EXISTS users_no_delete;
CREATE TRIGGER users_no_delete BEFORE DELETE ON users
BEGIN SELECT RAISE(ABORT, 'users cannot be deleted; deactivate with is_active = 0'); END;

DROP TRIGGER IF EXISTS users_keep_one_active_lead_coach;
CREATE TRIGGER users_keep_one_active_lead_coach
BEFORE UPDATE OF role, is_active ON users
WHEN OLD.role = 'Lead Coach'
  AND OLD.is_active = 1
  AND (NEW.role <> 'Lead Coach' OR NEW.is_active = 0)
  AND NOT EXISTS (
      SELECT 1 FROM users u
      WHERE u.user_id <> OLD.user_id AND u.role = 'Lead Coach' AND u.is_active = 1
  )
BEGIN SELECT RAISE(ABORT, 'cannot deactivate or demote the last active Lead Coach'); END;
"""

_USER_SELECT = (
    "SELECT user_id, username, full_name, role, is_active, must_change_password, "
    "temp_password_expires_at, failed_attempts, locked_until, created_at FROM users"
)

_USER_DF_SCHEMA = {
    "user_id": pl.Int64,
    "username": pl.String,
    "full_name": pl.String,
    "role": pl.String,
    "is_active": pl.Int64,
    "must_change_password": pl.Int64,
    "temp_password_expires_at": pl.String,
    "failed_attempts": pl.Int64,
    "locked_until": pl.String,
    "created_at": pl.String,
}


# ---- AuthRepository ----------------------------------------------------------------
class AuthRepository(Repository):
    """Manages coach/Lead Coach accounts, Argon2id credentials, lockouts, and sessions."""

    def __init__(
        self,
        db: Database,
        *,
        audit: AuditRepository | None = None,
        clock: Callable[[], datetime] = _utc_now,
        hasher: PasswordHasher = _HASHER,
    ) -> None:
        self._audit = audit
        self._clock = clock
        self._hasher = hasher
        super().__init__(db)

    @property
    def schema_sql(self) -> str:
        return _AUTH_SCHEMA_SQL

    def has_any_users(self) -> bool:
        return bool(self._db.query("SELECT 1 FROM users LIMIT 1"))

    def create_initial_lead_coach(self, username: str, full_name: str, password: str) -> User:
        """Bootstrap the very first Lead Coach account (only succeeds when `users` is empty)."""
        if self.has_any_users():
            raise ValidationError("initial Lead Coach already exists; create new users as Lead Coach")
        uname = _clean_username(username)
        fname = _clean_full_name(full_name)
        norm_pw = validate_new_password(password, username=uname)
        pw_hash = self._hasher.hash(norm_pw)
        now_iso = _iso_utc(self._clock())

        with self._db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, full_name, role, password_hash, "
                "is_active, must_change_password, created_at) "
                "VALUES (?, ?, ?, ?, 1, 0, ?)",
                (uname, fname, Role.LEAD_COACH.value, pw_hash, now_iso),
            )
            assert cur.lastrowid is not None
            uid = cur.lastrowid
            if self._audit is not None:
                self._audit.record(
                    "USER_CREATED",
                    actor_user_id=uid,
                    actor_username=uname,
                    target_table="users",
                    target_id=uid,
                    details={"username": uname, "role": Role.LEAD_COACH.value, "bootstrap": True},
                    conn=conn,
                )
        return self.get(uid)

    def create_user(
        self,
        username: str,
        full_name: str,
        role: str | Role,
        temporary_password: str | None = None,
        *,
        actor: User | None = None,
    ) -> tuple[User, str]:
        """Create a new coach/Lead Coach with a 72h expiring one-time temporary password."""
        uname = _clean_username(username)
        fname = _clean_full_name(full_name)
        role_enum = role if isinstance(role, Role) else parse_enum(Role, role, "role")
        temp_pw = (
            generate_temporary_password()
            if temporary_password is None
            else validate_new_password(temporary_password, username=uname)
        )
        pw_hash = self._hasher.hash(temp_pw)
        now = self._clock()
        now_iso = _iso_utc(now)
        expires_iso = _iso_utc(now + TEMP_PASSWORD_TTL)

        try:
            with self._db.transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, full_name, role, password_hash, "
                    "is_active, must_change_password, temp_password_expires_at, created_at) "
                    "VALUES (?, ?, ?, ?, 1, 1, ?, ?)",
                    (uname, fname, role_enum.value, pw_hash, expires_iso, now_iso),
                )
                assert cur.lastrowid is not None
                uid = cur.lastrowid
                if self._audit is not None:
                    self._audit.record(
                        "USER_CREATED",
                        actor_user_id=actor.user_id if actor else None,
                        actor_username=actor.username if actor else "system",
                        target_table="users",
                        target_id=uid,
                        details={
                            "username": uname,
                            "full_name": fname,
                            "role": role_enum.value,
                            "temp_password_expires_at": expires_iso,
                        },
                        conn=conn,
                    )
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorname == "SQLITE_CONSTRAINT_UNIQUE":
                raise DuplicateError(f"username '{uname}' already exists") from None
            raise ValidationError(f"database refused user '{uname}': {exc}") from None

        return self.get(uid), temp_pw

    def login(self, username: str, password: str) -> tuple[User, str]:
        """Authenticate a user and create a hashed 256-bit session token.

        Always returns the identical generic error on failure to prevent username enumeration.
        """
        uname = require_text(username, "username").strip().lower() if isinstance(username, str) else ""
        pw_norm = _normalize_password(password) if isinstance(password, str) else ""
        now = self._clock()
        now_iso = _iso_utc(now)

        rows = self._db.query(
            "SELECT user_id, username, full_name, role, password_hash, is_active, "
            "must_change_password, temp_password_expires_at, failed_attempts, locked_until, created_at "
            "FROM users WHERE lower(username) = lower(?)",
            (uname,),
        )

        # Constant-time defense when username does not exist
        if not rows:
            try:
                self._hasher.verify(_DUMMY_HASH, pw_norm or "x")
            except (VerifyMismatchError, VerificationError, InvalidHashError):
                pass
            if self._audit is not None:
                self._audit.record(
                    "LOGIN_FAILED",
                    actor_username=uname or "unknown",
                    target_table="users",
                    details={"reason": "unknown_username"},
                )
            raise AuthenticationError(GENERIC_LOGIN_ERROR)

        (
            uid, db_uname, _fname, _role, pw_hash, is_active,
            must_change, temp_exp, _failed, locked_until, _created_at,
        ) = rows[0]

        # Inactive account -> generic error + dummy verify to avoid timing leak
        if not is_active:
            try:
                self._hasher.verify(_DUMMY_HASH, pw_norm or "x")
            except (VerifyMismatchError, VerificationError, InvalidHashError):
                pass
            if self._audit is not None:
                self._audit.record(
                    "LOGIN_FAILED",
                    actor_user_id=uid,
                    actor_username=db_uname,
                    target_table="users",
                    target_id=uid,
                    details={"reason": "inactive_account"},
                )
            raise AuthenticationError(GENERIC_LOGIN_ERROR)

        # Currently locked out -> generic error (no enumeration) + dummy verify
        if locked_until is not None and now < _parse_iso_utc(locked_until):
            try:
                self._hasher.verify(_DUMMY_HASH, pw_norm or "x")
            except (VerifyMismatchError, VerificationError, InvalidHashError):
                pass
            if self._audit is not None:
                self._audit.record(
                    "LOGIN_FAILED",
                    actor_user_id=uid,
                    actor_username=db_uname,
                    target_table="users",
                    target_id=uid,
                    details={"reason": "account_locked", "locked_until": locked_until},
                )
            raise AuthenticationError(GENERIC_LOGIN_ERROR)

        # Verify Argon2id password hash
        verified = False
        try:
            verified = self._hasher.verify(pw_hash, pw_norm)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            verified = False

        if not verified:
            with self._db.transaction() as conn:
                # Atomic increment inside transaction to prevent concurrent lockout bypass
                conn.execute(
                    "UPDATE users SET failed_attempts = failed_attempts + 1 WHERE user_id = ?",
                    (uid,),
                )
                (new_attempts,) = conn.execute(
                    "SELECT failed_attempts FROM users WHERE user_id = ?", (uid,)
                ).fetchone()
                lock_iso: str | None = None
                if new_attempts >= MAX_FAILED_ATTEMPTS:
                    lock_iso = _iso_utc(now + LOCKOUT_DURATION)
                    conn.execute(
                        "UPDATE users SET locked_until = ? WHERE user_id = ?",
                        (lock_iso, uid),
                    )
                if self._audit is not None:
                    self._audit.record(
                        "LOGIN_FAILED",
                        actor_user_id=uid,
                        actor_username=db_uname,
                        target_table="users",
                        target_id=uid,
                        details={"reason": "bad_password", "failed_attempts": new_attempts},
                        conn=conn,
                    )
                    if lock_iso is not None:
                        self._audit.record(
                            "ACCOUNT_LOCKED",
                            actor_user_id=uid,
                            actor_username=db_uname,
                            target_table="users",
                            target_id=uid,
                            details={"locked_until": lock_iso, "failed_attempts": new_attempts},
                            conn=conn,
                        )
            raise AuthenticationError(GENERIC_LOGIN_ERROR)

        # Temporary password expiry check (72h window)
        if must_change and temp_exp is not None and now > _parse_iso_utc(temp_exp):
            if self._audit is not None:
                self._audit.record(
                    "LOGIN_FAILED",
                    actor_user_id=uid,
                    actor_username=db_uname,
                    target_table="users",
                    target_id=uid,
                    details={"reason": "temporary_password_expired"},
                )
            raise AuthenticationError(
                "Temporary password has expired; ask the Lead Coach for a new reset"
            )

        # Success! Reset failed_attempts/lockout, transparently rehash if needed, issue session token
        raw_token = secrets.token_urlsafe(32)
        token_hash = _hash_token(raw_token)
        expires_iso = _iso_utc(now + MAX_SESSION_LIFETIME)
        rehashed = self._hasher.hash(pw_norm) if self._hasher.check_needs_rehash(pw_hash) else None

        with self._db.transaction() as conn:
            if rehashed is not None:
                conn.execute(
                    "UPDATE users SET password_hash = ?, failed_attempts = 0, locked_until = NULL "
                    "WHERE user_id = ?",
                    (rehashed, uid),
                )
            else:
                conn.execute(
                    "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE user_id = ?",
                    (uid,),
                )
            conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, last_seen_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (token_hash, uid, now_iso, now_iso, expires_iso),
            )
            if self._audit is not None:
                self._audit.record(
                    "LOGIN_SUCCESS",
                    actor_user_id=uid,
                    actor_username=db_uname,
                    target_table="users",
                    target_id=uid,
                    details={"must_change_password": bool(must_change)},
                    conn=conn,
                )

        return self.get(uid), raw_token

    def validate_session(self, raw_token: str) -> User:
        """Validate a session token (checks active user, 2h idle timeout, 12h max lifetime)."""
        if not isinstance(raw_token, str) or not raw_token:
            raise AuthenticationError("Session is missing or invalid; please log in again")
        token_hash = _hash_token(raw_token)
        rows = self._db.query(
            "SELECT user_id, last_seen_at, expires_at FROM sessions WHERE token_hash = ?",
            (token_hash,),
        )
        if not rows:
            raise AuthenticationError("Session is invalid or expired; please log in again")
        uid, last_seen_str, expires_str = rows[0]
        now = self._clock()
        last_seen = _parse_iso_utc(last_seen_str)
        expires_at = _parse_iso_utc(expires_str)

        if now > expires_at or (now - last_seen) > IDLE_TIMEOUT:
            with self._db.transaction() as conn:
                conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            raise AuthenticationError("Session expired due to timeout; please log in again")

        user = self.get(uid)
        if not user.is_active:
            with self._db.transaction() as conn:
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
            raise AuthenticationError("Account is inactive; please contact the Lead Coach")

        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE token_hash = ?",
                (_iso_utc(now), token_hash),
            )
        return user

    def logout(self, raw_token: str) -> None:
        """Invalidate one session token."""
        if not isinstance(raw_token, str) or not raw_token:
            return
        token_hash = _hash_token(raw_token)
        rows = self._db.query(
            "SELECT s.user_id, u.username FROM sessions s JOIN users u ON u.user_id = s.user_id "
            "WHERE s.token_hash = ?",
            (token_hash,),
        )
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
            if rows and self._audit is not None:
                uid, uname = rows[0]
                self._audit.record(
                    "LOGOUT",
                    actor_user_id=uid,
                    actor_username=uname,
                    target_table="sessions",
                    target_id=uid,
                    conn=conn,
                )

    def revoke_all_sessions(
        self,
        user_id: int,
        *,
        actor: User | None = None,
        reason: str = "manual",
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Revoke all active sessions for `user_id`."""
        target = self.get(user_id)

        def _do(c: sqlite3.Connection) -> int:
            cur = c.execute("DELETE FROM sessions WHERE user_id = ?", (target.user_id,))
            if self._audit is not None:
                self._audit.record(
                    "SESSIONS_REVOKED",
                    actor_user_id=actor.user_id if actor else target.user_id,
                    actor_username=actor.username if actor else target.username,
                    target_table="users",
                    target_id=target.user_id,
                    details={"reason": reason, "revoked_count": cur.rowcount},
                    conn=c,
                )
            return cur.rowcount

        if conn is not None:
            return _do(conn)
        with self._db.transaction() as tx_conn:
            return _do(tx_conn)

    def change_password(self, user_id: int, current_password: str, new_password: str) -> User:
        """Verify current password, enforce policy & non-reuse, update hash, revoke all sessions."""
        rows = self._db.query(
            "SELECT username, password_hash, is_active FROM users WHERE user_id = ?",
            (user_id,),
        )
        if not rows:
            raise NotFoundError(f"user_id {user_id} does not exist")
        uname, current_hash, is_active = rows[0]
        if not is_active:
            raise AuthenticationError("Account is inactive")

        cur_norm = _normalize_password(current_password)
        try:
            self._hasher.verify(current_hash, cur_norm)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            raise AuthenticationError("Current password is incorrect") from None

        new_norm = validate_new_password(new_password, username=uname)
        if new_norm == cur_norm:
            raise ValidationError("new password must be different from your current/temporary password")

        new_hash = self._hasher.hash(new_norm)
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, must_change_password = 0, "
                "temp_password_expires_at = NULL, failed_attempts = 0, locked_until = NULL "
                "WHERE user_id = ?",
                (new_hash, user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            if self._audit is not None:
                self._audit.record(
                    "PASSWORD_CHANGED",
                    actor_user_id=user_id,
                    actor_username=uname,
                    target_table="users",
                    target_id=user_id,
                    details={"sessions_revoked": True},
                    conn=conn,
                )
        return self.get(user_id)

    def reset_password(
        self,
        target_user_id: int,
        temporary_password: str | None = None,
        *,
        actor: User | None = None,
        break_glass: bool = False,
    ) -> str:
        """Issue a 72h expiring one-time temporary password, unlock account, and revoke sessions."""
        target = self.get(target_user_id)
        temp_pw = (
            generate_temporary_password()
            if temporary_password is None
            else validate_new_password(temporary_password, username=target.username)
        )
        pw_hash = self._hasher.hash(temp_pw)
        expires_iso = _iso_utc(self._clock() + TEMP_PASSWORD_TTL)

        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE users SET password_hash = ?, must_change_password = 1, "
                "temp_password_expires_at = ?, failed_attempts = 0, locked_until = NULL "
                "WHERE user_id = ?",
                (pw_hash, expires_iso, target.user_id),
            )
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (target.user_id,))
            if self._audit is not None:
                self._audit.record(
                    "BREAK_GLASS_PASSWORD_RESET" if break_glass else "PASSWORD_RESET",
                    actor_user_id=actor.user_id if actor else None,
                    actor_username=actor.username if actor else "host-cli",
                    target_table="users",
                    target_id=target.user_id,
                    details={
                        "target_username": target.username,
                        "temp_password_expires_at": expires_iso,
                        "break_glass": break_glass,
                    },
                    conn=conn,
                )
        return temp_pw

    def unlock_user(self, target_user_id: int, *, actor: User | None = None) -> User:
        """Immediately clear brute-force lockout for a user."""
        target = self.get(target_user_id)
        with self._db.transaction() as conn:
            conn.execute(
                "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE user_id = ?",
                (target.user_id,),
            )
            if self._audit is not None:
                self._audit.record(
                    "ACCOUNT_UNLOCKED",
                    actor_user_id=actor.user_id if actor else None,
                    actor_username=actor.username if actor else "system",
                    target_table="users",
                    target_id=target.user_id,
                    details={"target_username": target.username},
                    conn=conn,
                )
        return self.get(target.user_id)

    def set_active(self, target_user_id: int, active: bool, *, actor: User | None = None) -> User:
        """Activate or deactivate a user (deactivating revokes all active sessions)."""
        if not isinstance(active, bool):
            raise ValidationError("active must be True or False")
        target = self.get(target_user_id)
        try:
            with self._db.transaction() as conn:
                conn.execute(
                    "UPDATE users SET is_active = ? WHERE user_id = ?",
                    (int(active), target.user_id),
                )
                if not active:
                    conn.execute("DELETE FROM sessions WHERE user_id = ?", (target.user_id,))
                if self._audit is not None:
                    self._audit.record(
                        "USER_ACTIVE_CHANGED",
                        actor_user_id=actor.user_id if actor else None,
                        actor_username=actor.username if actor else "system",
                        target_table="users",
                        target_id=target.user_id,
                        details={
                            "target_username": target.username,
                            "before": target.is_active,
                            "after": active,
                        },
                        conn=conn,
                    )
        except sqlite3.IntegrityError as exc:
            raise ValidationError(str(exc)) from None
        return self.get(target.user_id)

    def set_role(self, target_user_id: int, role: str | Role, *, actor: User | None = None) -> User:
        """Promote or demote a user; always revokes all active sessions for that user."""
        target = self.get(target_user_id)
        new_role = role if isinstance(role, Role) else parse_enum(Role, role, "role")
        try:
            with self._db.transaction() as conn:
                conn.execute(
                    "UPDATE users SET role = ? WHERE user_id = ?",
                    (new_role.value, target.user_id),
                )
                conn.execute("DELETE FROM sessions WHERE user_id = ?", (target.user_id,))
                if self._audit is not None:
                    self._audit.record(
                        "USER_ROLE_CHANGED",
                        actor_user_id=actor.user_id if actor else None,
                        actor_username=actor.username if actor else "system",
                        target_table="users",
                        target_id=target.user_id,
                        details={
                            "target_username": target.username,
                            "before": target.role.value,
                            "after": new_role.value,
                            "sessions_revoked": True,
                        },
                        conn=conn,
                    )
        except sqlite3.IntegrityError as exc:
            raise ValidationError(str(exc)) from None
        return self.get(target.user_id)

    def get(self, user_id: int) -> User:
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
            raise ValidationError("user_id must be a positive whole number")
        rows = self._db.query(_USER_SELECT + " WHERE user_id = ?", (user_id,))
        if not rows:
            raise NotFoundError(f"user_id {user_id} does not exist")
        return User.from_row(rows[0])

    def get_by_username(self, username: str) -> User:
        uname = _clean_username(username)
        rows = self._db.query(_USER_SELECT + " WHERE lower(username) = lower(?)", (uname,))
        if not rows:
            raise NotFoundError(f"username '{uname}' does not exist")
        return User.from_row(rows[0])

    def all(self, *, active_only: bool = False) -> list[User]:
        where = " WHERE is_active = 1" if active_only else ""
        return [
            User.from_row(r)
            for r in self._db.query(_USER_SELECT + where + " ORDER BY role, username")
        ]

    def to_dataframe(self, *, active_only: bool = False) -> pl.DataFrame:
        where = " WHERE is_active = 1" if active_only else ""
        rows = self._db.query(_USER_SELECT + where + " ORDER BY role, username")
        df = pl.DataFrame(rows, schema=_USER_DF_SCHEMA, orient="row")
        return df.with_columns(
            pl.col("role").cast(pl.Enum([r.value for r in Role])),
            pl.col("is_active").cast(pl.Boolean),
            pl.col("must_change_password").cast(pl.Boolean),
            pl.col("created_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"),
        )


# ---- Break-Glass Host CLI ----------------------------------------------------------
def break_glass_reset_lead_coach(
    db: Database,
    username: str,
    *,
    clock: Callable[[], datetime] = _utc_now,
) -> str:
    """Host-only recovery for a locked-out or forgotten Lead Coach password (audited)."""
    audit = AuditRepository(db, clock=clock)
    auth = AuthRepository(db, audit=audit, clock=clock)
    user = auth.get_by_username(username)
    if not user.is_lead_coach:
        raise ValidationError(f"user '{user.username}' is not a Lead Coach")
    if not user.is_active:
        auth.set_active(user.user_id, True)
    return auth.reset_password(user.user_id, break_glass=True)


def _cli(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "reset-lead-coach":
        with Database(DEFAULT_DB_PATH) as db:
            temp_pw = break_glass_reset_lead_coach(db, argv[2])
        print(f"Break-glass reset succeeded for Lead Coach '{argv[2]}'.")
        print(f"One-time temporary password (valid 72h, must change on login): {temp_pw}")
        return 0
    print("Usage: uv run python -m whc.auth reset-lead-coach <username>")
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
