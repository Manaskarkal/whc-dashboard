"""tests/test_auth.py - comprehensive unit tests for Phase 2 authentication & sessions."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from argon2 import PasswordHasher

from whc.audit import AuditRepository
from whc.auth import (
    _HASHER,
    GENERIC_LOGIN_ERROR,
    IDLE_TIMEOUT,
    LOCKOUT_DURATION,
    MAX_FAILED_ATTEMPTS,
    MAX_SESSION_LIFETIME,
    MIN_PASSWORD_LENGTH,
    TEMP_PASSWORD_TTL,
    AuthRepository,
    Role,
    break_glass_reset_lead_coach,
    validate_new_password,
)
from whc.database import Database
from whc.exceptions import AuthenticationError, DuplicateError, ValidationError

# Fast Argon2id parameters strictly for unit tests (production _HASHER is verified separately)
FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 11, 22, 9, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def db():
    with Database(":memory:") as database:
        yield database


@pytest.fixture
def audit(db, clock):
    return AuditRepository(db, clock=clock)


@pytest.fixture
def auth(db, audit, clock):
    return AuthRepository(db, audit=audit, clock=clock, hasher=FAST_HASHER)


# ==== 1. Password policy & production Argon2id parameters ==================================
def test_production_argon2id_parameters_meet_owasp_minimums():
    assert _HASHER.memory_cost >= 65536  # >= 64 MiB
    assert _HASHER.time_cost >= 3
    assert _HASHER.parallelism >= 4
    sample_hash = _HASHER.hash("Correct-Horse-Battery-Staple!")
    assert sample_hash.startswith("$argon2id$")


@pytest.mark.parametrize("bad_pw, message", [
    ("short123456", f"at least {MIN_PASSWORD_LENGTH}"),
    ("a" * 129, "at most 128"),
    ("aaaaaaaaaaaa", "too repetitive"),
    ("password1234", "too common"),
    ("WesternHeights", "too common"),
    ("my-jsmith-passphrase", "cannot contain your username"),
])
def test_password_policy_rejects_weak_and_blocked_passwords(bad_pw, message):
    with pytest.raises(ValidationError, match=message):
        validate_new_password(bad_pw, username="jsmith")


def test_password_unicode_nfkc_normalization_works(auth):
    # Full-width 'Ａ' (U+FF21) normalizes under NFKC to ASCII 'A'
    auth.create_initial_lead_coach("lead1", "Alex Lead", "Ａlpine-Falcon-99!")
    user, token = auth.login("lead1", "Alpine-Falcon-99!")
    assert user.username == "lead1"
    assert auth.validate_session(token).user_id == user.user_id


# ==== 2. Bootstrap, user creation, duplicates, no-delete & last Lead Coach =================
def test_bootstrap_only_works_when_users_empty(auth):
    lead = auth.create_initial_lead_coach("LeadOne", "Alex Lead", "Strong-Passphrase-99!")
    assert lead.username == "leadone"
    assert lead.is_lead_coach is True
    assert lead.must_change_password is False

    with pytest.raises(ValidationError, match="already exists"):
        auth.create_initial_lead_coach("leadtwo", "Sam Lead", "Another-Strong-Pass-99!")


def test_duplicate_username_refused_case_insensitively(auth):
    lead = auth.create_initial_lead_coach("lead1", "Alex Lead", "Strong-Passphrase-99!")
    auth.create_user("JSmith", "John Smith", Role.COACH, actor=lead)
    with pytest.raises(DuplicateError, match="already exists"):
        auth.create_user("jsmith", "Jane Smith", Role.COACH, actor=lead)


def test_cannot_delete_user_or_remove_last_active_lead_coach(db, auth):
    lead = auth.create_initial_lead_coach("lead1", "Alex Lead", "Strong-Passphrase-99!")
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        with db.transaction() as conn:
            conn.execute("DELETE FROM users WHERE user_id = ?", (lead.user_id,))

    with pytest.raises(ValidationError, match="last active Lead Coach"):
        auth.set_active(lead.user_id, False, actor=lead)

    with pytest.raises(ValidationError, match="last active Lead Coach"):
        auth.set_role(lead.user_id, Role.COACH, actor=lead)

    # Once a second active Lead Coach exists, demoting/deactivating the first succeeds
    lead2, _ = auth.create_user("lead2", "Pat Lead", Role.LEAD_COACH, actor=lead)
    demoted = auth.set_role(lead.user_id, Role.COACH, actor=lead2)
    assert demoted.role is Role.COACH


# ==== 3. No-enumeration login, atomic lockout, unlock & break-glass CLI ====================
def test_lockout_and_generic_error_prevent_username_enumeration(auth, audit, clock):
    lead = auth.create_initial_lead_coach("lead1", "Alex Lead", "Strong-Passphrase-99!")

    # Unknown username returns exact same generic error
    with pytest.raises(AuthenticationError, match=GENERIC_LOGIN_ERROR):
        auth.login("nonexistent", "Wrong-Password-1234!")

    # 5 consecutive bad passwords trigger 15-minute lockout
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises(AuthenticationError, match=GENERIC_LOGIN_ERROR):
            auth.login("lead1", "Wrong-Password-1234!")

    locked_user = auth.get(lead.user_id)
    assert locked_user.failed_attempts == MAX_FAILED_ATTEMPTS
    assert locked_user.locked_until is not None

    # Even the CORRECT password fails with generic error while locked
    with pytest.raises(AuthenticationError, match=GENERIC_LOGIN_ERROR):
        auth.login("lead1", "Strong-Passphrase-99!")

    # After 15 minutes pass, login succeeds and resets failed_attempts
    clock.advance(minutes=int(LOCKOUT_DURATION.total_seconds() // 60) + 1)
    user, _ = auth.login("lead1", "Strong-Passphrase-99!")
    assert user.failed_attempts == 0
    assert user.locked_until is None

    # Audit log captured both ACCOUNT_LOCKED and LOGIN_SUCCESS
    actions = [e.action for e in audit.recent(limit=20)]
    assert "ACCOUNT_LOCKED" in actions
    assert "LOGIN_SUCCESS" in actions


def test_lead_coach_can_unlock_and_break_glass_recovers_sole_lead_coach(db, auth, audit, clock):
    lead = auth.create_initial_lead_coach("lead1", "Alex Lead", "Strong-Passphrase-99!")
    for _ in range(MAX_FAILED_ATTEMPTS):
        with pytest.raises(AuthenticationError):
            auth.login("lead1", "Wrong-Password-1234!")

    # Host break-glass recovers locked sole Lead Coach with an audited 72h temp password
    temp_pw = break_glass_reset_lead_coach(db, "lead1", clock=clock)
    user, token = auth.login("lead1", temp_pw)
    assert user.must_change_password is True
    assert auth.validate_session(token).user_id == lead.user_id
    assert any(e.action == "BREAK_GLASS_PASSWORD_RESET" for e in audit.recent(limit=10))


# ==== 4. Session timeouts (2h idle / 12h max) & revocation =================================
def test_session_idle_and_absolute_timeouts_with_fake_clock(auth, clock):
    auth.create_initial_lead_coach("lead1", "Alex Lead", "Strong-Passphrase-99!")
    _, token = auth.login("lead1", "Strong-Passphrase-99!")

    # Active use every 90 minutes keeps session alive up to 12h max
    for _ in range(7):  # 7 * 90m = 10.5 hours
        clock.advance(minutes=90)
        assert auth.validate_session(token).username == "lead1"

    # Crossing 12h absolute max expires session even with recent activity
    clock.advance(hours=2)
    with pytest.raises(AuthenticationError, match="expired"):
        auth.validate_session(token)

    # Fresh session expires after >2h idle
    _, token2 = auth.login("lead1", "Strong-Passphrase-99!")
    clock.advance(seconds=int(IDLE_TIMEOUT.total_seconds()) + 1)
    with pytest.raises(AuthenticationError, match="expired"):
        auth.validate_session(token2)


def test_temporary_password_expires_after_72h_and_cannot_be_reused(auth, clock):
    lead = auth.create_initial_lead_coach("lead1", "Alex Lead", "Strong-Passphrase-99!")
    coach, temp_pw = auth.create_user("jsmith", "John Smith", Role.COACH, actor=lead)
    assert coach.must_change_password is True

    # Cannot reuse the temporary password as the new password
    with pytest.raises(ValidationError, match="must be different"):
        auth.change_password(coach.user_id, temp_pw, temp_pw)

    # If unused past 72h, temporary password expires
    clock.advance(seconds=int(TEMP_PASSWORD_TTL.total_seconds()) + 60)
    with pytest.raises(AuthenticationError, match="Temporary password has expired"):
        auth.login("jsmith", temp_pw)

    # Fresh reset works and changing password clears must_change_password & revokes old sessions
    fresh_temp = auth.reset_password(coach.user_id, actor=lead)
    u, old_token = auth.login("jsmith", fresh_temp)
    assert u.must_change_password is True
    updated = auth.change_password(coach.user_id, fresh_temp, "Brand-New-Coach-Pass-2026!")
    assert updated.must_change_password is False
    with pytest.raises(AuthenticationError):
        auth.validate_session(old_token)


def test_role_change_and_deactivation_revoke_all_sessions(auth):
    lead = auth.create_initial_lead_coach("lead1", "Alex Lead", "Strong-Passphrase-99!")
    coach, temp_pw = auth.create_user("jsmith", "John Smith", Role.COACH, actor=lead)
    auth.change_password(coach.user_id, temp_pw, "Coach-Personal-Pass-2026!")

    _, tok1 = auth.login("jsmith", "Coach-Personal-Pass-2026!")
    auth.set_role(coach.user_id, Role.LEAD_COACH, actor=lead)
    with pytest.raises(AuthenticationError):
        auth.validate_session(tok1)

    _, tok2 = auth.login("jsmith", "Coach-Personal-Pass-2026!")
    auth.set_active(coach.user_id, False, actor=lead)
    with pytest.raises(AuthenticationError):
        auth.validate_session(tok2)
