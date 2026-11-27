"""tests/test_services.py - end-to-end RBAC, audit trail, backup & method-coverage tests."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from argon2 import PasswordHasher

from whc.auth import Role
from whc.database import Database
from whc.exceptions import AuthenticationError, AuthorizationError
from whc.services import WhcService
from whc.test_results import TestResult

FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 11, 22, 9, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


def test_every_public_service_method_is_registered_in_permissions_map():
    """Guardrail: impossible to add a new public WhcService method and forget RBAC."""
    public_methods = {
        name
        for name, _fn in inspect.getmembers(WhcService, predicate=inspect.isfunction)
        if not name.startswith("_")
    }
    assert public_methods == set(WhcService.PERMISSIONS.keys())


@pytest.fixture
def env(tmp_path: Path):
    clock = FakeClock()
    with Database(tmp_path / "whc_service.db") as db:
        svc = WhcService(
            db,
            backup_dir=tmp_path / "backups",
            clock=clock,
            hasher=FAST_HASHER,
            enable_auto_backups=True,
        )
        lead = svc.bootstrap_lead_coach("lead1", "Alex Lead", "Lead-Coach-Pass-2026!")
        _, lead_tok = svc.login("lead1", "Lead-Coach-Pass-2026!")
        coach, temp_pw = svc.create_user(lead_tok, "coach1", "Sam Coach", Role.COACH)
        _, temp_tok = svc.login("coach1", temp_pw)
        yield svc, lead_tok, temp_tok, temp_pw, lead, coach, clock


def test_must_change_password_blocks_dashboard_until_password_updated(env):
    svc, _lead_tok, temp_tok, temp_pw, _lead, _coach, _clock = env

    # current_user works, but listing students is blocked until temp password is changed
    assert svc.current_user(temp_tok).must_change_password is True
    with pytest.raises(AuthenticationError, match="change your temporary password"):
        svc.list_students(temp_tok)

    svc.change_my_password(temp_tok, temp_pw, "Coach-New-Passphrase-2026!")
    _, fresh_tok = svc.login("coach1", "Coach-New-Passphrase-2026!")
    assert svc.list_students(fresh_tok) == []


def test_broad_coach_access_matrix_and_lead_only_enforcement(env):
    svc, lead_tok, temp_tok, temp_pw, _lead, coach, _clock = env
    svc.change_my_password(temp_tok, temp_pw, "Coach-New-Passphrase-2026!")
    _, coach_tok = svc.login("coach1", "Coach-New-Passphrase-2026!")

    # ---- Coach CAN manage students, enrolments, and test results ----
    s = svc.add_student(coach_tok, "AAA0001", "Jhon", "Lee", "14/03/2012", "Male")
    assert s.student_id == "AAA0001"
    fixed = svc.update_student(coach_tok, "AAA0001", first_name="John")
    assert fixed.first_name == "John"

    enr = svc.add_enrolment(coach_tok, "AAA0001", 2026, "AFL", "Year 8")
    assert enr.enrolment_id is not None

    # Batch entry triggers an automatic pre_bulk backup
    saved_batch = svc.add_results_batch(
        coach_tok,
        [
            TestResult.new("AAA0001", 902, "10/03/2026", 65.0),
            TestResult.new("AAA0001", 1, "10/03/2026", 2000.0),
        ],
    )
    assert len(saved_batch) == 2

    # IMTP soft >20% warning works for coaches without blocking save
    warn = svc.check_imtp_warning(coach_tok, "AAA0001", "10/03/2026", relative_score=1.8)
    assert warn is not None and "Warning: IMTP Relative" in warn
    r_rel = svc.add_result(coach_tok, "AAA0001", 2, "10/03/2026", 1.8)
    assert r_rel.result_id is not None

    # Coach can edit and delete results; full deleted row is preserved in audit_log!
    svc.update_result_score(coach_tok, r_rel.result_id, 3.1)
    svc.delete_result(coach_tok, r_rel.result_id)

    # ---- Coach CANNOT perform Lead Coach-only admin/catalogue/audit/backup actions ----
    lead_only_calls = [
        lambda: svc.add_sport(coach_tok, "Futsal"),
        lambda: svc.add_category(coach_tok, "Balance"),
        lambda: svc.add_unit(coach_tok, "Degrees"),
        lambda: svc.add_fitness_test(coach_tok, "40 Metre Sprint", "Speed", "Seconds", 4.0, 10.0),
        lambda: svc.set_fitness_test_range(coach_tok, 101, 5, 95),
        lambda: svc.list_users(coach_tok),
        lambda: svc.create_user(coach_tok, "coach2", "Other Coach", Role.COACH),
        lambda: svc.recent_audit_entries(coach_tok),
        lambda: svc.verify_audit_chain(coach_tok),
        lambda: svc.create_manual_backup(coach_tok),
        lambda: svc.list_backups(coach_tok),
    ]
    for call in lead_only_calls:
        with pytest.raises(AuthorizationError):
            call()

    # ---- Lead Coach CAN perform all admin, catalogue, audit & backup actions ----
    assert svc.add_sport(lead_tok, "Futsal") == "Futsal"
    cat_name, block = svc.add_category(lead_tok, "Balance")
    assert (cat_name, block) == ("Balance", 1000)
    assert svc.add_unit(lead_tok, "Degrees") == "Degrees"
    new_test = svc.add_fitness_test(
        lead_tok, "Ankle Dorsiflexion", "Balance", "Degrees", 0, 60, score_direction="higher"
    )
    assert new_test.test_id == 1001

    # Verify audit log captured AUTHZ_DENIED, full RESULT_DELETED snapshot, and unbroken hash chain
    ok, _ = svc.verify_audit_chain(lead_tok)
    assert ok is True
    entries = svc.recent_audit_entries(lead_tok, limit=100)
    actions = {e.action for e in entries}
    assert {"STUDENT_ADDED", "STUDENT_UPDATED", "RESULTS_BATCH_ADDED", "RESULT_DELETED", "AUTHZ_DENIED"} <= actions

    deleted_entry = next(e for e in entries if e.action == "RESULT_DELETED")
    assert deleted_entry.details["deleted"]["value"] == 3.1
    assert deleted_entry.details["deleted"]["student_id"] == "AAA0001"

    # Verify daily + pre_bulk + manual backups all exist
    svc.create_manual_backup(lead_tok)
    reasons = {b.reason for b in svc.list_backups(lead_tok)}
    assert {"daily", "pre_bulk", "manual"} <= reasons
