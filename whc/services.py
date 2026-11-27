"""services.py - Phase 2: Authenticated, RBAC-enforced, audited service facade.

`WhcService` is the single entry point for the Streamlit dashboard and data-entry UI
(Phases 4 & 5). No UI page should bypass `WhcService` to mutate repositories directly.

Key Guarantees:
    * Declarative RBAC: every public method MUST be registered in `WhcService.PERMISSIONS`
      (enforced by unit tests — impossible to add an unprotected method by accident).
    * Broad Coach Access:
        - `Coach` + `Lead Coach`: view all data, manage students/enrolments, record/edit/delete
          test results, check IMTP >20% soft warnings.
        - `Lead Coach` ONLY: manage user accounts, manage sports/categories/units/fitness tests
          and ranges, read audit logs, verify audit hash chain, and create/list backups.
    * `must_change_password` gate: users logging in with a one-time temporary password can ONLY
      call `current_user()` and `change_my_password()` until they set a new personal password.
    * Automatic audit logging (including `AUTHZ_DENIED` and full-row snapshots on deletes) and
      automatic daily + pre-bulk WAL-safe SQLite backups.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from argon2 import PasswordHasher

from whc.analytics import (
    DEFAULT_MIN_COHORT_SIZE,
    Basis,
    SeasonFilter,
    asymmetry_report as _asymmetry_report,
    athlete_category_profile as _athlete_category_profile,
    athlete_percentiles as _athlete_percentiles,
    athlete_personal_bests as _athlete_personal_bests,
    athlete_trends as _athlete_trends,
    build_enriched_results as _build_enriched_results,
    cohort_benchmarks as _cohort_benchmarks,
    compare_athletes as _compare_athletes,
    dashboard_overview as _dashboard_overview,
    relative_strength_metrics as _relative_strength_metrics,
    squad_leaderboard as _squad_leaderboard,
)
from whc.audit import (
    DEFAULT_BACKUP_DIR,
    AuditEntry,
    AuditRepository,
    BackupInfo,
    BackupManager,
)
from whc.auth import _HASHER, AuthRepository, Role, User
from whc.database import Database
from whc.enrolments import Enrolment, EnrolmentRepository
from whc.exceptions import AuthenticationError, AuthorizationError
from whc.fitness_tests import FitnessTest, FitnessTestRepository, ScoreDirection
from whc.students import Sex, Student, StudentRepository
from whc.test_results import TestResult, TestResultRepository

ANY_AUTHENTICATED: frozenset[Role] = frozenset({Role.LEAD_COACH, Role.COACH})
LEAD_ONLY: frozenset[Role] = frozenset({Role.LEAD_COACH})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _student_dict(s: Student) -> dict[str, Any]:
    return {
        "student_id": s.student_id,
        "first_name": s.first_name,
        "last_name": s.last_name,
        "dob": s.dob.isoformat() if s.dob else None,
        "sex": s.sex.value,
        "is_active": s.is_active,
    }


def _enrolment_dict(e: Enrolment) -> dict[str, Any]:
    return {
        "enrolment_id": e.enrolment_id,
        "student_id": e.student_id,
        "season_year": e.season_year,
        "sport": e.sport,
        "grade": e.grade.value,
    }


def _result_dict(r: TestResult) -> dict[str, Any]:
    return {
        "result_id": r.result_id,
        "student_id": r.student_id,
        "test_id": r.test_id,
        "test_date": r.test_date.isoformat() if r.test_date else None,
        "legacy_year": r.legacy_year,
        "legacy_semester": r.legacy_semester,
        "value": r.value,
        "level": r.level,
        "shuttle": r.shuttle,
    }


class WhcService:
    """Single authenticated, RBAC-enforced, audited facade over all WHC repositories."""

    # Declarative permission map: EVERY public method MUST be listed here.
    # Value is `None` for public/unauthenticated methods, or a frozenset of allowed Roles.
    PERMISSIONS: dict[str, frozenset[Role] | None] = {
        # Public / Auth lifecycle
        "has_any_users": None,
        "bootstrap_lead_coach": None,
        "login": None,
        "logout": None,
        "current_user": ANY_AUTHENTICATED,
        "change_my_password": ANY_AUTHENTICATED,
        # Students (Broad Coach Access)
        "list_students": ANY_AUTHENTICATED,
        "students_dataframe": ANY_AUTHENTICATED,
        "add_student": ANY_AUTHENTICATED,
        "update_student": ANY_AUTHENTICATED,
        "set_student_active": ANY_AUTHENTICATED,
        # Enrolments (Broad Coach Access; adding sports is Lead Coach only)
        "list_enrolments": ANY_AUTHENTICATED,
        "enrolments_dataframe": ANY_AUTHENTICATED,
        "list_sports": ANY_AUTHENTICATED,
        "add_enrolment": ANY_AUTHENTICATED,
        "update_enrolment_grade": ANY_AUTHENTICATED,
        "delete_enrolment": ANY_AUTHENTICATED,
        "add_sport": LEAD_ONLY,
        # Fitness Test Catalogue (Reads for all; structural edits Lead Coach only)
        "list_fitness_tests": ANY_AUTHENTICATED,
        "fitness_tests_dataframe": ANY_AUTHENTICATED,
        "list_categories": ANY_AUTHENTICATED,
        "list_units": ANY_AUTHENTICATED,
        "add_category": LEAD_ONLY,
        "add_unit": LEAD_ONLY,
        "add_fitness_test": LEAD_ONLY,
        "rename_fitness_test": LEAD_ONLY,
        "set_fitness_test_range": LEAD_ONLY,
        "set_level_shuttle_rules": LEAD_ONLY,
        # Test Results (Broad Coach Access)
        "list_results": ANY_AUTHENTICATED,
        "results_dataframe": ANY_AUTHENTICATED,
        "check_imtp_warning": ANY_AUTHENTICATED,
        "add_result": ANY_AUTHENTICATED,
        "add_results_batch": ANY_AUTHENTICATED,
        "update_result_score": ANY_AUTHENTICATED,
        "delete_result": ANY_AUTHENTICATED,
        # Headless Analytics (Phase 3 — Read-only, Broad Coach Access)
        "enriched_results_dataframe": ANY_AUTHENTICATED,
        "athlete_trends": ANY_AUTHENTICATED,
        "athlete_personal_bests": ANY_AUTHENTICATED,
        "cohort_benchmarks": ANY_AUTHENTICATED,
        "athlete_percentiles": ANY_AUTHENTICATED,
        "athlete_category_profile": ANY_AUTHENTICATED,
        "asymmetry_report": ANY_AUTHENTICATED,
        "relative_strength_metrics": ANY_AUTHENTICATED,
        "squad_leaderboard": ANY_AUTHENTICATED,
        "compare_athletes": ANY_AUTHENTICATED,
        "dashboard_overview": ANY_AUTHENTICATED,
        # Lead Coach Administration (Users, Audit Log, Backups)
        "list_users": LEAD_ONLY,
        "create_user": LEAD_ONLY,
        "reset_user_password": LEAD_ONLY,
        "unlock_user": LEAD_ONLY,
        "set_user_active": LEAD_ONLY,
        "set_user_role": LEAD_ONLY,
        "recent_audit_entries": LEAD_ONLY,
        "audit_dataframe": LEAD_ONLY,
        "verify_audit_chain": LEAD_ONLY,
        "create_manual_backup": LEAD_ONLY,
        "list_backups": LEAD_ONLY,
    }

    def __init__(
        self,
        db: Database,
        *,
        backup_dir: Path | str = DEFAULT_BACKUP_DIR,
        clock: Callable[[], datetime] = _utc_now,
        hasher: PasswordHasher = _HASHER,
        enable_auto_backups: bool = True,
    ) -> None:
        self._db = db
        self._clock = clock
        self._enable_auto_backups = enable_auto_backups
        self._audit = AuditRepository(db, clock=clock)
        self._auth = AuthRepository(db, audit=self._audit, clock=clock, hasher=hasher)
        self._backups = BackupManager(db, backup_dir=backup_dir, clock=clock)
        self._students = StudentRepository(db)
        self._enrolments = EnrolmentRepository(db)
        self._tests = FitnessTestRepository(db)
        self._tests.load_catalogue()
        self._results = TestResultRepository(db)
        # Analytics snapshot cache: keyed on (PRAGMA data_version, _write_version)
        self._write_version: int = 0
        self._snapshot_cache: (
            tuple[
                int,
                int,
                tuple[
                    pl.DataFrame,
                    pl.DataFrame,
                    pl.DataFrame,
                    pl.DataFrame,
                    dict[int, dict[int, int]],
                ],
            ]
            | None
        ) = None

    def _bump_cache(self) -> None:
        self._write_version += 1
        self._snapshot_cache = None

    def _analytics_snapshot(
        self,
    ) -> tuple[
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        pl.DataFrame,
        dict[int, dict[int, int]],
    ]:
        """Return cached `(students_df, enrolments_df, tests_df, results_df, shuttle_rules)`.

        Invalidates automatically whenever this service mutates data (`_write_version`) OR
        another connection writes to SQLite (`PRAGMA data_version`).
        """
        (data_ver,) = self._db.query("PRAGMA data_version")[0]
        key = (int(data_ver), self._write_version)
        if self._snapshot_cache is not None and (
            self._snapshot_cache[0],
            self._snapshot_cache[1],
        ) == key:
            return self._snapshot_cache[2]
        payload = (
            self._students.to_dataframe(),
            self._enrolments.to_dataframe(),
            self._tests.to_dataframe(),
            self._results.to_dataframe(),
            self._tests.all_level_shuttle_rules(),
        )
        self._snapshot_cache = (key[0], key[1], payload)
        return payload

    # ---- Authorization helper ------------------------------------------------------
    def _authorize(
        self,
        session_token: str,
        method_name: str,
        *,
        allow_during_password_reset: bool = False,
    ) -> User:
        user = self._auth.validate_session(session_token)
        if user.must_change_password and not allow_during_password_reset:
            raise AuthenticationError(
                "You must change your temporary password before accessing the dashboard"
            )
        allowed_roles = self.PERMISSIONS[method_name]
        assert allowed_roles is not None
        if user.role not in allowed_roles:
            self._audit.record(
                "AUTHZ_DENIED",
                actor_user_id=user.user_id,
                actor_username=user.username,
                target_table="service",
                target_id=method_name,
                details={"role": user.role.value, "required": sorted(r.value for r in allowed_roles)},
            )
            raise AuthorizationError(
                f"Action '{method_name}' requires Lead Coach privileges"
            )
        return user

    def _daily_backup(self, actor: User) -> None:
        self._bump_cache()
        if self._enable_auto_backups:
            self._backups.ensure_daily_backup(
                audit=self._audit,
                actor_user_id=actor.user_id,
                actor_username=actor.username,
            )

    # ---- Public / Auth lifecycle ---------------------------------------------------
    def has_any_users(self) -> bool:
        return self._auth.has_any_users()

    def bootstrap_lead_coach(self, username: str, full_name: str, password: str) -> User:
        return self._auth.create_initial_lead_coach(username, full_name, password)

    def login(self, username: str, password: str) -> tuple[User, str]:
        return self._auth.login(username, password)

    def logout(self, session_token: str) -> None:
        self._auth.logout(session_token)

    def current_user(self, session_token: str) -> User:
        return self._authorize(session_token, "current_user", allow_during_password_reset=True)

    def change_my_password(
        self, session_token: str, current_password: str, new_password: str
    ) -> User:
        actor = self._authorize(
            session_token, "change_my_password", allow_during_password_reset=True
        )
        return self._auth.change_password(actor.user_id, current_password, new_password)

    # ---- Students ------------------------------------------------------------------
    def list_students(self, session_token: str, *, active_only: bool = False) -> list[Student]:
        self._authorize(session_token, "list_students")
        return self._students.all(active_only=active_only)

    def students_dataframe(self, session_token: str, *, active_only: bool = False) -> pl.DataFrame:
        self._authorize(session_token, "students_dataframe")
        return self._students.to_dataframe(active_only=active_only)

    def add_student(
        self,
        session_token: str,
        student_id: str,
        first_name: str,
        last_name: str,
        dob: str,
        sex: str,
    ) -> Student:
        actor = self._authorize(session_token, "add_student")
        self._daily_backup(actor)
        student = Student.new(student_id, first_name, last_name, dob, sex)
        with self._db.transaction() as conn:
            self._students.add(student)
            self._audit.record(
                "STUDENT_ADDED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="students",
                target_id=student.student_id,
                details={"after": _student_dict(student)},
                conn=conn,
            )
        return student

    def update_student(
        self,
        session_token: str,
        student_id: str,
        *,
        first_name: str | None = None,
        last_name: str | None = None,
        sex: str | Sex | None = None,
        dob: str | None = None,
    ) -> Student:
        actor = self._authorize(session_token, "update_student")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            before = self._students.get(student_id)
            updated = self._students.update_details(
                student_id, first_name=first_name, last_name=last_name, sex=sex, dob=dob
            )
            self._audit.record(
                "STUDENT_UPDATED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="students",
                target_id=updated.student_id,
                details={"before": _student_dict(before), "after": _student_dict(updated)},
                conn=conn,
            )
        return updated

    def set_student_active(self, session_token: str, student_id: str, active: bool) -> None:
        actor = self._authorize(session_token, "set_student_active")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            before = self._students.get(student_id)
            self._students.set_active(student_id, active)
            self._audit.record(
                "STUDENT_ACTIVE_CHANGED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="students",
                target_id=before.student_id,
                details={"before": before.is_active, "after": active},
                conn=conn,
            )

    # ---- Enrolments & Sports -------------------------------------------------------
    def list_enrolments(
        self,
        session_token: str,
        *,
        student_id: str | None = None,
        season_year: int | None = None,
    ) -> list[Enrolment]:
        self._authorize(session_token, "list_enrolments")
        if student_id is not None:
            return self._enrolments.for_student(student_id, season_year=season_year)
        return self._enrolments.all(season_year=season_year)

    def enrolments_dataframe(
        self, session_token: str, *, season_year: int | None = None
    ) -> pl.DataFrame:
        self._authorize(session_token, "enrolments_dataframe")
        return self._enrolments.to_dataframe(season_year=season_year)

    def list_sports(self, session_token: str) -> list[str]:
        self._authorize(session_token, "list_sports")
        return self._enrolments.sports()

    def add_enrolment(
        self,
        session_token: str,
        student_id: str,
        season_year: int,
        sport: str,
        grade: str,
    ) -> Enrolment:
        actor = self._authorize(session_token, "add_enrolment")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            saved = self._enrolments.add(Enrolment.new(student_id, season_year, sport, grade))
            self._audit.record(
                "ENROLMENT_ADDED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="enrolments",
                target_id=saved.enrolment_id,
                details={"after": _enrolment_dict(saved)},
                conn=conn,
            )
        return saved

    def update_enrolment_grade(
        self, session_token: str, enrolment_id: int, grade: str
    ) -> None:
        actor = self._authorize(session_token, "update_enrolment_grade")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            before = self._enrolments.get(enrolment_id)
            self._enrolments.set_grade(enrolment_id, grade)
            after = self._enrolments.get(enrolment_id)
            self._audit.record(
                "ENROLMENT_GRADE_UPDATED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="enrolments",
                target_id=enrolment_id,
                details={"before": _enrolment_dict(before), "after": _enrolment_dict(after)},
                conn=conn,
            )

    def delete_enrolment(self, session_token: str, enrolment_id: int) -> None:
        actor = self._authorize(session_token, "delete_enrolment")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            before = self._enrolments.get(enrolment_id)
            self._enrolments.delete(enrolment_id)
            self._audit.record(
                "ENROLMENT_DELETED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="enrolments",
                target_id=enrolment_id,
                details={"deleted": _enrolment_dict(before)},
                conn=conn,
            )

    def add_sport(self, session_token: str, name: str) -> str:
        actor = self._authorize(session_token, "add_sport")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            added = self._enrolments.add_sport(name)
            self._audit.record(
                "SPORT_ADDED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="sports",
                target_id=added,
                details={"name": added},
                conn=conn,
            )
        return added

    # ---- Fitness Test Catalogue ----------------------------------------------------
    def list_fitness_tests(
        self, session_token: str, *, category: str | None = None
    ) -> list[FitnessTest]:
        self._authorize(session_token, "list_fitness_tests")
        return self._tests.all(category=category)

    def fitness_tests_dataframe(
        self, session_token: str, *, category: str | None = None
    ) -> pl.DataFrame:
        self._authorize(session_token, "fitness_tests_dataframe")
        return self._tests.to_dataframe(category=category)

    def list_categories(self, session_token: str) -> dict[str, int]:
        self._authorize(session_token, "list_categories")
        return self._tests.categories()

    def list_units(self, session_token: str) -> list[str]:
        self._authorize(session_token, "list_units")
        return self._tests.units()

    def add_category(
        self, session_token: str, name: str, id_start: int | None = None
    ) -> tuple[str, int]:
        actor = self._authorize(session_token, "add_category")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            cat_name, start = self._tests.add_category(name, id_start=id_start)
            self._audit.record(
                "CATEGORY_ADDED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="test_categories",
                target_id=cat_name,
                details={"name": cat_name, "id_start": start},
                conn=conn,
            )
        return cat_name, start

    def add_unit(self, session_token: str, name: str) -> str:
        actor = self._authorize(session_token, "add_unit")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            unit = self._tests.add_unit(name)
            self._audit.record(
                "UNIT_ADDED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="test_units",
                target_id=unit,
                details={"name": unit},
                conn=conn,
            )
        return unit

    def add_fitness_test(
        self,
        session_token: str,
        test_name: str,
        category: str,
        unit: str,
        min_value: float | None = None,
        max_value: float | None = None,
        *,
        score_direction: str | ScoreDirection = ScoreDirection.HIGHER,
        test_id: int | None = None,
    ) -> FitnessTest:
        actor = self._authorize(session_token, "add_fitness_test")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            tid = self._tests.next_id(category) if test_id is None else test_id
            test = FitnessTest.new(
                tid, test_name, category, unit, min_value, max_value, score_direction=score_direction
            )
            self._tests.add(test)
            saved = self._tests.get(tid)
            self._audit.record(
                "FITNESS_TEST_ADDED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="fitness_tests",
                target_id=saved.test_id,
                details={
                    "test_name": saved.test_name,
                    "category": saved.category,
                    "unit": saved.unit,
                    "min_value": saved.min_value,
                    "max_value": saved.max_value,
                    "score_direction": saved.score_direction.value,
                },
                conn=conn,
            )
        return saved

    def rename_fitness_test(
        self, session_token: str, test_id: int, new_name: str
    ) -> FitnessTest:
        actor = self._authorize(session_token, "rename_fitness_test")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            before = self._tests.get(test_id)
            self._tests.rename(test_id, new_name)
            after = self._tests.get(test_id)
            self._audit.record(
                "FITNESS_TEST_RENAMED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="fitness_tests",
                target_id=test_id,
                details={"before": before.test_name, "after": after.test_name},
                conn=conn,
            )
        return after

    def set_fitness_test_range(
        self, session_token: str, test_id: int, min_value: float, max_value: float
    ) -> FitnessTest:
        actor = self._authorize(session_token, "set_fitness_test_range")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            before = self._tests.get(test_id)
            self._tests.set_range(test_id, min_value, max_value)
            after = self._tests.get(test_id)
            self._audit.record(
                "FITNESS_TEST_RANGE_UPDATED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="fitness_tests",
                target_id=test_id,
                details={
                    "before": [before.min_value, before.max_value],
                    "after": [after.min_value, after.max_value],
                },
                conn=conn,
            )
        return after

    def set_level_shuttle_rules(
        self, session_token: str, test_id: int, rules: dict[int, int]
    ) -> None:
        actor = self._authorize(session_token, "set_level_shuttle_rules")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            self._tests.set_level_shuttle_rules(test_id, rules)
            self._audit.record(
                "LEVEL_SHUTTLE_RULES_UPDATED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="level_shuttle_rules",
                target_id=test_id,
                details={"level_count": len(rules)},
                conn=conn,
            )

    # ---- Test Results --------------------------------------------------------------
    def list_results(
        self,
        session_token: str,
        *,
        student_id: str | None = None,
        season_year: int | None = None,
        test_id: int | None = None,
    ) -> list[TestResult]:
        self._authorize(session_token, "list_results")
        return self._results.all(
            student_id=student_id, season_year=season_year, test_id=test_id
        )

    def results_dataframe(
        self,
        session_token: str,
        *,
        student_id: str | None = None,
        season_year: int | None = None,
        test_id: int | None = None,
    ) -> pl.DataFrame:
        self._authorize(session_token, "results_dataframe")
        return self._results.to_dataframe(
            student_id=student_id, season_year=season_year, test_id=test_id
        )

    def check_imtp_warning(
        self,
        session_token: str,
        student_id: str,
        test_date: str | date,
        *,
        peak_newtons: float | None = None,
        relative_score: float | None = None,
        weight_kg: float | None = None,
    ) -> str | None:
        self._authorize(session_token, "check_imtp_warning")
        return self._results.check_imtp_session_warning(
            student_id,
            test_date,
            peak_newtons=peak_newtons,
            relative_score=relative_score,
            weight_kg=weight_kg,
        )

    def add_result(
        self,
        session_token: str,
        student_id: str,
        test_id: int,
        test_date: str,
        value: float | None = None,
        *,
        level: int | None = None,
        shuttle: int | None = None,
    ) -> TestResult:
        actor = self._authorize(session_token, "add_result")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            saved = self._results.add(
                TestResult.new(student_id, test_id, test_date, value, level=level, shuttle=shuttle)
            )
            self._audit.record(
                "RESULT_ADDED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="test_results",
                target_id=saved.result_id,
                details={"after": _result_dict(saved)},
                conn=conn,
            )
        return saved

    def add_results_batch(
        self, session_token: str, results: list[TestResult]
    ) -> list[TestResult]:
        """Save a squad batch of test results with an automatic pre-bulk safety backup."""
        actor = self._authorize(session_token, "add_results_batch")
        self._bump_cache()
        if self._enable_auto_backups and results:
            self._backups.create_backup(
                "pre_bulk",
                audit=self._audit,
                actor_user_id=actor.user_id,
                actor_username=actor.username,
            )
        with self._db.transaction() as conn:
            saved = self._results.add_many(results)
            self._audit.record(
                "RESULTS_BATCH_ADDED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="test_results",
                target_id=f"batch:{len(saved)}",
                details={
                    "count": len(saved),
                    "result_ids": [r.result_id for r in saved],
                },
                conn=conn,
            )
        return saved

    def update_result_score(
        self,
        session_token: str,
        result_id: int,
        value: float | None = None,
        *,
        level: int | None = None,
        shuttle: int | None = None,
    ) -> TestResult:
        actor = self._authorize(session_token, "update_result_score")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            before = self._results.get(result_id)
            updated = self._results.update_score(result_id, value, level=level, shuttle=shuttle)
            self._audit.record(
                "RESULT_UPDATED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="test_results",
                target_id=result_id,
                details={"before": _result_dict(before), "after": _result_dict(updated)},
                conn=conn,
            )
        return updated

    def delete_result(self, session_token: str, result_id: int) -> None:
        """Delete a mistaken test result, recording the FULL deleted row in audit_log."""
        actor = self._authorize(session_token, "delete_result")
        self._daily_backup(actor)
        with self._db.transaction() as conn:
            before = self._results.get(result_id)
            self._results.delete(result_id)
            self._audit.record(
                "RESULT_DELETED",
                actor_user_id=actor.user_id,
                actor_username=actor.username,
                target_table="test_results",
                target_id=result_id,
                details={"deleted": _result_dict(before)},
                conn=conn,
            )

    # ---- Headless Analytics (Phase 3) ----------------------------------------------
    def enriched_results_dataframe(
        self,
        session_token: str,
        *,
        season_year: int | None = None,
        sport: str | None = None,
        active_only: bool = False,
    ) -> pl.DataFrame:
        self._authorize(session_token, "enriched_results_dataframe")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        df = _build_enriched_results(
            results_df,
            students_df,
            enrolments_df,
            tests_df,
            shuttle_rules,
            sport=sport,
            active_only=active_only,
        )
        if season_year is not None:
            df = df.filter(pl.col("season_year") == season_year)
        return df

    def athlete_trends(
        self,
        session_token: str,
        student_id: str,
        *,
        test_id: int | None = None,
        category: str | None = None,
    ) -> pl.DataFrame:
        self._authorize(session_token, "athlete_trends")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df, students_df, enrolments_df, tests_df, shuttle_rules
        )
        return _athlete_trends(enriched, student_id, test_id=test_id, category=category)

    def athlete_personal_bests(
        self,
        session_token: str,
        student_id: str,
        *,
        season_year: int | None = None,
    ) -> pl.DataFrame:
        self._authorize(session_token, "athlete_personal_bests")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df, students_df, enrolments_df, tests_df, shuttle_rules
        )
        return _athlete_personal_bests(enriched, student_id, season_year=season_year)

    def cohort_benchmarks(
        self,
        session_token: str,
        *,
        season_year: SeasonFilter = None,
        semester: int | None = None,
        grade: str | None = None,
        sex: str | None = None,
        sport: str | None = None,
        test_id: int | None = None,
        category: str | None = None,
        basis: Basis = "latest",
        active_only: bool = False,
        min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
    ) -> pl.DataFrame:
        self._authorize(session_token, "cohort_benchmarks")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df,
            students_df,
            enrolments_df,
            tests_df,
            shuttle_rules,
            sport=sport,
            active_only=active_only,
        )
        return _cohort_benchmarks(
            enriched,
            enrolments_df,
            students_df,
            shuttle_rules,
            season_year=season_year,
            semester=semester,
            grade=grade,
            sex=sex,
            sport=sport,
            test_id=test_id,
            category=category,
            basis=basis,
            active_only=active_only,
            min_cohort_size=min_cohort_size,
        )

    def athlete_percentiles(
        self,
        session_token: str,
        student_id: str,
        *,
        season_year: SeasonFilter = None,
        semester: int | None = None,
        sport: str | None = None,
        match_grade: bool = True,
        match_sex: bool = True,
        basis: Basis = "latest",
        active_only: bool = False,
        min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
    ) -> pl.DataFrame:
        self._authorize(session_token, "athlete_percentiles")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df, students_df, enrolments_df, tests_df, shuttle_rules, sport=sport
        )
        return _athlete_percentiles(
            enriched,
            enrolments_df,
            students_df,
            student_id,
            season_year=season_year,
            semester=semester,
            sport=sport,
            match_grade=match_grade,
            match_sex=match_sex,
            basis=basis,
            active_only=active_only,
            min_cohort_size=min_cohort_size,
        )

    def athlete_category_profile(
        self,
        session_token: str,
        student_id: str,
        *,
        season_year: SeasonFilter = None,
        semester: int | None = None,
        sport: str | None = None,
        match_grade: bool = True,
        match_sex: bool = True,
        basis: Basis = "latest",
        active_only: bool = False,
        min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
        primary_tests: dict[str, int] | None = None,
    ) -> pl.DataFrame:
        self._authorize(session_token, "athlete_category_profile")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df, students_df, enrolments_df, tests_df, shuttle_rules, sport=sport
        )
        pcts = _athlete_percentiles(
            enriched,
            enrolments_df,
            students_df,
            student_id,
            season_year=season_year,
            semester=semester,
            sport=sport,
            match_grade=match_grade,
            match_sex=match_sex,
            basis=basis,
            active_only=active_only,
            min_cohort_size=min_cohort_size,
        )
        return _athlete_category_profile(pcts, tests_df, primary_tests=primary_tests)

    def asymmetry_report(
        self,
        session_token: str,
        *,
        student_id: str | None = None,
        season_year: SeasonFilter = "all",
        semester: int | None = None,
        sport: str | None = None,
        grade: str | None = None,
        active_only: bool = False,
        thresholds_by_test: dict[str, tuple[float, float]] | None = None,
        thresholds_by_unit: dict[str, tuple[float, float]] | None = None,
    ) -> pl.DataFrame:
        self._authorize(session_token, "asymmetry_report")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df,
            students_df,
            enrolments_df,
            tests_df,
            shuttle_rules,
            sport=sport,
            active_only=active_only,
        )
        return _asymmetry_report(
            enriched,
            tests_df,
            student_id=student_id,
            season_year=season_year,
            semester=semester,
            grade=grade,
            thresholds_by_test=thresholds_by_test,
            thresholds_by_unit=thresholds_by_unit,
        )

    def relative_strength_metrics(
        self,
        session_token: str,
        *,
        student_id: str | None = None,
        season_year: SeasonFilter = "all",
        sport: str | None = None,
        active_only: bool = False,
    ) -> pl.DataFrame:
        self._authorize(session_token, "relative_strength_metrics")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df,
            students_df,
            enrolments_df,
            tests_df,
            shuttle_rules,
            sport=sport,
            active_only=active_only,
        )
        return _relative_strength_metrics(
            enriched, student_id=student_id, season_year=season_year
        )

    def squad_leaderboard(
        self,
        session_token: str,
        test_id: int,
        *,
        season_year: SeasonFilter = None,
        semester: int | None = None,
        grade: str | None = None,
        sex: str | None = None,
        sport: str | None = None,
        basis: Basis = "latest",
        active_only: bool = False,
        min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
        limit: int | None = None,
    ) -> pl.DataFrame:
        self._authorize(session_token, "squad_leaderboard")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df,
            students_df,
            enrolments_df,
            tests_df,
            shuttle_rules,
            sport=sport,
            active_only=active_only,
        )
        return _squad_leaderboard(
            enriched,
            enrolments_df,
            students_df,
            test_id,
            tests_df=tests_df,
            season_year=season_year,
            semester=semester,
            grade=grade,
            sex=sex,
            sport=sport,
            basis=basis,
            active_only=active_only,
            min_cohort_size=min_cohort_size,
            limit=limit,
        )

    def compare_athletes(
        self,
        session_token: str,
        student_ids: list[str],
        *,
        season_year: SeasonFilter = None,
        semester: int | None = None,
        sport: str | None = None,
        test_ids: list[int] | None = None,
        category: str | None = None,
        basis: Basis = "latest",
        match_grade: bool = False,
        match_sex: bool = False,
        active_only: bool = False,
        min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
    ) -> pl.DataFrame:
        self._authorize(session_token, "compare_athletes")
        students_df, enrolments_df, tests_df, results_df, shuttle_rules = self._analytics_snapshot()
        enriched = _build_enriched_results(
            results_df, students_df, enrolments_df, tests_df, shuttle_rules, sport=sport
        )
        return _compare_athletes(
            enriched,
            enrolments_df,
            students_df,
            student_ids,
            season_year=season_year,
            semester=semester,
            sport=sport,
            test_ids=test_ids,
            category=category,
            basis=basis,
            match_grade=match_grade,
            match_sex=match_sex,
            active_only=active_only,
            min_cohort_size=min_cohort_size,
        )

    def dashboard_overview(
        self,
        session_token: str,
        *,
        season_year: int | None = None,
        sport: str | None = None,
        grade: str | None = None,
        active_only: bool = False,
    ) -> dict[str, Any]:
        self._authorize(session_token, "dashboard_overview")
        students_df, enrolments_df, _tests_df, results_df, _shuttle_rules = self._analytics_snapshot()
        return _dashboard_overview(
            students_df,
            enrolments_df,
            results_df,
            season_year=season_year,
            sport=sport,
            grade=grade,
            active_only=active_only,
        )

    # ---- Lead Coach Administration -------------------------------------------------
    def list_users(self, session_token: str, *, active_only: bool = False) -> list[User]:
        self._authorize(session_token, "list_users")
        return self._auth.all(active_only=active_only)

    def create_user(
        self,
        session_token: str,
        username: str,
        full_name: str,
        role: str | Role,
        temporary_password: str | None = None,
    ) -> tuple[User, str]:
        actor = self._authorize(session_token, "create_user")
        self._daily_backup(actor)
        return self._auth.create_user(
            username, full_name, role, temporary_password, actor=actor
        )

    def reset_user_password(
        self,
        session_token: str,
        target_user_id: int,
        temporary_password: str | None = None,
    ) -> str:
        actor = self._authorize(session_token, "reset_user_password")
        return self._auth.reset_password(target_user_id, temporary_password, actor=actor)

    def unlock_user(self, session_token: str, target_user_id: int) -> User:
        actor = self._authorize(session_token, "unlock_user")
        return self._auth.unlock_user(target_user_id, actor=actor)

    def set_user_active(
        self, session_token: str, target_user_id: int, active: bool
    ) -> User:
        actor = self._authorize(session_token, "set_user_active")
        return self._auth.set_active(target_user_id, active, actor=actor)

    def set_user_role(
        self, session_token: str, target_user_id: int, role: str | Role
    ) -> User:
        actor = self._authorize(session_token, "set_user_role")
        return self._auth.set_role(target_user_id, role, actor=actor)

    def recent_audit_entries(
        self,
        session_token: str,
        *,
        limit: int = 100,
        actor_username: str | None = None,
        action: str | None = None,
    ) -> list[AuditEntry]:
        self._authorize(session_token, "recent_audit_entries")
        return self._audit.recent(limit=limit, actor_username=actor_username, action=action)

    def audit_dataframe(
        self,
        session_token: str,
        *,
        limit: int = 500,
        actor_username: str | None = None,
        action: str | None = None,
    ) -> pl.DataFrame:
        self._authorize(session_token, "audit_dataframe")
        return self._audit.to_dataframe(
            limit=limit, actor_username=actor_username, action=action
        )

    def verify_audit_chain(self, session_token: str) -> tuple[bool, str]:
        self._authorize(session_token, "verify_audit_chain")
        return self._audit.verify_chain()

    def create_manual_backup(self, session_token: str) -> Path:
        actor = self._authorize(session_token, "create_manual_backup")
        return self._backups.create_backup(
            "manual",
            audit=self._audit,
            actor_user_id=actor.user_id,
            actor_username=actor.username,
        )

    def list_backups(self, session_token: str) -> list[BackupInfo]:
        self._authorize(session_token, "list_backups")
        return self._backups.list_backups()
