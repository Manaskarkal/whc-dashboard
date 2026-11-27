# pyright: reportArgumentType=false
"""Tries to break the test_results table. Run with: uv run python -m pytest

Sections:
    1. TestResult: dated & legacy constructors, cleaning and validation (Python side)
    2. Repository: saving, batching, filtering, update_score, delete, multi-sport 3NF
    3. Legacy import lock & gradual DOB backfill enforcement
    4. Sports-science protocol & physics rules (Yo-Yo/Beep protocol, sprint splits, IMTP >20% warning)
    5. Cross-table protection (last enrolment delete guard, fitness_tests set_range guard)
    6. The database itself refuses bad data, even if Python is bypassed
    7. Dataframes and Database WAL/versioning upkeep
"""

import sqlite3
from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import polars as pl
import pytest

import whc.test_results as results_module
from whc.database import SCHEMA_VERSION, Database
from whc.enrolments import Enrolment, EnrolmentRepository
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.fitness_tests import FitnessTestRepository
from whc.students import Student, StudentRepository
from whc.test_results import TestResult, TestResultRepository, check_imtp_warning

RAW_INSERT = (
    "INSERT INTO test_results "
    "(student_id, test_id, test_date, legacy_year, legacy_semester, value, level, shuttle) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


@pytest.fixture(autouse=True)
def frozen_today(monkeypatch):
    monkeypatch.setattr(results_module, "_today", lambda: date(2026, 11, 5))


@pytest.fixture
def db():
    with Database(":memory:") as database:
        StudentRepository(database)
        EnrolmentRepository(database)
        FitnessTestRepository(database).load_catalogue()
        TestResultRepository(database)
        yield database


@pytest.fixture
def students(db):
    return StudentRepository(db)


@pytest.fixture
def enrolments(db):
    return EnrolmentRepository(db)


@pytest.fixture
def fitness_tests(db):
    return FitnessTestRepository(db)


@pytest.fixture
def repo(db):
    return TestResultRepository(db)


@pytest.fixture
def sam(students, enrolments):
    """Sam (AAA0001) with a real DOB, enrolled in 2025 AFL Year 8."""
    students.add(Student.new("AAA0001", "Sam", "Lee", "14/03/2012", "Male"))
    enrolments.add(Enrolment.new("AAA0001", 2025, "AFL", "Year 8"))
    return "AAA0001"


# ==== 1. TestResult: cleaning and validation ==============================================
def test_new_numeric_result_and_semester_calculation():
    s1 = TestResult.new("aaa0001", 101, "15/03/2025", 34.5)
    assert s1.student_id == "AAA0001"
    assert s1.season_year == 2025
    assert s1.semester == 1
    assert s1.is_legacy is False
    assert s1.period_label == "15/03/2025"
    assert s1.is_level_shuttle is False
    assert s1.formatted_score == "34.5"

    s2 = TestResult.new("AAA0001", 101, "20/08/2025", 36)
    assert s2.semester == 2
    assert s2.value == 36.0


def test_legacy_result_has_no_fabricated_date():
    leg = TestResult.legacy("AAA0001", 101, 2025, 1, 34.5)
    assert leg.test_date is None
    assert leg.is_legacy is True
    assert (leg.season_year, leg.semester) == (2025, 1)
    assert leg.period_label == "2025 Sem 1"


def test_from_level_shuttle_text_parses_colon_and_dot():
    a = TestResult.from_level_shuttle_text("AAA0001", 401, "01/03/2025", " 16 : 3 ")
    assert (a.level, a.shuttle, a.value) == (16, 3, None)
    assert a.is_level_shuttle is True
    assert a.formatted_score == "16:3"

    b = TestResult.from_level_shuttle_text("AAA0001", 401, "01/08/2025", "15.4")
    assert (b.level, b.shuttle) == (15, 4)


@pytest.mark.parametrize("kwargs, message", [
    (dict(student_id="BAD", test_id=101, test_date="01/03/2025", value=30.0), "student_id"),
    (dict(student_id="AAA0001", test_id=100, test_date="01/03/2025", value=30.0), "test_id must be between"),
    (dict(student_id="AAA0001", test_id=101, test_date="2025-03-01", value=30.0), "real date"),
    (dict(student_id="AAA0001", test_id=101, test_date=None, value=30.0), "test_date must be text"),
    (dict(student_id="AAA0001", test_id=101, test_date="01/03/1999", value=30.0), "on or after"),
    (dict(student_id="AAA0001", test_id=101, test_date="06/11/2026", value=30.0), "in the future"),
    (dict(student_id="AAA0001", test_id=101, test_date="01/03/2025"), "supply either value"),
    (dict(student_id="AAA0001", test_id=101, test_date="01/03/2025", value=30.0, level=10, shuttle=2), "supply either value"),
    (dict(student_id="AAA0001", test_id=401, test_date="01/03/2025", level=10), "supply either value"),
    (dict(student_id="AAA0001", test_id=401, test_date="01/03/2025", level=0, shuttle=2), "level must be between"),
    (dict(student_id="AAA0001", test_id=401, test_date="01/03/2025", level=31, shuttle=2), "level must be between"),
    (dict(student_id="AAA0001", test_id=401, test_date="01/03/2025", level=10, shuttle=0), "shuttle must be between"),
    (dict(student_id="AAA0001", test_id=401, test_date="01/03/2025", level=10, shuttle=21), "shuttle must be between"),
    (dict(student_id="AAA0001", test_id=101, test_date="01/03/2025", value=float("nan")), "must be a number"),
    (dict(student_id="AAA0001", test_id=101, test_date="01/03/2025", value=True), "must be a number"),
])
def test_bad_input_refused(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        TestResult.new(**kwargs)


def test_cannot_mix_dated_and_legacy_fields():
    with pytest.raises(ValidationError, match="never both"):
        TestResult(None, "AAA0001", 101, date(2025, 3, 1), legacy_year=2025, legacy_semester=1, value=30.0)


def test_result_is_frozen():
    r = TestResult.new("AAA0001", 101, "01/03/2025", 30.0)
    with pytest.raises(FrozenInstanceError):
        r.value = 35.0  # pyright: ignore[reportAttributeAccessIssue]


# ==== 2. Repository & Multi-Sport 3NF =====================================================
def test_add_get_and_for_student(repo, sam):
    r1 = repo.add(TestResult.new(sam, 101, "01/03/2025", 33.7))
    r2 = repo.add(TestResult.new(sam, 401, "01/08/2025", level=15, shuttle=4))
    assert r1.result_id is not None and r2.result_id is not None
    assert repo.get(r1.result_id) == r1
    assert repo.for_student("aaa0001", season_year=2025) == [r1, r2]
    assert repo.for_student("AAA0001", test_id=401) == [r2]


def test_multi_sport_athlete_has_one_score_per_test_per_day(repo, enrolments, sam):
    """Sam plays BOTH AFL and Basketball in 2025; still only ONE Height on 01/03/2025."""
    enrolments.add(Enrolment.new(sam, 2025, "Basketball", "Year 8"))
    repo.add(TestResult.new(sam, 901, "01/03/2025", 175.0))
    with pytest.raises(DuplicateError, match="already has a result"):
        repo.add(TestResult.new(sam, 901, "01/03/2025", 176.0))
    # Different date in the same semester is allowed
    repo.add(TestResult.new(sam, 901, "15/05/2025", 175.5))
    assert len(repo.for_student(sam, test_id=901)) == 2


def test_add_many_is_all_or_nothing(repo, sam):
    good = TestResult.new(sam, 901, "01/03/2025", 175.0)
    bad = TestResult.new(sam, 902, "01/03/2025", 999.0)  # outside Weight 20..200
    with pytest.raises(ValidationError, match="outside the allowed range"):
        repo.add_many([good, bad])
    assert repo.all() == []


def test_update_score_and_delete(repo, sam):
    saved = repo.add(TestResult.new(sam, 101, "01/03/2025", 33.7))
    assert saved.result_id is not None
    updated = repo.update_score(saved.result_id, 35.2)
    assert repo.get(saved.result_id).value == 35.2
    assert updated.value == 35.2

    with pytest.raises(ValidationError, match="outside the allowed range"):
        repo.update_score(saved.result_id, 500.0)
    assert repo.get(saved.result_id).value == 35.2

    repo.delete(saved.result_id)
    with pytest.raises(NotFoundError):
        repo.get(saved.result_id)


# ==== 3. Legacy import lock & gradual DOB backfill ========================================
def test_legacy_results_blocked_when_locked_and_allowed_during_import(db, students, enrolments, repo):
    with db.legacy_import_mode():
        students.add(Student.legacy("LEG0001", "Alex", "Old", "Male"))
        enrolments.add(Enrolment.new("LEG0001", 2025, "AFL", "Year 8"))
        saved = repo.add(TestResult.legacy("LEG0001", 101, 2025, 1, 32.0))
    assert saved.test_date is None
    assert (saved.season_year, saved.semester) == (2025, 1)

    # Once legacy_import_mode closes, SQLite permanently blocks inserting undated results
    with pytest.raises(ValidationError, match="test_date is required"):
        repo.add(TestResult.legacy("LEG0001", 101, 2025, 2, 33.0))

    # Trying to add a NEW dated result for LEG0001 while their DOB is still NULL is blocked!
    with pytest.raises(ValidationError, match="student dob must be recorded"):
        repo.add(TestResult.new("LEG0001", 101, "15/08/2025", 33.5))

    # Once DOB is backfilled, new dated results work immediately
    students.set_dob("LEG0001", "10/05/2012")
    dated = repo.add(TestResult.new("LEG0001", 101, "15/08/2025", 33.5))
    assert dated.test_date == date(2025, 8, 15)


def test_inactive_or_unenrolled_student_cannot_get_results(students, repo, sam):
    # Enrolment exists for 2025, NOT 2024
    with pytest.raises(ValidationError, match="enrolment in that season_year"):
        repo.add(TestResult.new(sam, 101, "01/03/2024", 30.0))

    # Deactivated student cannot get new results even in an enrolled season
    students.set_active(sam, False)
    with pytest.raises(ValidationError, match="not active"):
        repo.add(TestResult.new(sam, 101, "01/03/2025", 30.0))


# ==== 4. Sports-science protocol & physics rules ==========================================
def test_yoyo_and_beep_protocol_rules_enforced_by_database(repo, sam):
    # Official Yo-Yo IRT1: level 15 max is 8 shuttles; 15:9 and nonexistent level 7:4 are BLOCKED
    repo.add(TestResult.new(sam, 401, "01/03/2025", level=15, shuttle=8))
    with pytest.raises(ValidationError, match="not valid for this test protocol"):
        repo.add(TestResult.new(sam, 401, "02/03/2025", level=15, shuttle=9))
    with pytest.raises(ValidationError, match="not valid for this test protocol"):
        repo.add(TestResult.new(sam, 401, "03/03/2025", level=7, shuttle=4))

    # 20m Beep Test: level 1 max is 7 shuttles; 1:8 is BLOCKED
    repo.add(TestResult.new(sam, 402, "01/03/2025", level=1, shuttle=7))
    with pytest.raises(ValidationError, match="not valid for this test protocol"):
        repo.add(TestResult.new(sam, 402, "02/03/2025", level=1, shuttle=8))


def test_sprint_split_ordering_enforced_only_when_both_recorded(repo, sam):
    # Recording ONLY 5m Sprint (no 10m or 20m) succeeds — never compares against NULL/0!
    r5 = repo.add(TestResult.new(sam, 201, "10/03/2025", 1.25))
    assert r5.value == 1.25

    # Adding 10m <= 5m on the SAME date is BLOCKED
    with pytest.raises(ValidationError, match="5m < 10m < 20m"):
        repo.add(TestResult.new(sam, 202, "10/03/2025", 1.25))

    # Valid 10m and 20m succeed
    r10 = repo.add(TestResult.new(sam, 202, "10/03/2025", 2.05))
    repo.add(TestResult.new(sam, 203, "10/03/2025", 3.45))

    # Updating 10m to be >= 20m is also BLOCKED
    assert r10.result_id is not None
    with pytest.raises(ValidationError, match="5m < 10m < 20m"):
        repo.update_score(r10.result_id, 3.50)

    # A different date has independent sprint splits
    repo.add(TestResult.new(sam, 202, "20/03/2025", 1.95))


def test_imtp_warning_flags_over_20_percent_without_blocking(repo, sam):
    # Peak=2000 N, Weight=70 kg -> expected Rel = 2000 / (70 * 9.81) = 2.91
    assert check_imtp_warning(2000, 3.0, 70.0) is None  # ~3% off -> no warning
    warn = check_imtp_warning(2000, 2.0, 70.0)  # ~46% off -> warning!
    assert warn is not None and "Warning: IMTP Relative" in warn

    # Save Weight, Peak, and an >20% off Relative: warning is returned, but DB STILL allows saving!
    repo.add(TestResult.new(sam, 902, "01/03/2025", 70.0))
    repo.add(TestResult.new(sam, 1, "01/03/2025", 2000.0))
    session_warn = repo.check_imtp_session_warning(sam, "01/03/2025", relative_score=2.0)
    assert session_warn is not None and "confirm" in session_warn
    saved = repo.add(TestResult.new(sam, 2, "01/03/2025", 2.0))
    assert saved.value == 2.0


# ==== 5. Cross-table protection ===========================================================
def test_unit_shape_must_match_fitness_test(repo, sam):
    with pytest.raises(ValidationError, match="does not match the test unit"):
        repo.add(TestResult.new(sam, 101, "01/03/2025", level=10, shuttle=2))
    with pytest.raises(ValidationError, match="does not match the test unit"):
        repo.add(TestResult.new(sam, 401, "01/03/2025", 15.4))


def test_cannot_delete_last_enrolment_for_tested_season(repo, enrolments, sam):
    afl = enrolments.for_student(sam, season_year=2025)[0]
    bball = enrolments.add(Enrolment.new(sam, 2025, "Basketball", "Year 8"))
    saved = repo.add(TestResult.new(sam, 101, "01/03/2025", 30.0))
    assert afl.enrolment_id and bball.enrolment_id and saved.result_id

    # Deleting ONE of two sports in 2025 is allowed
    enrolments.delete(bball.enrolment_id)
    # Deleting the LAST remaining enrolment for 2025 while results exist is BLOCKED
    with pytest.raises(ValidationError, match="last enrolment for a season that has test results"):
        enrolments.delete(afl.enrolment_id)

    # Once the result is removed, deleting the enrolment succeeds
    repo.delete(saved.result_id)
    enrolments.delete(afl.enrolment_id)


def test_narrowing_fitness_test_range_blocked_if_existing_results_violate(repo, fitness_tests, sam):
    repo.add(TestResult.new(sam, 101, "01/03/2025", 30.0))
    with pytest.raises(ValidationError, match="invalidate existing test results"):
        fitness_tests.set_range(101, 35.0, 90.0)
    # Narrowing while still containing 30.0 is allowed
    fitness_tests.set_range(101, 10.0, 80.0)
    assert fitness_tests.get(101).max_value == 80.0


# ==== 6. The database itself refuses bad data, even if Python is bypassed =================
@pytest.mark.parametrize("values, message", [
    (("ZZZ9999", 101, "2025-03-01", None, None, 30.0, None, None), "FOREIGN KEY"),
    (("AAA0001", 999, "2025-03-01", None, None, 30.0, None, None), "FOREIGN KEY"),
    (("AAA0001", 101, "01/03/2025", None, None, 30.0, None, None), "test_date"),
    (("AAA0001", 101, "2025-03-01", 2025, 1, 30.0, None, None), "temporal_shape"),
    (("AAA0001", 101, None, 2025, 1, 30.0, None, None), "test_date is required"),
    (("AAA0001", 101, "2025-03-01", None, None, None, None, None), "does not match the test unit"),
    (("AAA0001", 101, "2025-03-01", None, None, 30.0, 10, 2), "score_shape"),
    (("AAA0001", 101, "2025-03-01", None, None, "fast", None, None), "value"),
    (("AAA0001", 101, "2026-03-01", None, None, 30.0, None, None), "enrolment in that season_year"),
    (("AAA0001", 101, "2025-03-01", None, None, 500.0, None, None), "allowed range"),
    (("AAA0001", 401, "2025-03-01", None, None, 15.4, None, None), "does not match the test unit"),
    (("AAA0001", 401, "2025-03-01", None, None, None, 15, 9), "not valid for this test protocol"),
])
def test_database_refuses_bad_rows(db, sam, values, message):
    with pytest.raises(sqlite3.IntegrityError, match=message):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, values)


def test_database_refuses_future_test_date(db, students, enrolments):
    tomorrow = date.today() + timedelta(days=1)
    students.add(Student.new("EEE0001", "Max", "Ray", "01/05/2013", "Male"))
    enrolments.add(Enrolment.new("EEE0001", tomorrow.year, "AFL", "Year 8"))
    with pytest.raises(sqlite3.IntegrityError, match="in the future"):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, ("EEE0001", 101, tomorrow.isoformat(), None, None, 30.0, None, None))


def test_locked_columns_cannot_be_updated(db, repo, sam):
    saved = repo.add(TestResult.new(sam, 101, "01/03/2025", 30.0))
    for col, val in (
        ("test_id", 102),
        ("test_date", "2025-04-01"),
        ("created_at", "2000-01-01T00:00:00Z"),
    ):
        with pytest.raises(sqlite3.DatabaseError, match="locked"):
            with db.transaction() as conn:
                conn.execute(f"UPDATE test_results SET {col} = ? WHERE result_id = ?", (val, saved.result_id))


# ==== 7. Dataframes and Database WAL/versioning upkeep =====================================
def test_dataframe_types_even_when_empty(repo):
    schema = repo.to_dataframe().schema
    assert schema["result_id"] == pl.Int64
    assert schema["student_id"] == pl.String
    assert schema["test_id"] == pl.Int64
    assert schema["test_date"] == pl.Date
    assert schema["season_year"] == pl.Int64
    assert schema["semester"] == pl.Int64
    assert schema["value"] == pl.Float64
    assert schema["level"] == pl.Int64
    assert schema["shuttle"] == pl.Int64


def test_dataframe_mixes_dated_and_legacy_rows_cleanly(db, repo, sam):
    with db.legacy_import_mode():
        repo.add(TestResult.legacy(sam, 901, 2025, 1, 178.5))
    repo.add(TestResult.new(sam, 401, "01/08/2025", level=15, shuttle=4))
    df = repo.to_dataframe()
    assert df["test_id"].to_list() == [901, 401]
    assert df["test_date"].to_list() == [None, date(2025, 8, 1)]
    assert df["season_year"].to_list() == [2025, 2025]
    assert df["semester"].to_list() == [1, 2]
    assert df["value"].to_list() == [178.5, None]
    assert df["level"].to_list() == [None, 15]
    assert df["shuttle"].to_list() == [None, 4]


def test_file_database_uses_wal_and_schema_version(tmp_path):
    db_path = tmp_path / "test_wal.db"
    with Database(db_path) as file_db:
        (mode,) = file_db.query("PRAGMA journal_mode")[0]
        assert mode.lower() == "wal"
        assert file_db.schema_version == SCHEMA_VERSION
        # Verify legacy_import_mode auto-relocks even on exception
        with pytest.raises(RuntimeError):
            with file_db.legacy_import_mode():
                assert file_db.is_legacy_import_open() is True
                raise RuntimeError("boom")
        assert file_db.is_legacy_import_open() is False
