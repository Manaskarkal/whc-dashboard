# pyright: reportArgumentType=false
"""Tries to break the test_results table. Run with: uv run python -m pytest

Sections:
    1. TestResult: cleaning and validation (Python side)
    2. Repository: saving, batching, filtering, update_score, delete
    3. Cross-table rules: season_year match, unit match, plausibility range, enrolment delete guard
    4. The database itself refuses bad data, even if Python is bypassed
    5. Dataframes and table upkeep
"""

import sqlite3
from dataclasses import FrozenInstanceError
from datetime import date, timedelta

import polars as pl
import pytest

import whc.test_results as results_module
from whc.database import Database
from whc.enrolments import Enrolment, EnrolmentRepository
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.fitness_tests import FitnessTestRepository
from whc.students import Student, StudentRepository
from whc.test_results import TestResult, TestResultRepository

RAW_INSERT = (
    "INSERT INTO test_results (enrolment_id, test_id, test_date, value, level, shuttle) "
    "VALUES (?, ?, ?, ?, ?, ?)"
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
def repo(db):
    return TestResultRepository(db)


@pytest.fixture
def enrolment(students, enrolments):
    """Sam (AAA0001) enrolled in 2025 AFL Year 8."""
    students.add(Student.new("AAA0001", "Sam", "Lee", None, "Male"))
    return enrolments.add(Enrolment.new("AAA0001", 2025, "AFL", "Year 8"))


# ==== 1. TestResult: cleaning and validation ==============================================
def test_new_numeric_result_and_semester_calculation():
    s1 = TestResult.new(1, 101, "15/03/2025", 34.5)
    assert s1.semester == 1
    assert s1.is_level_shuttle is False
    assert s1.formatted_score == "34.5"

    s2 = TestResult.new(1, 101, "20/08/2025", 36)
    assert s2.semester == 2
    assert s2.value == 36.0


def test_from_level_shuttle_text_parses_colon_and_dot():
    a = TestResult.from_level_shuttle_text(1, 401, "01/03/2025", " 16 : 3 ")
    assert (a.level, a.shuttle, a.value) == (16, 3, None)
    assert a.is_level_shuttle is True
    assert a.formatted_score == "16:3"

    b = TestResult.from_level_shuttle_text(1, 401, "01/08/2025", "15.4")
    assert (b.level, b.shuttle) == (15, 4)


@pytest.mark.parametrize("kwargs, message", [
    (dict(enrolment_id=0, test_id=101, test_date="01/03/2025", value=30.0), "enrolment_id"),
    (dict(enrolment_id=True, test_id=101, test_date="01/03/2025", value=30.0), "enrolment_id"),
    (dict(enrolment_id=1, test_id=100, test_date="01/03/2025", value=30.0), "test_id must be between"),
    (dict(enrolment_id=1, test_id=101, test_date="2025-03-01", value=30.0), "real date"),
    (dict(enrolment_id=1, test_id=101, test_date="01/03/1999", value=30.0), "on or after"),
    (dict(enrolment_id=1, test_id=101, test_date="06/11/2026", value=30.0), "in the future"),
    (dict(enrolment_id=1, test_id=101, test_date="01/03/2025"), "supply either value"),
    (dict(enrolment_id=1, test_id=101, test_date="01/03/2025", value=30.0, level=10, shuttle=2), "supply either value"),
    (dict(enrolment_id=1, test_id=401, test_date="01/03/2025", level=10), "supply either value"),
    (dict(enrolment_id=1, test_id=401, test_date="01/03/2025", level=0, shuttle=2), "level must be between"),
    (dict(enrolment_id=1, test_id=401, test_date="01/03/2025", level=31, shuttle=2), "level must be between"),
    (dict(enrolment_id=1, test_id=401, test_date="01/03/2025", level=10, shuttle=0), "shuttle must be between"),
    (dict(enrolment_id=1, test_id=401, test_date="01/03/2025", level=10, shuttle=21), "shuttle must be between"),
    (dict(enrolment_id=1, test_id=101, test_date="01/03/2025", value=float("nan")), "must be a number"),
    (dict(enrolment_id=1, test_id=101, test_date="01/03/2025", value=True), "must be a number"),
])
def test_bad_input_refused(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        TestResult.new(**kwargs)


def test_result_is_frozen():
    r = TestResult.new(1, 101, "01/03/2025", 30.0)
    with pytest.raises(FrozenInstanceError):
        r.value = 35.0  # pyright: ignore[reportAttributeAccessIssue]


# ==== 2. Repository =======================================================================
def test_add_get_and_for_student(repo, students, enrolments, enrolment):
    r1 = repo.add(TestResult.new(enrolment.enrolment_id, 101, "01/03/2025", 33.7))
    r2 = repo.add(TestResult.new(enrolment.enrolment_id, 401, "01/08/2025", level=15, shuttle=4))
    assert r1.result_id is not None and r2.result_id is not None
    assert repo.get(r1.result_id) == r1
    assert repo.for_enrolment(enrolment.enrolment_id) == [r1, r2]
    assert repo.for_student("aaa0001", season_year=2025) == [r1, r2]
    assert repo.for_student("AAA0001", test_id=401) == [r2]


def test_duplicate_test_on_same_date_refused(repo, enrolment):
    repo.add(TestResult.new(enrolment.enrolment_id, 101, "01/03/2025", 33.7))
    with pytest.raises(DuplicateError, match="already has a result"):
        repo.add(TestResult.new(enrolment.enrolment_id, 101, "01/03/2025", 34.0))
    # Different date in the same semester is allowed.
    repo.add(TestResult.new(enrolment.enrolment_id, 101, "15/05/2025", 34.5))
    assert len(repo.for_enrolment(enrolment.enrolment_id, test_id=101)) == 2


def test_add_many_is_all_or_nothing(repo, enrolment):
    good = TestResult.new(enrolment.enrolment_id, 901, "01/03/2025", 175.0)
    bad = TestResult.new(enrolment.enrolment_id, 902, "01/03/2025", 999.0)  # outside Weight 20..200
    with pytest.raises(ValidationError, match="outside the allowed range"):
        repo.add_many([good, bad])
    assert repo.all() == []


def test_update_score_and_delete(repo, enrolment):
    saved = repo.add(TestResult.new(enrolment.enrolment_id, 101, "01/03/2025", 33.7))
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


# ==== 3. Cross-table rules ================================================================
def test_date_must_match_enrolment_season_year(repo, enrolment):
    with pytest.raises(ValidationError, match="enrolment season_year"):
        repo.add(TestResult.new(enrolment.enrolment_id, 101, "01/03/2026", 30.0))


def test_unit_shape_must_match_fitness_test(repo, enrolment):
    # 101 (CMJ Height) is Centimetres: giving level/shuttle is refused.
    with pytest.raises(ValidationError, match="does not match the test unit"):
        repo.add(TestResult.new(enrolment.enrolment_id, 101, "01/03/2025", level=10, shuttle=2))
    # 401 (YOYO IRT1) is Level:Shuttle: giving a single value is refused.
    with pytest.raises(ValidationError, match="does not match the test unit"):
        repo.add(TestResult.new(enrolment.enrolment_id, 401, "01/03/2025", 15.4))


def test_enrolment_cannot_be_deleted_while_results_exist(repo, enrolments, enrolment):
    saved = repo.add(TestResult.new(enrolment.enrolment_id, 101, "01/03/2025", 30.0))
    assert saved.result_id is not None
    with pytest.raises(ValidationError, match="test results are recorded"):
        enrolments.delete(enrolment.enrolment_id)
    repo.delete(saved.result_id)
    enrolments.delete(enrolment.enrolment_id)


# ==== 4. The database itself refuses bad data, even if Python is bypassed =================
@pytest.mark.parametrize("values, message", [
    ((999, 101, "2025-03-01", 30.0, None, None), "FOREIGN KEY"),
    ((1, 999, "2025-03-01", 30.0, None, None), "FOREIGN KEY"),
    ((1, 101, "01/03/2025", 30.0, None, None), "test_date"),
    ((1, 101, "2025-03-01", None, None, None), "does not match the test unit"),
    ((1, 101, "2025-03-01", 30.0, 10, 2), "score_shape"),
    ((1, 101, "2025-03-01", "fast", None, None), "value"),
    ((1, 101, "2026-03-01", 30.0, None, None), "season_year"),
    ((1, 101, "2025-03-01", 500.0, None, None), "allowed range"),
    ((1, 401, "2025-03-01", 15.4, None, None), "does not match the test unit"),
])
def test_database_refuses_bad_rows(db, enrolment, values, message):
    eid = enrolment.enrolment_id if values[0] == 1 else values[0]
    row = (eid, *values[1:])
    with pytest.raises(sqlite3.IntegrityError, match=message):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, row)


def test_database_refuses_future_test_date(db, students, enrolments):
    tomorrow = date.today() + timedelta(days=1)
    students.add(Student.new("EEE0001", "Max", "Ray", None, "Male"))
    enr = enrolments.add(Enrolment.new("EEE0001", tomorrow.year, "AFL", "Year 8"))
    with pytest.raises(sqlite3.IntegrityError, match="in the future"):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, (enr.enrolment_id, 101, tomorrow.isoformat(), 30.0, None, None))


def test_locked_columns_cannot_be_updated(db, repo, enrolment):
    saved = repo.add(TestResult.new(enrolment.enrolment_id, 101, "01/03/2025", 30.0))
    for col, val in (("test_id", 102), ("test_date", "2025-04-01"), ("created_at", "2000-01-01T00:00:00Z")):
        with pytest.raises(sqlite3.DatabaseError, match="locked"):
            with db.transaction() as conn:
                conn.execute(f"UPDATE test_results SET {col} = ? WHERE result_id = ?", (val, saved.result_id))


# ==== 5. Dataframes and table upkeep ======================================================
def test_dataframe_types_even_when_empty(repo):
    schema = repo.to_dataframe().schema
    assert schema["result_id"] == pl.Int64
    assert schema["enrolment_id"] == pl.Int64
    assert schema["test_id"] == pl.Int64
    assert schema["test_date"] == pl.Date
    assert schema["value"] == pl.Float64
    assert schema["level"] == pl.Int64
    assert schema["shuttle"] == pl.Int64


def test_dataframe_has_real_values(repo, enrolment):
    repo.add(TestResult.new(enrolment.enrolment_id, 901, "01/03/2025", 178.5))
    repo.add(TestResult.new(enrolment.enrolment_id, 401, "01/08/2025", level=15, shuttle=4))
    df = repo.to_dataframe()
    assert df["test_id"].to_list() == [901, 401]
    assert df["value"].to_list() == [178.5, None]
    assert df["level"].to_list() == [None, 15]
    assert df["shuttle"].to_list() == [None, 4]
