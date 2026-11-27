"""tests/test_importer.py - unit & integration tests for Phase 1c database importer."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from whc.database import Database
from whc.exceptions import ValidationError
from whc.importer import DEFAULT_CLEANED_DIR, import_cleaned_data
from whc.students import Student, StudentRepository
from whc.test_results import TestResult, TestResultRepository


def _write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def test_import_idempotency_and_post_import_lock(tmp_path: Path) -> None:
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    _write_csv(
        cleaned / "students_clean.csv",
        ["student_id", "first_name", "last_name", "dob", "sex", "is_active"],
        [
            ["AAA0001", "Inès", "Smith", "", "Female", "True"],
            ["BBB0002", "Alex", "Jones", "", "Male", "True"],
        ],
    )
    _write_csv(
        cleaned / "enrolments_clean.csv",
        ["student_id", "season_year", "sport", "grade"],
        [
            ["AAA0001", 2025, "Basketball", "Year 9"],
            ["AAA0001", 2025, "Athletics", "Year 9"],
        ],
    )
    _write_csv(
        cleaned / "test_results_clean.csv",
        ["student_id", "season_year", "semester", "test_id", "test_name", "value", "level", "shuttle", "source_row"],
        [
            ["AAA0001", 2025, 1, 902, "Weight", 60.5, "", "", 2],
            ["AAA0001", 2025, 1, 401, "YOYO Intermittent Recovery Test 1 (IRT1)", "", 15, 1, 2],
            ["AAA0001", 2025, 1, 201, "5 Metre Sprint", 1.25, "", "", 2],
            ["AAA0001", 2025, 1, 202, "10 Metre Sprint", 2.05, "", "", 2],
        ],
    )

    with Database(tmp_path / "test.db") as db:
        first = import_cleaned_data(db, cleaned)
        assert (first.students_added, first.students_skipped) == (2, 0)
        assert (first.enrolments_added, first.enrolments_skipped) == (2, 0)
        assert (first.results_added, first.results_skipped) == (4, 0)
        assert first.total_fitness_tests == 37
        assert first.legacy_import_locked is True
        assert first.foreign_key_errors == 0
        assert first.integrity_ok is True

        # Re-running is 100% idempotent
        second = import_cleaned_data(db, cleaned)
        assert (second.students_added, second.students_skipped) == (0, 2)
        assert (second.enrolments_added, second.enrolments_skipped) == (0, 2)
        assert (second.results_added, second.results_skipped) == (0, 4)
        assert second.legacy_import_locked is True

        # Post-import DB lock prevents inserting new students without DOB or results without test_date
        students = StudentRepository(db)
        results = TestResultRepository(db)

        with pytest.raises(ValidationError, match="dob is required"):
            students.add(Student.legacy("CCC0003", "No", "Dob", "Male"))

        with pytest.raises(ValidationError, match="test_date is required"):
            results.add(TestResult.legacy("AAA0001", 901, 2025, 2, 170.0))

        # New dated result for legacy student is blocked until DOB is backfilled
        with pytest.raises(ValidationError, match="student dob must be recorded"):
            results.add(TestResult.new("AAA0001", 901, "15/08/2025", 170.0))

        students.set_dob("AAA0001", "10/05/2011")
        saved = results.add(TestResult.new("AAA0001", 901, "15/08/2025", 170.0))
        assert saved.test_date is not None and saved.value == 170.0


def test_real_cleaned_csvs_import_cleanly_into_fresh_db(tmp_path: Path) -> None:
    if not (DEFAULT_CLEANED_DIR / "test_results_clean.csv").exists():
        pytest.skip("data/cleaned CSVs not present")

    with Database(tmp_path / "real_import.db") as db:
        summary = import_cleaned_data(db, DEFAULT_CLEANED_DIR)
        assert summary.total_students > 0
        assert summary.total_enrolments > 0
        assert summary.total_results > 0
        assert summary.total_fitness_tests == 37
        assert summary.legacy_import_locked is True
        assert summary.foreign_key_errors == 0
        assert summary.integrity_ok is True

        # Verify Level:Shuttle rows round-trip with NULL test_date and valid level/shuttle
        results = TestResultRepository(db)
        yoyo = results.all(test_id=401)
        assert len(yoyo) > 0
        assert all(r.test_date is None and r.level is not None and r.shuttle is not None for r in yoyo)
