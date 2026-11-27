"""importer.py - Phase 1c: Transactional, idempotent importer for cleaned Phase 1b CSVs.

Reads the spotless CSV outputs in `data/cleaned/`:
    1. students_clean.csv      -> Student.legacy(...) into `students`
    2. enrolments_clean.csv    -> Enrolment.new(...) into `enrolments`
    3. test_results_clean.csv  -> TestResult.legacy(...) into `test_results`

Key Guarantees:
    * Opens `db.legacy_import_mode()` ONLY for the duration of the import, and ALWAYS
      re-locks `allow_legacy_import = '0'` on exit (even if an error occurs).
    * Validates every single row through the domain constructors AND SQLite triggers
      (never bypasses repository/database constraints).
    * Idempotent: safe to re-run without duplicating rows or failing on existing records.
    * Runs `PRAGMA foreign_key_check` and `PRAGMA integrity_check` after import.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from whc.database import DEFAULT_DB_PATH, Database
from whc.enrolments import Enrolment, EnrolmentRepository
from whc.exceptions import DuplicateError
from whc.fitness_tests import FitnessTestRepository
from whc.students import Student, StudentRepository
from whc.test_results import TestResult, TestResultRepository

DEFAULT_CLEANED_DIR = Path(__file__).resolve().parent.parent / "data" / "cleaned"


@dataclass(frozen=True, slots=True)
class ImportSummary:
    students_added: int
    students_skipped: int
    enrolments_added: int
    enrolments_skipped: int
    results_added: int
    results_skipped: int
    total_students: int
    total_enrolments: int
    total_fitness_tests: int
    total_results: int
    legacy_import_locked: bool
    foreign_key_errors: int
    integrity_ok: bool


def import_cleaned_data(
    db: Database,
    cleaned_dir: Path | str = DEFAULT_CLEANED_DIR,
) -> ImportSummary:
    """Import spotless Phase 1b CSVs into `db` cleanly, idempotently, and with post-import locking."""
    base = Path(cleaned_dir)
    p_students = base / "students_clean.csv"
    p_enrolments = base / "enrolments_clean.csv"
    p_results = base / "test_results_clean.csv"

    for p in (p_students, p_enrolments, p_results):
        if not p.exists():
            raise FileNotFoundError(
                f"Required cleaned CSV not found: {p}. Run `uv run python -m whc.cleaner` first."
            )

    students_repo = StudentRepository(db)
    enrolments_repo = EnrolmentRepository(db)
    tests_repo = FitnessTestRepository(db)
    tests_repo.load_catalogue()
    results_repo = TestResultRepository(db)

    s_added = s_skipped = 0
    e_added = e_skipped = 0
    r_added = r_skipped = 0

    with db.legacy_import_mode():
        # 1. Import students
        with p_students.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                dob_raw = row["dob"].strip()
                is_active = row["is_active"].strip().lower() in ("true", "1", "yes")
                if dob_raw:
                    student = Student.new(
                        student_id=row["student_id"],
                        first_name=row["first_name"],
                        last_name=row["last_name"],
                        dob=dob_raw,
                        sex=row["sex"],
                        is_active=is_active,
                    )
                else:
                    student = Student.legacy(
                        student_id=row["student_id"],
                        first_name=row["first_name"],
                        last_name=row["last_name"],
                        sex=row["sex"],
                        is_active=is_active,
                    )
                try:
                    students_repo.add(student)
                    s_added += 1
                except DuplicateError:
                    s_skipped += 1

        # 2. Import enrolments
        with p_enrolments.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                enrolment = Enrolment.new(
                    student_id=row["student_id"],
                    season_year=int(row["season_year"]),
                    sport=row["sport"],
                    grade=row["grade"],
                )
                try:
                    enrolments_repo.add(enrolment)
                    e_added += 1
                except DuplicateError:
                    e_skipped += 1

        # 3. Import legacy test results (NO fabricated dates!)
        with p_results.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                val_raw = row["value"].strip()
                lvl_raw = row["level"].strip()
                sh_raw = row["shuttle"].strip()
                result = TestResult.legacy(
                    student_id=row["student_id"],
                    test_id=int(row["test_id"]),
                    season_year=int(row["season_year"]),
                    semester=int(row["semester"]),
                    value=float(val_raw) if val_raw != "" else None,
                    level=int(lvl_raw) if lvl_raw != "" else None,
                    shuttle=int(sh_raw) if sh_raw != "" else None,
                )
                try:
                    results_repo.add(result)
                    r_added += 1
                except DuplicateError:
                    r_skipped += 1

    # Verify DB integrity & post-import lock
    fk_errors = db.query("PRAGMA foreign_key_check")
    integrity = db.query("PRAGMA integrity_check")
    integrity_ok = integrity == [("ok",)]

    if fk_errors:
        raise RuntimeError(f"PRAGMA foreign_key_check failed with {len(fk_errors)} error(s)")
    if not integrity_ok:
        raise RuntimeError(f"PRAGMA integrity_check failed: {integrity}")

    return ImportSummary(
        students_added=s_added,
        students_skipped=s_skipped,
        enrolments_added=e_added,
        enrolments_skipped=e_skipped,
        results_added=r_added,
        results_skipped=r_skipped,
        total_students=len(students_repo.all()),
        total_enrolments=len(enrolments_repo.all()),
        total_fitness_tests=len(tests_repo.all()),
        total_results=len(results_repo.all()),
        legacy_import_locked=not db.is_legacy_import_open(),
        foreign_key_errors=len(fk_errors),
        integrity_ok=integrity_ok,
    )


def main() -> None:
    with Database(DEFAULT_DB_PATH) as db:
        summary = import_cleaned_data(db, DEFAULT_CLEANED_DIR)
    print("Phase 1c Transactional Database Import Complete")
    print(f"  Database:            {DEFAULT_DB_PATH}")
    print(f"  Students:            +{summary.students_added} added, {summary.students_skipped} skipped (total: {summary.total_students})")
    print(f"  Enrolments:          +{summary.enrolments_added} added, {summary.enrolments_skipped} skipped (total: {summary.total_enrolments})")
    print(f"  Fitness Tests:       {summary.total_fitness_tests} catalogue tests loaded")
    print(f"  Test Results:        +{summary.results_added} added, {summary.results_skipped} skipped (total: {summary.total_results})")
    print(f"  Legacy Import Lock:  {'LOCKED' if summary.legacy_import_locked else 'OPEN'}")
    print(f"  FK Check / Integrity: {summary.foreign_key_errors} FK errors | integrity_ok={summary.integrity_ok}")


if __name__ == "__main__":
    main()
