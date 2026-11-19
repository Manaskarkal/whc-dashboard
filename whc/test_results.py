"""test_results.py - one row per test measurement (who was tested, in which test, on what date,
and what they scored).

    TestResult:             one validated test result (frozen once created)
    TestResultRepository:   saves and loads test results (the only code that touches the table)

How scores are stored:
    * Standard numeric tests (Seconds, Centimetres, Kilograms, ...):
          `value` holds the number (e.g. 33.7); `level` and `shuttle` are None.
    * Level:Shuttle tests (Yo-Yo IRT1/IRT2, 20m Beep Test):
          `level` and `shuttle` hold the two whole numbers (e.g. level=16, shuttle=3);
          `value` is None.

How dates and semesters work:
    * Every result has a real `test_date` (typed dd/mm/yyyy, stored YYYY-MM-DD).
    * `result.semester` is derived directly from `test_date` (Jan-Jun -> 1, Jul-Dec -> 2).
    * When importing legacy rows that only recorded Year + Semester, a standard placeholder
      date (01/03/YYYY for Semester 1, 01/08/YYYY for Semester 2) is used.

Business rules enforced (each one has tests):
    1. enrolment_id and test_id must exist in their parent tables.
    2. Only one result per (enrolment_id, test_id, test_date).
    3. test_date cannot be in the future and its year must match the enrolment's season_year.
    4. Score shape must match the test's unit: `value` for normal units, `level` + `shuttle`
       for Level:Shuttle tests.
    5. For normal units, `value` must sit within the fitness test's [min_value, max_value].
       For Level:Shuttle tests, `level` must be 1-30 and `shuttle` 1-20.
    6. Identity details (result_id, enrolment_id, test_id, test_date, created_at) are locked;
       a score typo can be corrected with update_score() or removed with delete().
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date, datetime

import polars as pl

from whc.database import Database, Repository
from whc.enrolments import EnrolmentRepository
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.fitness_tests import (
    IDS_PER_CATEGORY,
    MAX_TEST_ID,
    VALUE_LIMIT,
    FitnessTestRepository,
    Unit,
    _is_plain_number,
)
from whc.students import DATE_FORMAT, clean_student_id, require_text

# ---- Settings (change rules here) --------------------------------------------------
MIN_TEST_DATE = date(2000, 1, 1)
LEVEL_MIN, LEVEL_MAX = 1, 30
SHUTTLE_MIN, SHUTTLE_MAX = 1, 20
_LEVEL_SHUTTLE_TEXT = re.compile(r"\s*([0-9]+)\s*[:.]\s*([0-9]+)\s*")


def _today() -> date:
    """The one place 'today' comes from, so tests can freeze it."""
    return date.today()


def _parse_test_date(text: object) -> date:
    try:
        return datetime.strptime(require_text(text, "test_date").strip(), DATE_FORMAT).date()
    except ValueError:
        raise ValidationError("test_date must be a real date written dd/mm/yyyy") from None


def _check_positive_int(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValidationError(f"{label} must be a positive whole number")


def _check_score_fields(value: float | None, level: int | None, shuttle: int | None) -> None:
    """Either `value` alone (normal test) or `level` + `shuttle` together (Level:Shuttle)."""
    has_value = value is not None
    has_level = level is not None
    has_shuttle = shuttle is not None

    if has_value and not has_level and not has_shuttle:
        if not _is_plain_number(value):
            raise ValidationError("value must be a number (not text, bool or NaN)")
        if abs(value) > VALUE_LIMIT:
            raise ValidationError(f"value must be between -{VALUE_LIMIT:,} and {VALUE_LIMIT:,}")
        return

    if not has_value and has_level and has_shuttle:
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValidationError("level must be a whole number")
        if isinstance(shuttle, bool) or not isinstance(shuttle, int):
            raise ValidationError("shuttle must be a whole number")
        if not LEVEL_MIN <= level <= LEVEL_MAX:
            raise ValidationError(f"level must be between {LEVEL_MIN} and {LEVEL_MAX}")
        if not SHUTTLE_MIN <= shuttle <= SHUTTLE_MAX:
            raise ValidationError(f"shuttle must be between {SHUTTLE_MIN} and {SHUTTLE_MAX}")
        return

    raise ValidationError(
        "supply either value (for standard tests) or both level and shuttle (for Level:Shuttle tests)"
    )


# ---- The TestResult (value object) -------------------------------------------------
@dataclass(frozen=True, slots=True)
class TestResult:
    """One validated test measurement. Frozen: once created it cannot be changed in memory."""

    __test__ = False  # class name starts with "Test"; prevent pytest from collecting it

    result_id: int | None
    enrolment_id: int
    test_id: int
    test_date: date
    value: float | None = None
    level: int | None = None
    shuttle: int | None = None

    def __post_init__(self) -> None:
        """Rules that must hold for EVERY TestResult, however it was created."""
        if self.result_id is not None:
            _check_positive_int(self.result_id, "result_id")
        _check_positive_int(self.enrolment_id, "enrolment_id")
        if isinstance(self.test_id, bool) or not isinstance(self.test_id, int):
            raise ValidationError("test_id must be a whole number")
        if not 1 <= self.test_id <= MAX_TEST_ID or self.test_id % IDS_PER_CATEGORY == 0:
            raise ValidationError(
                f"test_id must be between 1 and {MAX_TEST_ID} and cannot be a multiple of "
                f"{IDS_PER_CATEGORY}"
            )
        if not isinstance(self.test_date, date) or isinstance(self.test_date, datetime):
            raise ValidationError("test_date must be a date")
        if self.test_date < MIN_TEST_DATE:
            raise ValidationError(f"test_date must be on or after {MIN_TEST_DATE.isoformat()}")
        if self.test_date > _today():
            raise ValidationError("test_date cannot be in the future")
        _check_score_fields(self.value, self.level, self.shuttle)

    @classmethod
    def new(
        cls,
        enrolment_id: int,
        test_id: int,
        test_date: str,
        value: float | None = None,
        *,
        level: int | None = None,
        shuttle: int | None = None,
    ) -> TestResult:
        """Build a test result from raw typed input. The database assigns result_id."""
        return cls(
            result_id=None,
            enrolment_id=enrolment_id,
            test_id=test_id,
            test_date=_parse_test_date(test_date),
            value=float(value) if _is_plain_number(value) else value,
            level=level,
            shuttle=shuttle,
        )

    @classmethod
    def from_level_shuttle_text(
        cls, enrolment_id: int, test_id: int, test_date: str, text: str
    ) -> TestResult:
        """Build a Level:Shuttle result from '16:3' or '16.3' notation."""
        raw = require_text(text, "level_shuttle")
        match = _LEVEL_SHUTTLE_TEXT.fullmatch(raw)
        if not match:
            raise ValidationError(
                "level_shuttle must be written as 'level:shuttle' or 'level.shuttle', e.g. '16:3'"
            )
        return cls.new(
            enrolment_id=enrolment_id,
            test_id=test_id,
            test_date=test_date,
            level=int(match.group(1)),
            shuttle=int(match.group(2)),
        )

    @classmethod
    def from_row(cls, row: tuple) -> TestResult:
        """Build a TestResult from a database row (unpacks all 8 columns on purpose)."""
        result_id, enrolment_id, test_id, test_date, value, level, shuttle, _created_at = row
        return cls(
            result_id=result_id,
            enrolment_id=int(enrolment_id),
            test_id=int(test_id),
            test_date=date.fromisoformat(test_date),
            value=value,
            level=level,
            shuttle=shuttle,
        )

    @property
    def semester(self) -> int:
        """1 for Jan-Jun, 2 for Jul-Dec."""
        return 1 if self.test_date.month <= 6 else 2

    @property
    def is_level_shuttle(self) -> bool:
        return self.value is None

    @property
    def formatted_score(self) -> str:
        if self.is_level_shuttle:
            return f"{self.level}:{self.shuttle}"
        assert self.value is not None
        return f"{self.value:g}"


# ---- The table (SQL) ---------------------------------------------------------------
_SCORE_MISMATCHES_UNIT = f"""EXISTS (
    SELECT 1 FROM fitness_tests ft
    WHERE ft.test_id = NEW.test_id
      AND ((ft.unit = '{Unit.LEVEL_SHUTTLE.value}' AND NEW.value IS NOT NULL)
        OR (ft.unit <> '{Unit.LEVEL_SHUTTLE.value}' AND NEW.value IS NULL))
)"""

_VALUE_OUTSIDE_RANGE = """NEW.value IS NOT NULL AND EXISTS (
    SELECT 1 FROM fitness_tests ft
    WHERE ft.test_id = NEW.test_id
      AND ft.min_value IS NOT NULL
      AND NOT (NEW.value BETWEEN ft.min_value AND ft.max_value)
)"""

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS test_results (
    result_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    enrolment_id INTEGER NOT NULL REFERENCES enrolments(enrolment_id),
    test_id      INTEGER NOT NULL REFERENCES fitness_tests(test_id),
    test_date    TEXT NOT NULL
                 CHECK (date(test_date) = test_date AND test_date >= '{MIN_TEST_DATE.isoformat()}'),
    value        REAL,
    level        INTEGER,
    shuttle      INTEGER,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    CONSTRAINT score_shape CHECK (
        (value IS NOT NULL AND value BETWEEN -{VALUE_LIMIT} AND {VALUE_LIMIT}
         AND level IS NULL AND shuttle IS NULL)
        OR (value IS NULL
            AND level BETWEEN {LEVEL_MIN} AND {LEVEL_MAX}
            AND shuttle BETWEEN {SHUTTLE_MIN} AND {SHUTTLE_MAX})
    )
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_test_results_enrolment_test_date
    ON test_results (enrolment_id, test_id, test_date);
CREATE INDEX IF NOT EXISTS ix_test_results_test_id ON test_results (test_id);
CREATE INDEX IF NOT EXISTS ix_test_results_test_date ON test_results (test_date);

DROP TRIGGER IF EXISTS test_results_date_not_future;
CREATE TRIGGER test_results_date_not_future BEFORE INSERT ON test_results
WHEN NEW.test_date > date('now', 'localtime')
BEGIN SELECT RAISE(ABORT, 'test_date cannot be in the future'); END;

DROP TRIGGER IF EXISTS test_results_date_matches_season;
CREATE TRIGGER test_results_date_matches_season BEFORE INSERT ON test_results
WHEN EXISTS (
    SELECT 1 FROM enrolments e
    WHERE e.enrolment_id = NEW.enrolment_id
      AND e.season_year <> CAST(substr(NEW.test_date, 1, 4) AS INTEGER)
)
BEGIN SELECT RAISE(ABORT, 'test_date must fall in the enrolment season_year'); END;

DROP TRIGGER IF EXISTS test_results_score_matches_unit;
CREATE TRIGGER test_results_score_matches_unit BEFORE INSERT ON test_results
WHEN {_SCORE_MISMATCHES_UNIT}
BEGIN SELECT RAISE(ABORT, 'score does not match the test unit (use level and shuttle for Level:Shuttle tests, or value for other units)'); END;

DROP TRIGGER IF EXISTS test_results_score_matches_unit_on_update;
CREATE TRIGGER test_results_score_matches_unit_on_update
BEFORE UPDATE OF value, level, shuttle ON test_results
WHEN {_SCORE_MISMATCHES_UNIT}
BEGIN SELECT RAISE(ABORT, 'score does not match the test unit (use level and shuttle for Level:Shuttle tests, or value for other units)'); END;

DROP TRIGGER IF EXISTS test_results_value_in_range;
CREATE TRIGGER test_results_value_in_range BEFORE INSERT ON test_results
WHEN {_VALUE_OUTSIDE_RANGE}
BEGIN SELECT RAISE(ABORT, 'value is outside the allowed range for this fitness test'); END;

DROP TRIGGER IF EXISTS test_results_value_in_range_on_update;
CREATE TRIGGER test_results_value_in_range_on_update
BEFORE UPDATE OF value ON test_results
WHEN {_VALUE_OUTSIDE_RANGE}
BEGIN SELECT RAISE(ABORT, 'value is outside the allowed range for this fitness test'); END;

DROP TRIGGER IF EXISTS test_results_lock_identity;
CREATE TRIGGER test_results_lock_identity
BEFORE UPDATE OF result_id, enrolment_id, test_id, test_date, created_at ON test_results
BEGIN SELECT RAISE(ABORT, 'test result identity details are locked'); END;
"""

_SELECT = (
    "SELECT result_id, enrolment_id, test_id, test_date, value, level, shuttle, created_at "
    "FROM test_results"
)

_DATAFRAME_RAW_SCHEMA = {
    "result_id": pl.Int64,
    "enrolment_id": pl.Int64,
    "test_id": pl.Int64,
    "test_date": pl.String,
    "value": pl.Float64,
    "level": pl.Int64,
    "shuttle": pl.Int64,
    "created_at": pl.String,
}


# ---- The repository ----------------------------------------------------------------
class TestResultRepository(Repository):
    """Saves and loads fitness test results. The rest of the project never writes SQL for this table."""

    __test__ = False  # class name starts with "Test"; prevent pytest from collecting it

    def __init__(self, db: Database) -> None:
        EnrolmentRepository(db)  # ensures students and enrolments exist first
        FitnessTestRepository(db)  # ensures fitness_tests exists first
        super().__init__(db)

    @property
    def schema_sql(self) -> str:
        return _SCHEMA_SQL

    def add(self, result: TestResult) -> TestResult:
        """Insert one test result and return it with the database-generated result_id."""
        return self.add_many([result])[0]

    def add_many(self, results: Iterable[TestResult]) -> list[TestResult]:
        """Insert several test results all-or-nothing: if one fails, none are saved."""
        batch = list(results)
        for item in batch:
            if not isinstance(item, TestResult):
                raise ValidationError("only TestResult objects can be added")
            if item.result_id is not None:
                raise ValidationError("result_id is assigned by the database; do not supply one")
        saved: list[TestResult] = []
        current: TestResult | None = None
        try:
            with self._db.transaction() as conn:
                for current in batch:
                    cursor = conn.execute(
                        "INSERT INTO test_results (enrolment_id, test_id, test_date, value, level, shuttle) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            current.enrolment_id,
                            current.test_id,
                            current.test_date.isoformat(),
                            current.value,
                            current.level,
                            current.shuttle,
                        ),
                    )
                    new_id = cursor.lastrowid
                    if new_id is None:
                        raise RuntimeError("the database did not return a result_id")
                    saved.append(replace(current, result_id=new_id))
        except sqlite3.IntegrityError as exc:
            assert current is not None
            raise self._translate(exc, current) from None
        return saved

    def get(self, result_id: int) -> TestResult:
        _check_positive_int(result_id, "result_id")
        rows = self._db.query(_SELECT + " WHERE result_id = ?", (result_id,))
        if not rows:
            raise NotFoundError(f"result_id {result_id} does not exist")
        return TestResult.from_row(rows[0])

    def all(
        self, *, enrolment_id: int | None = None, test_id: int | None = None
    ) -> list[TestResult]:
        return [TestResult.from_row(r) for r in self._rows(enrolment_id=enrolment_id, test_id=test_id)]

    def for_enrolment(self, enrolment_id: int, *, test_id: int | None = None) -> list[TestResult]:
        _check_positive_int(enrolment_id, "enrolment_id")
        return self.all(enrolment_id=enrolment_id, test_id=test_id)

    def for_student(
        self,
        student_id: str,
        *,
        season_year: int | None = None,
        test_id: int | None = None,
    ) -> list[TestResult]:
        """All results for a student across their enrolments, ordered chronologically."""
        sid = clean_student_id(student_id)
        conditions = ["e.student_id = ?"]
        params: list[object] = [sid]
        if season_year is not None:
            if isinstance(season_year, bool) or not isinstance(season_year, int):
                raise ValidationError("season_year must be a whole number")
            conditions.append("e.season_year = ?")
            params.append(season_year)
        if test_id is not None:
            if isinstance(test_id, bool) or not isinstance(test_id, int):
                raise ValidationError("test_id must be a whole number")
            conditions.append("r.test_id = ?")
            params.append(test_id)
        sql = (
            "SELECT r.result_id, r.enrolment_id, r.test_id, r.test_date, "
            "r.value, r.level, r.shuttle, r.created_at "
            "FROM test_results r JOIN enrolments e ON e.enrolment_id = r.enrolment_id "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY r.test_date, r.enrolment_id, r.test_id"
        )
        return [TestResult.from_row(row) for row in self._db.query(sql, tuple(params))]

    def update_score(
        self,
        result_id: int,
        value: float | None = None,
        *,
        level: int | None = None,
        shuttle: int | None = None,
    ) -> TestResult:
        """Correct the score on an existing test result. Identity details remain locked."""
        _check_positive_int(result_id, "result_id")
        current = self.get(result_id)
        clean_val = float(value) if _is_plain_number(value) else value
        updated = replace(current, value=clean_val, level=level, shuttle=shuttle)
        try:
            with self._db.transaction() as conn:
                conn.execute(
                    "UPDATE test_results SET value = ?, level = ?, shuttle = ? WHERE result_id = ?",
                    (updated.value, updated.level, updated.shuttle, result_id),
                )
        except sqlite3.IntegrityError as exc:
            raise self._translate(exc, updated) from None
        return updated

    def delete(self, result_id: int) -> None:
        """Delete a mistakenly entered test result."""
        _check_positive_int(result_id, "result_id")
        with self._db.transaction() as conn:
            cursor = conn.execute("DELETE FROM test_results WHERE result_id = ?", (result_id,))
        if cursor.rowcount == 0:
            raise NotFoundError(f"result_id {result_id} does not exist")

    def to_dataframe(
        self, *, enrolment_id: int | None = None, test_id: int | None = None
    ) -> pl.DataFrame:
        """The table as a Polars dataframe with proper Date/Float64/Int64/Datetime types."""
        rows = self._rows(enrolment_id=enrolment_id, test_id=test_id)
        df = pl.DataFrame(rows, schema=_DATAFRAME_RAW_SCHEMA, orient="row")
        return df.with_columns(
            pl.col("test_date").str.to_date("%Y-%m-%d"),
            pl.col("created_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"),
        )

    # -- private helpers --
    def _rows(
        self, *, enrolment_id: int | None = None, test_id: int | None = None
    ) -> list[tuple]:
        conditions: list[str] = []
        params: list[object] = []
        if enrolment_id is not None:
            _check_positive_int(enrolment_id, "enrolment_id")
            conditions.append("enrolment_id = ?")
            params.append(enrolment_id)
        if test_id is not None:
            if isinstance(test_id, bool) or not isinstance(test_id, int):
                raise ValidationError("test_id must be a whole number")
            conditions.append("test_id = ?")
            params.append(test_id)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        return self._db.query(
            _SELECT + where + " ORDER BY test_date, enrolment_id, test_id", tuple(params)
        )

    def _translate(self, exc: sqlite3.IntegrityError, result: TestResult) -> Exception:
        """Turn a raw database error into a clear project error (uses SQLite's error code)."""
        code = exc.sqlite_errorname
        if code == "SQLITE_CONSTRAINT_FOREIGNKEY":
            has_enrolment = self._db.query(
                "SELECT 1 FROM enrolments WHERE enrolment_id = ?", (result.enrolment_id,)
            )
            if not has_enrolment:
                return NotFoundError(f"enrolment_id {result.enrolment_id} does not exist")
            return NotFoundError(f"test_id {result.test_id} does not exist")
        if code == "SQLITE_CONSTRAINT_UNIQUE":
            return DuplicateError(
                f"enrolment_id {result.enrolment_id} already has a result for "
                f"test_id {result.test_id} on {result.test_date.isoformat()}"
            )
        if code == "SQLITE_CONSTRAINT_TRIGGER":
            return ValidationError(str(exc))
        return ValidationError(f"database refused this test result: {exc}")
