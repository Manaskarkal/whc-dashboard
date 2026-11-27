"""test_results.py - one row per test measurement (who was tested, in which test, when,
and what they scored).

    TestResult:             one validated test result (frozen once created)
    TestResultRepository:   saves and loads test results (the only code that touches the table)
    check_imtp_warning:     soft >20% warning helper for IMTP Peak vs Relative & Weight (never hard-blocks)

How ownership & dates work (zero assumptions):
    * Linked directly to `student_id` (3NF) — a multi-sport athlete has ONE score per test per day;
      Sport and Grade are joined from `enrolments` on `(student_id, season_year)`.
    * Future entries (`TestResult.new`): require a real `test_date` (`dd/mm/yyyy` input, ISO stored).
      `legacy_year` and `legacy_semester` are forced to NULL by SQLite CHECK; `season_year` and
      `semester` are computed automatically by SQLite VIRTUAL generated columns.
    * Legacy historical entries (`TestResult.legacy`): store `(legacy_year, legacy_semester)` with
      `test_date = NULL` (NO fake placeholder dates!). Only allowed while `db.legacy_import_mode()`
      is active during Phase 1c import; permanently locked by SQLite trigger afterwards.

Business rules enforced in BOTH Python and SQLite:
    1. `student_id` and `test_id` must exist; student must be `is_active = 1`.
    2. Student must have at least one `enrolments` row in that `season_year`.
    3. For all new dated results (`test_date IS NOT NULL`), student must have a recorded `dob`
       (guarantees gradual backfill so every actively tested student has an accurate yearly age).
    4. Only one result per `(student_id, test_id, test_date)` for dated results, and one per
       `(student_id, test_id, legacy_year, legacy_semester)` for legacy results.
    5. Score shape & range must match `fitness_tests.unit` and `[min_value, max_value]`.
    6. Official Level:Shuttle protocols (`level_shuttle_rules`) are enforced by trigger
       (e.g. Yo-Yo IRT1 `15:9` or nonexistent level `7:4` are blocked by SQLite).
    7. Sprint split ordering (`5m < 10m < 20m`) is enforced by trigger on the same testing session
       ONLY when both sprint tests exist (missing splits are ignored, never compared against 0).
    8. Identity details are locked; scores can be corrected with `update_score()` or removed with `delete()`.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date, datetime

import polars as pl

from whc.database import Database, Repository
from whc.enrolments import MAX_SEASON_YEAR, MIN_SEASON_YEAR, EnrolmentRepository
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.fitness_tests import (
    IDS_PER_CATEGORY,
    MAX_TEST_ID,
    VALUE_LIMIT,
    FitnessTestRepository,
    Unit,
    _is_plain_number,
)
from whc.students import DATE_FORMAT, ID_PATTERN, clean_student_id, require_text

# ---- Settings (change rules here) --------------------------------------------------
MIN_TEST_DATE = date(2000, 1, 1)
LEVEL_MIN, LEVEL_MAX = 1, 30
SHUTTLE_MIN, SHUTTLE_MAX = 1, 20
IMTP_WARNING_TOLERANCE = 0.20  # >20% deviation triggers a soft warning for coach confirmation
_LEVEL_SHUTTLE_TEXT = re.compile(r"\s*([0-9]+)\s*[:.]\s*([0-9]+)\s*")

# Sprint test IDs for split-ordering checks (5m < 10m < 20m)
SPRINT_5M_ID = 201
SPRINT_10M_ID = 202
SPRINT_20M_ID = 203
IMTP_PEAK_ID = 1
IMTP_REL_ID = 2
WEIGHT_ID = 902


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


def check_imtp_warning(
    peak_newtons: float | None,
    relative_score: float | None,
    weight_kg: float | None,
    *,
    tolerance: float = IMTP_WARNING_TOLERANCE,
) -> str | None:
    """Soft physics check for IMTP Relative vs Peak / (Weight * 9.81).

    Returns a clear warning message if the recorded Relative score differs by more than
    `tolerance` (default 20%) from `Peak / (Weight * 9.81)`, otherwise None.
    Never blocks saving — allows coaches to verify and confirm if body weight changed.
    """
    if peak_newtons is None or relative_score is None or weight_kg is None:
        return None
    if weight_kg <= 0 or relative_score <= 0:
        return None
    expected_rel = peak_newtons / (weight_kg * 9.81)
    pct_diff = abs(expected_rel - relative_score) / relative_score
    if pct_diff > tolerance:
        return (
            f"Warning: IMTP Relative ({relative_score:g}) differs by {pct_diff * 100:.0f}% "
            f"from Peak ({peak_newtons:g} N) / (Weight {weight_kg:g} kg x 9.81) = {expected_rel:.2f}. "
            "Please check and confirm if the athlete's testing-day weight changed."
        )
    return None


# ---- The TestResult (value object) -------------------------------------------------
@dataclass(frozen=True, slots=True)
class TestResult:
    """One validated test measurement. Frozen: once created it cannot be changed in memory."""

    __test__ = False  # class name starts with "Test"; prevent pytest from collecting it

    result_id: int | None
    student_id: str
    test_id: int
    test_date: date | None
    legacy_year: int | None = None
    legacy_semester: int | None = None
    value: float | None = None
    level: int | None = None
    shuttle: int | None = None

    def __post_init__(self) -> None:
        """Rules that must hold for EVERY TestResult, however it was created."""
        if self.result_id is not None:
            _check_positive_int(self.result_id, "result_id")
        if not isinstance(self.student_id, str) or not ID_PATTERN.fullmatch(self.student_id):
            raise ValidationError("student_id must be 3 capital letters then 4 digits, e.g. AAA0000")
        if isinstance(self.test_id, bool) or not isinstance(self.test_id, int):
            raise ValidationError("test_id must be a whole number")
        if not 1 <= self.test_id <= MAX_TEST_ID or self.test_id % IDS_PER_CATEGORY == 0:
            raise ValidationError(
                f"test_id must be between 1 and {MAX_TEST_ID} and cannot be a multiple of "
                f"{IDS_PER_CATEGORY}"
            )

        # Mutually exclusive temporal fields: either real test_date OR (legacy_year, legacy_semester)
        has_date = self.test_date is not None
        has_legacy = self.legacy_year is not None or self.legacy_semester is not None
        if has_date and has_legacy:
            raise ValidationError(
                "supply either test_date (for new results) OR legacy_year + legacy_semester "
                "(for legacy results), never both"
            )
        if not has_date and not has_legacy:
            raise ValidationError("test_date is required")

        if has_date:
            if not isinstance(self.test_date, date) or isinstance(self.test_date, datetime):
                raise ValidationError("test_date must be a date")
            if self.test_date < MIN_TEST_DATE:
                raise ValidationError(f"test_date must be on or after {MIN_TEST_DATE.isoformat()}")
            if self.test_date > _today():
                raise ValidationError("test_date cannot be in the future")
        else:
            if isinstance(self.legacy_year, bool) or not isinstance(self.legacy_year, int):
                raise ValidationError("legacy_year must be a whole number")
            if not MIN_SEASON_YEAR <= self.legacy_year <= _today().year:
                raise ValidationError(
                    f"legacy_year must be between {MIN_SEASON_YEAR} and {_today().year}"
                )
            if isinstance(self.legacy_semester, bool) or self.legacy_semester not in (1, 2):
                raise ValidationError("legacy_semester must be 1 or 2")

        _check_score_fields(self.value, self.level, self.shuttle)

    @classmethod
    def new(
        cls,
        student_id: str,
        test_id: int,
        test_date: str,
        value: float | None = None,
        *,
        level: int | None = None,
        shuttle: int | None = None,
    ) -> TestResult:
        """Build a NEW dated test result. Coaches can ONLY supply a real `test_date`."""
        return cls(
            result_id=None,
            student_id=clean_student_id(student_id),
            test_id=test_id,
            test_date=_parse_test_date(test_date),
            legacy_year=None,
            legacy_semester=None,
            value=float(value) if _is_plain_number(value) else value,
            level=level,
            shuttle=shuttle,
        )

    @classmethod
    def legacy(
        cls,
        student_id: str,
        test_id: int,
        season_year: int,
        semester: int,
        value: float | None = None,
        *,
        level: int | None = None,
        shuttle: int | None = None,
    ) -> TestResult:
        """Build a legacy historical test result (Year + Semester only, test_date=None).

        Can only be inserted into SQLite while `db.legacy_import_mode()` is active.
        """
        return cls(
            result_id=None,
            student_id=clean_student_id(student_id),
            test_id=test_id,
            test_date=None,
            legacy_year=season_year,
            legacy_semester=semester,
            value=float(value) if _is_plain_number(value) else value,
            level=level,
            shuttle=shuttle,
        )

    @classmethod
    def from_level_shuttle_text(
        cls, student_id: str, test_id: int, test_date: str, text: str
    ) -> TestResult:
        """Build a dated Level:Shuttle result from '16:3' or '16.3' notation."""
        raw = require_text(text, "level_shuttle")
        match = _LEVEL_SHUTTLE_TEXT.fullmatch(raw)
        if not match:
            raise ValidationError(
                "level_shuttle must be written as 'level:shuttle' or 'level.shuttle', e.g. '16:3'"
            )
        return cls.new(
            student_id=student_id,
            test_id=test_id,
            test_date=test_date,
            level=int(match.group(1)),
            shuttle=int(match.group(2)),
        )

    @classmethod
    def from_row(cls, row: tuple) -> TestResult:
        """Build a TestResult from a database row (unpacks all 10 stored columns)."""
        (
            result_id,
            student_id,
            test_id,
            test_date,
            legacy_year,
            legacy_semester,
            value,
            level,
            shuttle,
            _created_at,
        ) = row
        return cls(
            result_id=result_id,
            student_id=student_id,
            test_id=int(test_id),
            test_date=date.fromisoformat(test_date) if test_date is not None else None,
            legacy_year=int(legacy_year) if legacy_year is not None else None,
            legacy_semester=int(legacy_semester) if legacy_semester is not None else None,
            value=value,
            level=level,
            shuttle=shuttle,
        )

    @property
    def is_legacy(self) -> bool:
        return self.test_date is None

    @property
    def season_year(self) -> int:
        if self.test_date is not None:
            return self.test_date.year
        assert self.legacy_year is not None
        return self.legacy_year

    @property
    def semester(self) -> int:
        """1 for Jan-Jun, 2 for Jul-Dec (or legacy_semester for historical rows)."""
        if self.test_date is not None:
            return 1 if self.test_date.month <= 6 else 2
        assert self.legacy_semester is not None
        return self.legacy_semester

    @property
    def period_label(self) -> str:
        """Human-friendly date/period label (never fabricates a day for legacy rows)."""
        if self.test_date is not None:
            return self.test_date.strftime(DATE_FORMAT)
        return f"{self.season_year} Sem {self.semester}"

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

_INVALID_LEVEL_SHUTTLE_PROTOCOL = """NEW.level IS NOT NULL
AND EXISTS (SELECT 1 FROM level_shuttle_rules WHERE test_id = NEW.test_id)
AND NOT EXISTS (
    SELECT 1 FROM level_shuttle_rules
    WHERE test_id = NEW.test_id
      AND level = NEW.level
      AND NEW.shuttle BETWEEN 1 AND max_shuttle
)"""

_SAME_SESSION_PREDICATE = """r.student_id = NEW.student_id
  AND (
      (NEW.test_date IS NOT NULL AND r.test_date = NEW.test_date)
      OR (NEW.test_date IS NULL AND r.test_date IS NULL
          AND r.legacy_year = NEW.legacy_year
          AND r.legacy_semester = NEW.legacy_semester)
  )"""

_SPRINT_ORDER_VIOLATED = f"""NEW.value IS NOT NULL
AND NEW.test_id IN ({SPRINT_5M_ID}, {SPRINT_10M_ID}, {SPRINT_20M_ID})
AND EXISTS (
    SELECT 1 FROM test_results r
    WHERE {_SAME_SESSION_PREDICATE}
      AND r.value IS NOT NULL
      AND (
          (NEW.test_id = {SPRINT_5M_ID} AND r.test_id IN ({SPRINT_10M_ID}, {SPRINT_20M_ID}) AND NEW.value >= r.value)
       OR (NEW.test_id = {SPRINT_10M_ID} AND r.test_id = {SPRINT_5M_ID} AND NEW.value <= r.value)
       OR (NEW.test_id = {SPRINT_10M_ID} AND r.test_id = {SPRINT_20M_ID} AND NEW.value >= r.value)
       OR (NEW.test_id = {SPRINT_20M_ID} AND r.test_id IN ({SPRINT_5M_ID}, {SPRINT_10M_ID}) AND NEW.value <= r.value)
      )
)"""

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS test_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      TEXT NOT NULL REFERENCES students(student_id),
    test_id         INTEGER NOT NULL REFERENCES fitness_tests(test_id),
    test_date       TEXT
                    CHECK (test_date IS NULL
                           OR (date(test_date) = test_date AND test_date >= '{MIN_TEST_DATE.isoformat()}')),
    legacy_year     INTEGER
                    CHECK (legacy_year IS NULL OR legacy_year BETWEEN {MIN_SEASON_YEAR} AND {MAX_SEASON_YEAR}),
    legacy_semester INTEGER
                    CHECK (legacy_semester IS NULL OR legacy_semester IN (1, 2)),
    season_year     INTEGER GENERATED ALWAYS AS (
                        COALESCE(CAST(substr(test_date, 1, 4) AS INTEGER), legacy_year)
                    ) VIRTUAL,
    semester        INTEGER GENERATED ALWAYS AS (
                        CASE
                            WHEN test_date IS NULL THEN legacy_semester
                            WHEN CAST(substr(test_date, 6, 2) AS INTEGER) <= 6 THEN 1
                            ELSE 2
                        END
                    ) VIRTUAL,
    value           REAL,
    level           INTEGER,
    shuttle         INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    CONSTRAINT temporal_shape CHECK (
        (test_date IS NOT NULL AND legacy_year IS NULL AND legacy_semester IS NULL)
        OR (test_date IS NULL AND legacy_year IS NOT NULL AND legacy_semester IS NOT NULL)
    ),
    CONSTRAINT score_shape CHECK (
        (value IS NOT NULL AND value BETWEEN -{VALUE_LIMIT} AND {VALUE_LIMIT}
         AND level IS NULL AND shuttle IS NULL)
        OR (value IS NULL
            AND level BETWEEN {LEVEL_MIN} AND {LEVEL_MAX}
            AND shuttle BETWEEN {SHUTTLE_MIN} AND {SHUTTLE_MAX})
    )
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_test_results_dated
    ON test_results (student_id, test_id, test_date)
    WHERE test_date IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_test_results_legacy
    ON test_results (student_id, test_id, legacy_year, legacy_semester)
    WHERE test_date IS NULL;
CREATE INDEX IF NOT EXISTS ix_test_results_student_id ON test_results (student_id);
CREATE INDEX IF NOT EXISTS ix_test_results_test_id ON test_results (test_id);
CREATE INDEX IF NOT EXISTS ix_test_results_test_date ON test_results (test_date);

DROP TRIGGER IF EXISTS test_results_require_date_when_locked;
CREATE TRIGGER test_results_require_date_when_locked BEFORE INSERT ON test_results
WHEN NEW.test_date IS NULL
  AND COALESCE((SELECT value FROM system_config WHERE key = 'allow_legacy_import'), '0') <> '1'
BEGIN SELECT RAISE(ABORT, 'test_date is required for all new test results'); END;

DROP TRIGGER IF EXISTS test_results_date_not_future;
CREATE TRIGGER test_results_date_not_future BEFORE INSERT ON test_results
WHEN NEW.test_date IS NOT NULL AND NEW.test_date > date('now', 'localtime')
BEGIN SELECT RAISE(ABORT, 'test_date cannot be in the future'); END;

DROP TRIGGER IF EXISTS test_results_student_active;
CREATE TRIGGER test_results_student_active BEFORE INSERT ON test_results
WHEN (SELECT is_active FROM students WHERE student_id = NEW.student_id) = 0
BEGIN SELECT RAISE(ABORT, 'student is not active; reactivate them before recording results'); END;

DROP TRIGGER IF EXISTS test_results_require_enrolment_in_season;
CREATE TRIGGER test_results_require_enrolment_in_season BEFORE INSERT ON test_results
WHEN EXISTS (SELECT 1 FROM students WHERE student_id = NEW.student_id)
  AND (NEW.test_date IS NULL OR date(NEW.test_date) = NEW.test_date)
  AND NOT EXISTS (
      SELECT 1 FROM enrolments e
      WHERE e.student_id = NEW.student_id
        AND e.season_year = COALESCE(CAST(substr(NEW.test_date, 1, 4) AS INTEGER), NEW.legacy_year)
  )
BEGIN SELECT RAISE(ABORT, 'student must have an enrolment in that season_year before recording results'); END;

DROP TRIGGER IF EXISTS test_results_require_dob_for_new_results;
CREATE TRIGGER test_results_require_dob_for_new_results BEFORE INSERT ON test_results
WHEN NEW.test_date IS NOT NULL
  AND EXISTS (SELECT 1 FROM students WHERE student_id = NEW.student_id AND dob IS NULL)
BEGIN SELECT RAISE(ABORT, 'student dob must be recorded before adding new test results'); END;

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

DROP TRIGGER IF EXISTS test_results_level_shuttle_protocol;
CREATE TRIGGER test_results_level_shuttle_protocol BEFORE INSERT ON test_results
WHEN {_INVALID_LEVEL_SHUTTLE_PROTOCOL}
BEGIN SELECT RAISE(ABORT, 'level:shuttle is not valid for this test protocol'); END;

DROP TRIGGER IF EXISTS test_results_level_shuttle_protocol_on_update;
CREATE TRIGGER test_results_level_shuttle_protocol_on_update
BEFORE UPDATE OF level, shuttle ON test_results
WHEN {_INVALID_LEVEL_SHUTTLE_PROTOCOL}
BEGIN SELECT RAISE(ABORT, 'level:shuttle is not valid for this test protocol'); END;

DROP TRIGGER IF EXISTS test_results_sprint_split_order;
CREATE TRIGGER test_results_sprint_split_order BEFORE INSERT ON test_results
WHEN {_SPRINT_ORDER_VIOLATED}
BEGIN SELECT RAISE(ABORT, 'sprint split times must satisfy 5m < 10m < 20m on the same testing session'); END;

DROP TRIGGER IF EXISTS test_results_sprint_split_order_on_update;
CREATE TRIGGER test_results_sprint_split_order_on_update
BEFORE UPDATE OF value ON test_results
WHEN {_SPRINT_ORDER_VIOLATED}
BEGIN SELECT RAISE(ABORT, 'sprint split times must satisfy 5m < 10m < 20m on the same testing session'); END;

DROP TRIGGER IF EXISTS test_results_lock_identity;
CREATE TRIGGER test_results_lock_identity
BEFORE UPDATE OF result_id, student_id, test_id, test_date, legacy_year, legacy_semester, created_at ON test_results
BEGIN SELECT RAISE(ABORT, 'test result identity details are locked'); END;

-- Cross-table protection: cannot delete a student's LAST enrolment in a season with test results
DROP TRIGGER IF EXISTS enrolments_protect_tested_season;
CREATE TRIGGER enrolments_protect_tested_season BEFORE DELETE ON enrolments
WHEN EXISTS (
    SELECT 1 FROM test_results r
    WHERE r.student_id = OLD.student_id AND r.season_year = OLD.season_year
) AND NOT EXISTS (
    SELECT 1 FROM enrolments e
    WHERE e.student_id = OLD.student_id
      AND e.season_year = OLD.season_year
      AND e.enrolment_id <> OLD.enrolment_id
)
BEGIN SELECT RAISE(ABORT, 'cannot delete the last enrolment for a season that has test results'); END;

-- Cross-table protection: narrowing fitness_tests range cannot invalidate existing results
DROP TRIGGER IF EXISTS fitness_tests_range_fits_existing_results;
CREATE TRIGGER fitness_tests_range_fits_existing_results
BEFORE UPDATE OF min_value, max_value ON fitness_tests
WHEN NEW.min_value IS NOT NULL AND EXISTS (
    SELECT 1 FROM test_results r
    WHERE r.test_id = NEW.test_id
      AND r.value IS NOT NULL
      AND NOT (r.value BETWEEN NEW.min_value AND NEW.max_value)
)
BEGIN SELECT RAISE(ABORT, 'new min_value/max_value would invalidate existing test results'); END;
"""

_STORED_COLUMNS = (
    "result_id", "student_id", "test_id", "test_date",
    "legacy_year", "legacy_semester", "value", "level", "shuttle", "created_at",
)
_SELECT = f"SELECT {', '.join(_STORED_COLUMNS)} FROM test_results"
_SELECT_DF = (
    "SELECT result_id, student_id, test_id, test_date, season_year, semester, "
    "value, level, shuttle, created_at FROM test_results"
)

_DATAFRAME_RAW_SCHEMA = {
    "result_id": pl.Int64,
    "student_id": pl.String,
    "test_id": pl.Int64,
    "test_date": pl.String,
    "season_year": pl.Int64,
    "semester": pl.Int64,
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
        FitnessTestRepository(db)  # ensures fitness_tests + level_shuttle_rules exist first
        super().__init__(db)
        self._check_table_is_current()

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
                        "INSERT INTO test_results "
                        "(student_id, test_id, test_date, legacy_year, legacy_semester, value, level, shuttle) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            current.student_id,
                            current.test_id,
                            current.test_date.isoformat() if current.test_date is not None else None,
                            current.legacy_year,
                            current.legacy_semester,
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
        self,
        *,
        student_id: str | None = None,
        season_year: int | None = None,
        test_id: int | None = None,
    ) -> list[TestResult]:
        return [
            TestResult.from_row(r)
            for r in self._rows(student_id=student_id, season_year=season_year, test_id=test_id)
        ]

    def for_student(
        self,
        student_id: str,
        *,
        season_year: int | None = None,
        test_id: int | None = None,
    ) -> list[TestResult]:
        """All results for a student, ordered chronologically."""
        return self.all(student_id=student_id, season_year=season_year, test_id=test_id)

    def check_imtp_session_warning(
        self,
        student_id: str,
        test_date: str | date,
        *,
        peak_newtons: float | None = None,
        relative_score: float | None = None,
        weight_kg: float | None = None,
    ) -> str | None:
        """Check whether an IMTP Relative entry deviates >20% from Peak / (Weight * 9.81).

        Looks up any missing values from the student's existing records (same date for Peak/Rel,
        same date or most recent recorded Weight) so the UI can warn and ask for confirmation.
        """
        sid = clean_student_id(student_id)
        tdate_iso = (
            test_date.isoformat()
            if isinstance(test_date, date) and not isinstance(test_date, datetime)
            else _parse_test_date(test_date).isoformat()
        )
        if peak_newtons is None:
            rows = self._db.query(
                "SELECT value FROM test_results WHERE student_id = ? AND test_id = ? AND test_date = ?",
                (sid, IMTP_PEAK_ID, tdate_iso),
            )
            peak_newtons = rows[0][0] if rows else None
        if relative_score is None:
            rows = self._db.query(
                "SELECT value FROM test_results WHERE student_id = ? AND test_id = ? AND test_date = ?",
                (sid, IMTP_REL_ID, tdate_iso),
            )
            relative_score = rows[0][0] if rows else None
        if weight_kg is None:
            rows = self._db.query(
                "SELECT value FROM test_results WHERE student_id = ? AND test_id = ? "
                "ORDER BY COALESCE(test_date, printf('%04d-%02d-01', legacy_year, CASE legacy_semester WHEN 1 THEN 3 ELSE 8 END)) DESC "
                "LIMIT 1",
                (sid, WEIGHT_ID),
            )
            weight_kg = rows[0][0] if rows else None
        return check_imtp_warning(peak_newtons, relative_score, weight_kg)

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
        self,
        *,
        student_id: str | None = None,
        season_year: int | None = None,
        test_id: int | None = None,
    ) -> pl.DataFrame:
        """The table as a Polars dataframe with proper Date/Int64/Float64/Datetime types."""
        rows = self._rows(
            student_id=student_id,
            season_year=season_year,
            test_id=test_id,
            for_df=True,
        )
        df = pl.DataFrame(rows, schema=_DATAFRAME_RAW_SCHEMA, orient="row")
        return df.with_columns(
            pl.col("test_date").str.to_date("%Y-%m-%d"),
            pl.col("created_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"),
        )

    # -- private helpers --
    def _check_table_is_current(self) -> None:
        columns = tuple(row[1] for row in self._db.query("PRAGMA table_info(test_results)"))
        if columns != _STORED_COLUMNS:
            raise RuntimeError(
                "data/whc.db has an old test_results table. Rebuild it cleanly before importing."
            )

    def _rows(
        self,
        *,
        student_id: str | None = None,
        season_year: int | None = None,
        test_id: int | None = None,
        for_df: bool = False,
    ) -> list[tuple]:
        conditions: list[str] = []
        params: list[object] = []
        if student_id is not None:
            conditions.append("student_id = ?")
            params.append(clean_student_id(student_id))
        if season_year is not None:
            if isinstance(season_year, bool) or not isinstance(season_year, int):
                raise ValidationError("season_year must be a whole number")
            conditions.append("season_year = ?")
            params.append(season_year)
        if test_id is not None:
            if isinstance(test_id, bool) or not isinstance(test_id, int):
                raise ValidationError("test_id must be a whole number")
            conditions.append("test_id = ?")
            params.append(test_id)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        base = _SELECT_DF if for_df else _SELECT
        return self._db.query(
            base + where + " ORDER BY season_year, semester, COALESCE(test_date, ''), student_id, test_id",
            tuple(params),
        )

    def _translate(self, exc: sqlite3.IntegrityError, result: TestResult) -> Exception:
        """Turn a raw database error into a clear project error (uses SQLite's error code)."""
        code = exc.sqlite_errorname
        if code == "SQLITE_CONSTRAINT_FOREIGNKEY":
            has_student = self._db.query(
                "SELECT 1 FROM students WHERE student_id = ?", (result.student_id,)
            )
            if not has_student:
                return NotFoundError(f"student_id {result.student_id} does not exist")
            return NotFoundError(f"test_id {result.test_id} does not exist")
        if code == "SQLITE_CONSTRAINT_UNIQUE":
            return DuplicateError(
                f"student_id {result.student_id} already has a result for "
                f"test_id {result.test_id} on {result.period_label}"
            )
        if code == "SQLITE_CONSTRAINT_TRIGGER":
            return ValidationError(str(exc))
        return ValidationError(f"database refused this test result: {exc}")
