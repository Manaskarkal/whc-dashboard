"""enrolments.py - one row per student, per season, per sport.

    Sport:               default seeded sports (additional sports can be added via add_sport())
    Grade:               allowed school years (each has a numeric `level`, 7-12)
    Enrolment:           one validated enrolment (cannot be edited once created)
    EnrolmentRepository: saves and loads enrolments (the only code that touches the table)

Rules about a single enrolment are checked in Python (friendly messages).
Rules that need OTHER rows (the student's age, their other enrolments) live in SQL triggers,
so they apply to every insert, even one that bypasses this file. The triggers are the one
and only copy of those rules; the repository just turns their messages into ValidationError.

Business rules enforced (each one has tests):
    1. A student can be enrolled in a sport once per season.
    2. The season cannot be more than MAX_YEARS_AHEAD years in the future.
    3. The grade must fit the student's age that season (see AGE_OFFSET_MIN/MAX).
    4. A student has ONE grade per season (several sports, same grade).
    5. A grade can repeat but never go backwards between seasons.
    6. Inactive students cannot get new enrolments.
    7. Sports are database-driven: Lead Coaches can add new sports via `add_sport()`
       with zero code changes.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum

import polars as pl

from whc.database import Database, Repository
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.students import (
    ID_PATTERN,
    StudentRepository,
    clean_student_id,
    parse_enum,
    require_text,
)

# ---- Settings (change rules here) --------------------------------------------------
MIN_SEASON_YEAR = 2000
MAX_SEASON_YEAR = 2100  # broad guard for raw SQL; the real "future" rule is a trigger
MAX_YEARS_AHEAD = 1  # allow planning one season ahead
SPORT_MAX_LENGTH = 50
SPORT_NAME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9 '&/-]*")

# Grade/age rule. A student's "season age" is season_year - birth_year (the age they turn
# during that calendar year). A Year N student is typically N+5 or N+6 (Year 7 -> 12 or 13).
# We allow two years either side, so Year N accepts N+3 ... N+8. Widen if you have
# accelerated or repeating students.
AGE_OFFSET_MIN = 3
AGE_OFFSET_MAX = 8


class Sport(StrEnum):
    """Default seeded sports. Extra sports can be added to the DB at runtime via add_sport()."""

    AFL = "AFL"
    ATHLETICS = "Athletics"
    BADMINTON = "Badminton"
    BASKETBALL = "Basketball"
    CRICKET = "Cricket"
    CROSS_COUNTRY = "Cross Country"
    HOCKEY = "Hockey"
    NETBALL = "Netball"
    RUGBY = "Rugby"
    SOCCER = "Soccer"
    SWIMMING = "Swimming"
    TENNIS = "Tennis"
    VOLLEYBALL = "Volleyball"


class Grade(StrEnum):
    """Allowed school years (fixed Year 7-12)."""

    YEAR_7 = "Year 7"
    YEAR_8 = "Year 8"
    YEAR_9 = "Year 9"
    YEAR_10 = "Year 10"
    YEAR_11 = "Year 11"
    YEAR_12 = "Year 12"

    @property
    def level(self) -> int:
        """'Year 9' -> 9. Used to compare grades ('is Year 7 lower than Year 9?')."""
        return int(self.value.rsplit(" ", 1)[1])


# ---- Small pure helpers ------------------------------------------------------------
def _today() -> date:
    return date.today()


def _latest_allowed_year() -> int:
    return _today().year + MAX_YEARS_AHEAD


def _tidy_sport_name(value: object) -> str:
    """Clean a sport name. Matches default Sport enum values case-insensitively."""
    raw = unicodedata.normalize("NFC", " ".join(require_text(value, "sport").split()))
    for member in Sport:
        if member.value.casefold() == raw.casefold():
            return member.value
    if not raw or len(raw) > SPORT_MAX_LENGTH or not SPORT_NAME_PATTERN.fullmatch(raw):
        raise ValidationError(
            f"sport must be 1-{SPORT_MAX_LENGTH} characters: letters, digits, single spaces "
            "and ' & / - only, starting with a letter"
        )
    return raw


# ---- The Enrolment (value object) --------------------------------------------------
@dataclass(frozen=True, slots=True)
class Enrolment:
    """One student's participation in one sport for one season.

    The ID is None until the repository saves it; the database generates it.
    `sport` is a clean sport name (matches Sport enum values or custom DB-added sports).
    """

    enrolment_id: int | None
    student_id: str
    season_year: int
    sport: str
    grade: Grade

    def __post_init__(self) -> None:
        """Rules that must hold for EVERY enrolment, however it was created."""
        if self.enrolment_id is not None and (
            isinstance(self.enrolment_id, bool)  # True is an int in Python; refuse it
            or not isinstance(self.enrolment_id, int)
            or self.enrolment_id < 1
        ):
            raise ValidationError("enrolment_id must be a positive integer assigned by the database")
        # fullmatch (not match + "$"): "$" would also accept a trailing newline.
        if not isinstance(self.student_id, str) or not ID_PATTERN.fullmatch(self.student_id):
            raise ValidationError("student_id must be 3 capital letters then 4 digits, e.g. AAA0000")
        if isinstance(self.season_year, bool) or not isinstance(self.season_year, int):
            raise ValidationError("season_year must be a whole number")
        latest = _latest_allowed_year()
        if not MIN_SEASON_YEAR <= self.season_year <= latest:
            raise ValidationError(f"season_year must be between {MIN_SEASON_YEAR} and {latest}")
        if (
            not isinstance(self.sport, str)
            or len(self.sport) > SPORT_MAX_LENGTH
            or not SPORT_NAME_PATTERN.fullmatch(self.sport)
            or self.sport != self.sport.strip()
            or "  " in self.sport
        ):
            raise ValidationError(
                f"sport must be 1-{SPORT_MAX_LENGTH} characters: letters, digits, single spaces "
                "and ' & / - only, starting with a letter"
            )
        if not isinstance(self.grade, Grade):
            raise ValidationError("grade must be a Grade value")

    @classmethod
    def new(cls, student_id: str, season_year: int, sport: str, grade: str) -> Enrolment:
        """Build an enrolment from raw typed input. The database assigns the ID."""
        return cls(
            enrolment_id=None,
            student_id=clean_student_id(student_id),
            season_year=season_year,
            sport=_tidy_sport_name(sport),
            grade=parse_enum(Grade, grade, "grade"),
        )

    @classmethod
    def from_row(cls, row: tuple) -> Enrolment:
        """Build an enrolment from a database row (the ID already exists).

        Unpacks all 6 columns on purpose, so a changed _SELECT fails loudly.
        """
        enrolment_id, student_id, season_year, sport, grade, _created_at = row
        return cls(enrolment_id, student_id, int(season_year), sport, Grade(grade))

    @property
    def season_label(self) -> str:
        return f"{self.season_year} {self.sport}"


# ---- The table (SQL) ---------------------------------------------------------------
_BIRTH_YEAR = "(SELECT CAST(substr(dob, 1, 4) AS INTEGER) FROM students WHERE student_id = NEW.student_id)"
_NEW_LEVEL = "(SELECT level FROM grades WHERE name = NEW.grade)"

_GRADE_DOES_NOT_FIT_AGE = f"""NOT COALESCE(
        ((NEW.season_year - {_BIRTH_YEAR})
         BETWEEN {_NEW_LEVEL} + {AGE_OFFSET_MIN} AND {_NEW_LEVEL} + {AGE_OFFSET_MAX}), 1)"""

_GRADE_GOES_BACKWARDS = f"""EXISTS (
        SELECT 1 FROM enrolments e JOIN grades g ON g.name = e.grade
        WHERE e.student_id = NEW.student_id
          AND ((e.season_year < NEW.season_year AND g.level > {_NEW_LEVEL})
            OR (e.season_year > NEW.season_year AND g.level < {_NEW_LEVEL})))"""

_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS sports (
    name TEXT PRIMARY KEY
         CHECK (name = trim(name, char(32, 9, 10, 13))
                AND length(name) BETWEEN 1 AND {SPORT_MAX_LENGTH}
                AND name GLOB '[A-Za-z]*'
                AND name NOT LIKE '%  %')
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_sports_name ON sports (lower(name));

CREATE TABLE IF NOT EXISTS grades (
    name  TEXT PRIMARY KEY,
    level INTEGER NOT NULL UNIQUE
) STRICT;

CREATE TABLE IF NOT EXISTS enrolments (
    enrolment_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- AUTOINCREMENT: an ID is never reused
    student_id   TEXT NOT NULL REFERENCES students(student_id),
    season_year  INTEGER NOT NULL
                 CHECK (season_year BETWEEN {MIN_SEASON_YEAR} AND {MAX_SEASON_YEAR}),
    sport        TEXT NOT NULL REFERENCES sports(name),
    grade        TEXT NOT NULL REFERENCES grades(name),
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_enrolments_student_year_sport
    ON enrolments (student_id, season_year, sport);
CREATE INDEX IF NOT EXISTS ix_enrolments_season ON enrolments (season_year);
CREATE INDEX IF NOT EXISTS ix_enrolments_student_year ON enrolments (student_id, season_year);

-- Retired in this version (its job is now done by enrolments_grade_fits_age).
DROP TRIGGER IF EXISTS enrolments_season_after_birth;

DROP TRIGGER IF EXISTS enrolments_season_not_future;
CREATE TRIGGER enrolments_season_not_future BEFORE INSERT ON enrolments
WHEN NEW.season_year > CAST(strftime('%Y', 'now', 'localtime') AS INTEGER) + {MAX_YEARS_AHEAD}
BEGIN SELECT RAISE(ABORT, 'season_year cannot be in the future'); END;

DROP TRIGGER IF EXISTS enrolments_student_active;
CREATE TRIGGER enrolments_student_active BEFORE INSERT ON enrolments
WHEN (SELECT is_active FROM students WHERE student_id = NEW.student_id) = 0
BEGIN SELECT RAISE(ABORT, 'student is not active; reactivate them before enrolling'); END;

DROP TRIGGER IF EXISTS enrolments_grade_fits_age;
CREATE TRIGGER enrolments_grade_fits_age BEFORE INSERT ON enrolments
WHEN {_GRADE_DOES_NOT_FIT_AGE}
BEGIN SELECT RAISE(ABORT, 'grade does not fit the student age in that season'); END;

DROP TRIGGER IF EXISTS enrolments_grade_fits_age_on_update;
CREATE TRIGGER enrolments_grade_fits_age_on_update BEFORE UPDATE OF grade ON enrolments
WHEN {_GRADE_DOES_NOT_FIT_AGE}
BEGIN SELECT RAISE(ABORT, 'grade does not fit the student age in that season'); END;

DROP TRIGGER IF EXISTS enrolments_one_grade_per_season;
CREATE TRIGGER enrolments_one_grade_per_season BEFORE INSERT ON enrolments
WHEN EXISTS (SELECT 1 FROM enrolments
             WHERE student_id = NEW.student_id AND season_year = NEW.season_year
               AND sport <> NEW.sport AND grade <> NEW.grade)
-- (same sport + different grade is left to the unique index, which reports a duplicate)
BEGIN SELECT RAISE(ABORT, 'student already has a different grade in this season'); END;

DROP TRIGGER IF EXISTS enrolments_grade_not_backwards;
CREATE TRIGGER enrolments_grade_not_backwards BEFORE INSERT ON enrolments
WHEN {_GRADE_GOES_BACKWARDS}
BEGIN SELECT RAISE(ABORT, 'grade cannot go backwards between seasons'); END;

DROP TRIGGER IF EXISTS enrolments_grade_not_backwards_on_update;
CREATE TRIGGER enrolments_grade_not_backwards_on_update BEFORE UPDATE OF grade ON enrolments
WHEN {_GRADE_GOES_BACKWARDS}
BEGIN SELECT RAISE(ABORT, 'grade cannot go backwards between seasons'); END;

DROP TRIGGER IF EXISTS enrolments_lock_identity;
CREATE TRIGGER enrolments_lock_identity
BEFORE UPDATE OF enrolment_id, student_id, season_year, sport, created_at ON enrolments
BEGIN SELECT RAISE(ABORT, 'enrolment identity details are locked'); END;

DROP TRIGGER IF EXISTS enrolments_dob_fits_existing_grades;
CREATE TRIGGER enrolments_dob_fits_existing_grades
BEFORE UPDATE OF dob ON students
WHEN NEW.dob IS NOT NULL AND EXISTS (
    SELECT 1 FROM enrolments e JOIN grades g ON g.name = e.grade
    WHERE e.student_id = NEW.student_id
      AND NOT ((e.season_year - CAST(substr(NEW.dob, 1, 4) AS INTEGER))
               BETWEEN g.level + {AGE_OFFSET_MIN} AND g.level + {AGE_OFFSET_MAX})
)
BEGIN SELECT RAISE(ABORT, 'grade does not fit the student age in that season'); END;
"""

_SELECT = (
    "SELECT enrolment_id, student_id, season_year, sport, grade, created_at FROM enrolments"
)

_DATAFRAME_RAW_SCHEMA = {
    "enrolment_id": pl.Int64, "student_id": pl.String, "season_year": pl.Int64,
    "sport": pl.String, "grade": pl.String, "created_at": pl.String,
}


# ---- The repository ----------------------------------------------------------------
class EnrolmentRepository(Repository):
    """Saves and loads enrolments."""

    def __init__(self, db: Database) -> None:
        StudentRepository(db)  # the parent table must exist first; safe to repeat
        super().__init__(db)  # creates the enrolments tables and triggers
        self._check_table_is_current()
        self._fill_lookup_tables()

    @property
    def schema_sql(self) -> str:
        return _SCHEMA_SQL

    def sports(self) -> list[str]:
        """Return all available sports from the database lookup table, sorted."""
        return [row[0] for row in self._db.query("SELECT name FROM sports ORDER BY name")]

    def add_sport(self, name: str) -> str:
        """Add a new sport to the database lookup table (no code changes needed)."""
        clean = _tidy_sport_name(name)
        try:
            with self._db.transaction() as conn:
                conn.execute("INSERT INTO sports (name) VALUES (?)", (clean,))
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorname in ("SQLITE_CONSTRAINT_PRIMARYKEY", "SQLITE_CONSTRAINT_UNIQUE"):
                raise DuplicateError(f"sport '{clean}' already exists") from None
            raise ValidationError(f"database refused sport '{clean}': {exc}") from None
        return clean

    def add(self, enrolment: Enrolment) -> Enrolment:
        """Insert one enrolment and return it with the database-generated ID."""
        if not isinstance(enrolment, Enrolment):
            raise ValidationError("only Enrolment objects can be added")
        if enrolment.enrolment_id is not None:
            raise ValidationError("enrolment_id is assigned by the database; do not supply one")
        canonical_sport = self._resolve_sport(enrolment.sport)
        enrolment = replace(enrolment, sport=canonical_sport)
        try:
            with self._db.transaction() as conn:
                cursor = conn.execute(
                    "INSERT INTO enrolments (student_id, season_year, sport, grade) "
                    "VALUES (?, ?, ?, ?)",
                    (enrolment.student_id, enrolment.season_year,
                     enrolment.sport, enrolment.grade.value),
                )
                new_id = cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            raise self._translate(exc, enrolment) from None
        if new_id is None:
            raise RuntimeError("the database did not return an enrolment_id")
        return replace(enrolment, enrolment_id=new_id)

    def get(self, enrolment_id: int) -> Enrolment:
        _check_enrolment_id(enrolment_id)
        rows = self._db.query(_SELECT + " WHERE enrolment_id = ?", (enrolment_id,))
        if not rows:
            raise NotFoundError(f"enrolment_id {enrolment_id} does not exist")
        return Enrolment.from_row(rows[0])

    def all(self, *, season_year: int | None = None) -> list[Enrolment]:
        return [Enrolment.from_row(row) for row in self._rows(season_year=season_year)]

    def for_student(self, student_id: str, *, season_year: int | None = None) -> list[Enrolment]:
        rows = self._rows(student_id=clean_student_id(student_id), season_year=season_year)
        return [Enrolment.from_row(row) for row in rows]

    def to_dataframe(self, *, season_year: int | None = None) -> pl.DataFrame:
        """The table as a Polars dataframe with dynamic Enum/Datetime types."""
        df = pl.DataFrame(self._rows(season_year=season_year), schema=_DATAFRAME_RAW_SCHEMA, orient="row")
        return df.with_columns(
            pl.col("sport").cast(pl.Enum(self.sports())),
            pl.col("grade").cast(pl.Enum([g.value for g in Grade])),
            pl.col("created_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def set_grade(self, enrolment_id: int, grade: str | Grade) -> None:
        """Correct a grade. This is the only change allowed on an existing enrolment.

        A grade belongs to the student's SEASON, not to one sport, so this corrects every
        enrolment that student has in the same season in one go.
        """
        _check_enrolment_id(enrolment_id)
        new_grade = grade if isinstance(grade, Grade) else parse_enum(Grade, grade, "grade")
        try:
            with self._db.transaction() as conn:
                cursor = conn.execute(
                    "UPDATE enrolments SET grade = ? "
                    "WHERE (student_id, season_year) = "
                    "      (SELECT student_id, season_year FROM enrolments WHERE enrolment_id = ?)",
                    (new_grade.value, enrolment_id),
                )
        except sqlite3.IntegrityError as exc:
            raise ValidationError(str(exc)) from None  # the trigger's own message
        if cursor.rowcount == 0:
            raise NotFoundError(f"enrolment_id {enrolment_id} does not exist")

    def delete(self, enrolment_id: int) -> None:
        """Delete an enrolment only while it does not leave tested seasons without an enrolment."""
        _check_enrolment_id(enrolment_id)
        try:
            with self._db.transaction() as conn:
                cursor = conn.execute("DELETE FROM enrolments WHERE enrolment_id = ?", (enrolment_id,))
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorname == "SQLITE_CONSTRAINT_TRIGGER":
                raise ValidationError(str(exc)) from None
            raise ValidationError(
                f"cannot delete enrolment_id {enrolment_id} while test results are recorded for it"
            ) from None
        if cursor.rowcount == 0:
            raise NotFoundError(f"enrolment_id {enrolment_id} does not exist")

    # -- private helpers --
    def _resolve_sport(self, sport: str) -> str:
        """Match a sport name case-insensitively against the live `sports` table."""
        rows = self._db.query("SELECT name FROM sports WHERE lower(name) = lower(?)", (sport,))
        if not rows:
            allowed = self.sports()
            raise ValidationError(f"sport must be one of {allowed} (or add it via add_sport())")
        return rows[0][0]

    def _rows(self, *, student_id: str | None = None, season_year: int | None = None) -> list[tuple]:
        """The ONE place that builds the SELECT, so every read orders and filters the same way."""
        conditions: list[str] = []
        params: list[object] = []
        if student_id is not None:
            conditions.append("student_id = ?")
            params.append(student_id)
        if season_year is not None:
            conditions.append("season_year = ?")
            params.append(season_year)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        return self._db.query(
            _SELECT + where + " ORDER BY season_year, student_id, sport", tuple(params)
        )

    def _fill_lookup_tables(self) -> None:
        """Make sure every default Sport and Grade exists in the database (safe to repeat)."""
        with self._db.transaction() as conn:
            conn.executemany("INSERT OR IGNORE INTO sports (name) VALUES (?)",
                             [(s.value,) for s in Sport])
            conn.executemany("INSERT OR IGNORE INTO grades (name, level) VALUES (?, ?)",
                             [(g.value, g.level) for g in Grade])

    def _check_table_is_current(self) -> None:
        """Stop with a clear message if whc.db still has an old enrolments table."""
        linked_tables = {row[2] for row in self._db.query("PRAGMA foreign_key_list(enrolments)")}
        if not {"sports", "grades"} <= linked_tables:
            raise RuntimeError(
                "data/whc.db has an old enrolments table. Delete the file to rebuild it "
                "(or copy the rows across first)."
            )

    def _translate(self, exc: sqlite3.IntegrityError, enrolment: Enrolment) -> Exception:
        """Turn a raw database error into a clear project error (uses SQLite's error code)."""
        code = exc.sqlite_errorname
        if code == "SQLITE_CONSTRAINT_FOREIGNKEY":
            exists = self._db.query(
                "SELECT 1 FROM students WHERE student_id = ?", (enrolment.student_id,)
            )
            if not exists:
                return NotFoundError(f"student_id {enrolment.student_id} does not exist")
            return ValidationError("sport or grade is missing from the lookup tables")
        if code == "SQLITE_CONSTRAINT_UNIQUE":
            return DuplicateError(
                f"student_id {enrolment.student_id} is already enrolled in "
                f"{enrolment.sport} for {enrolment.season_year}"
            )
        if code == "SQLITE_CONSTRAINT_TRIGGER":
            return ValidationError(str(exc))  # the rule's own message, e.g. 'grade cannot go backwards...'
        return ValidationError(f"database refused this enrolment: {exc}")


def _check_enrolment_id(enrolment_id: object) -> None:
    """A real positive int (not True, not '7'), so lookups never see odd types."""
    if isinstance(enrolment_id, bool) or not isinstance(enrolment_id, int):
        raise ValidationError("enrolment_id must be a whole number")
