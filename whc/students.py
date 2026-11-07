"""students.py - everything about students.

    Sex:                 the allowed values

    Student:             one validated student (cannot be edited once created)

    StudentRepository:   saves and loads students (the only code that touches the table)

    Shared helpers (also used by enrolments.py):
        require_text, clean_student_id, parse_enum, ID_PATTERN

Where each rule lives (so you always know where to look):
    * Python  -> friendly, specific error messages for typed input.
    * SQL     -> the same rules again, so bad data cannot get in even if Python is bypassed.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import TypeVar

import polars as pl

from whc.database import Repository
from whc.exceptions import DuplicateError, NotFoundError, ValidationError

# ---- Settings (change rules here) --------------------------------------------------
DATE_FORMAT = "%d/%m/%Y"  # how a DOB is typed in
ID_PATTERN = re.compile(r"[A-Z]{3}[0-9]{4}")  # AAA0000 - always used with fullmatch()
NAME_PATTERN = re.compile(r"[^\W\d_]+(?:[ '’-][^\W\d_]+)*")  # letters joined by space, - or '
NAME_MAX_LENGTH = 50
MIN_ENTRY_AGE, MAX_ENTRY_AGE = 4, 25  # accepted only when a student is first added
MIN_DOB = date(1900, 1, 1)  # earliest date of birth the database will store

# NOTE on changing settings: the SQL below is only written when a table is FIRST created.
# Changing NAME_MAX_LENGTH, MIN_DOB or the Sex values later does not alter an existing
# whc.db table (SQLite cannot edit a CHECK). Rebuild the table (or the file) if you change them.


def _today() -> date:
    """The one place 'today' comes from, so tests can freeze it."""
    return date.today()


E = TypeVar("E", bound=StrEnum)


class Sex(StrEnum):
    """Allowed values. The SQL CHECK on `sex` is generated from this enum."""

    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"


# ---- Small pure helpers (shared with enrolments.py) --------------------------------
def require_text(value: object, label: str) -> str:
    """Return `value` if it is text; otherwise raise our own error (not a raw TypeError)."""
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be text")
    return value


def clean_student_id(value: object) -> str:
    """Typed ID -> trimmed and uppercased. Does NOT check the format (the classes do that)."""
    return require_text(value, "student_id").strip().upper()


def parse_enum(enum_class: type[E], value: object, label: str) -> E:
    """Match typed text to an enum member, ignoring capitals and extra spaces.

    'cross  country' -> Sport.CROSS_COUNTRY.  Anything else (including None) -> ValidationError.
    """
    wanted = " ".join(value.split()).casefold() if isinstance(value, str) else None
    for member in enum_class:
        if member.value.casefold() == wanted:
            return member
    raise ValidationError(f"{label} must be one of {[m.value for m in enum_class]}")


def _tidy(text: object, label: str) -> str:
    """Trim, collapse repeated whitespace (incl. newlines) and use one form for accents.

    NFC matters: 'é' can be typed as one character or as 'e' + a combining accent.
    Without normalising, the two would look like different names.
    """
    return unicodedata.normalize("NFC", " ".join(require_text(text, label).split()))


def _parse_dob(text: object) -> date:
    try:
        return datetime.strptime(require_text(text, "dob").strip(), DATE_FORMAT).date()
    except ValueError:
        raise ValidationError("dob must be a real date written dd/mm/yyyy") from None


def _sql_list(values: list[str]) -> str:
    """['a', 'b'] -> "'a', 'b'" for an SQL IN (...) list. Only used with OUR OWN constants."""
    return ", ".join(f"'{value}'" for value in values)


# ---- The Student (value object) ----------------------------------------------------
@dataclass(frozen=True, slots=True)
class Student:
    """One student. Frozen: once created it cannot be changed, which matches the rule
    that identity details are locked. Invalid students cannot exist."""

    student_id: str
    first_name: str
    last_name: str
    dob: date
    sex: Sex
    is_active: bool = True

    def __post_init__(self) -> None:
        """Rules that must hold for EVERY student, however it was created.

        Types are checked first so a wrong type gives a ValidationError, not a TypeError.
        """
        if not isinstance(self.student_id, str) or not ID_PATTERN.fullmatch(self.student_id):
            raise ValidationError("student_id must be 3 capital letters then 4 digits, e.g. AAA0000")
        for label, name in (("first_name", self.first_name), ("last_name", self.last_name)):
            # fullmatch (not match + "$"): "$" would also accept a trailing newline.
            if not isinstance(name, str) or len(name) > NAME_MAX_LENGTH or not NAME_PATTERN.fullmatch(name):
                raise ValidationError(
                    f"{label} must be letters only (single spaces, - and ' allowed), "
                    f"max {NAME_MAX_LENGTH} characters"
                )
        if not isinstance(self.sex, Sex):
            raise ValidationError("sex must be a Sex value")
        if not isinstance(self.is_active, bool):
            raise ValidationError("is_active must be True or False")
        # datetime is a subclass of date, so it is excluded explicitly.
        if not isinstance(self.dob, date) or isinstance(self.dob, datetime):
            raise ValidationError("dob must be a date")
        if self.dob < MIN_DOB:
            raise ValidationError(f"dob must be on or after {MIN_DOB.isoformat()}")
        if self.dob >= _today():
            raise ValidationError("dob must be in the past")

    @classmethod
    def new(cls, student_id: str, first_name: str, last_name: str, dob: str, sex: str,
            *, is_active: bool = True) -> Student:
        """Build a student from raw typed input. Cleans it, then applies the entry-only age rule."""
        student = cls(
            student_id=clean_student_id(student_id),
            first_name=_tidy(first_name, "first_name"),
            last_name=_tidy(last_name, "last_name"),
            dob=_parse_dob(dob),
            sex=parse_enum(Sex, sex, "sex"),
            is_active=is_active,
        )
        age = student.age()
        if not MIN_ENTRY_AGE <= age <= MAX_ENTRY_AGE:
            raise ValidationError(f"dob gives an age of {age}; expected {MIN_ENTRY_AGE}-{MAX_ENTRY_AGE}")
        return student

    @classmethod
    def from_row(cls, row: tuple) -> Student:
        """Build a student from a database row (no entry-age rule: old students stay valid).

        Unpacking all 7 columns on purpose: if _SELECT ever changes, this fails loudly
        instead of silently putting a value in the wrong field.
        """
        student_id, first_name, last_name, dob, sex, is_active, _created_at = row
        return cls(student_id, first_name, last_name, date.fromisoformat(dob), Sex(sex), bool(is_active))

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def age(self, on: date | None = None) -> int:
        """Age in whole years on a given date (today by default).

        A 29 February birthday counts as 1 March in years with no 29 February.
        """
        on = on or _today()
        if on < self.dob:
            raise ValidationError("cannot work out an age before the date of birth")
        had_birthday = (on.month, on.day) >= (self.dob.month, self.dob.day)
        return on.year - self.dob.year - (0 if had_birthday else 1)


# ---- The table (SQL) ---------------------------------------------------------------
# Triggers are DROPPED and re-created every start-up so they always match the settings
# above (CREATE TRIGGER IF NOT EXISTS would keep an old, out-of-date version forever).
_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS students (
    student_id  TEXT PRIMARY KEY
                CHECK (student_id GLOB '[A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9]'),
    first_name  TEXT NOT NULL
                CHECK (first_name = trim(first_name, char(32, 9, 10, 13))
                       AND length(first_name) BETWEEN 1 AND {NAME_MAX_LENGTH}),
    last_name   TEXT NOT NULL
                CHECK (last_name = trim(last_name, char(32, 9, 10, 13))
                       AND length(last_name) BETWEEN 1 AND {NAME_MAX_LENGTH}),
    dob         TEXT NOT NULL
                CHECK (date(dob) = dob AND dob >= '{MIN_DOB.isoformat()}'),
    sex         TEXT NOT NULL
                CHECK (sex IN ({_sql_list([s.value for s in Sex])})),
    is_active   INTEGER NOT NULL DEFAULT 1
                CHECK (is_active IN (0, 1)),
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
) STRICT;

-- Backstop for exact (ASCII) duplicates. Accented names are caught in Python (SQLite's
-- lower() only understands A-Z), see StudentRepository._refuse_same_person.
CREATE UNIQUE INDEX IF NOT EXISTS ux_students_person
    ON students (lower(first_name), lower(last_name), dob);

DROP TRIGGER IF EXISTS students_dob_in_past;
CREATE TRIGGER students_dob_in_past BEFORE INSERT ON students
WHEN NEW.dob >= date('now', 'localtime')   -- local time, same clock as Python's date.today()
BEGIN SELECT RAISE(ABORT, 'dob must be in the past'); END;

DROP TRIGGER IF EXISTS students_lock_identity;
CREATE TRIGGER students_lock_identity
BEFORE UPDATE OF student_id, first_name, last_name, dob, sex, created_at ON students
BEGIN SELECT RAISE(ABORT, 'student identity details are locked'); END;

DROP TRIGGER IF EXISTS students_no_delete;
CREATE TRIGGER students_no_delete BEFORE DELETE ON students
BEGIN SELECT RAISE(ABORT, 'students cannot be deleted; set is_active to 0'); END;
"""

_SELECT = (
    "SELECT student_id, first_name, last_name, dob, sex, is_active, created_at FROM students"
)

_DATAFRAME_RAW_SCHEMA = {
    "student_id": pl.String, "first_name": pl.String, "last_name": pl.String,
    "dob": pl.String, "sex": pl.String, "is_active": pl.Int64, "created_at": pl.String,
}


# ---- The repository ----------------------------------------------------------------
class StudentRepository(Repository):
    """Saves and loads students. The rest of the project never writes SQL for this table."""

    @property
    def schema_sql(self) -> str:
        return _SCHEMA_SQL

    def add(self, student: Student) -> None:
        """Save a new student. Raises DuplicateError if the ID or the person already exists."""
        try:
            with self._db.transaction() as conn:
                self._refuse_same_person(conn, student)
                conn.execute(
                    "INSERT INTO students (student_id, first_name, last_name, dob, sex, is_active) "
                    "VALUES (?, ?, ?, ?, ?, ?)",  # ? placeholders: values are never pasted into SQL
                    (student.student_id, student.first_name, student.last_name,
                     student.dob.isoformat(), student.sex.value, int(student.is_active)),
                )
        except sqlite3.IntegrityError as exc:
            raise self._translate(exc, student) from None

    def set_active(self, student_id: str, active: bool) -> None:
        """The only change allowed on an existing student."""
        if not isinstance(active, bool):  # int('no') / 5 would otherwise leak raw errors
            raise ValidationError("active must be True or False")
        student_id = clean_student_id(student_id)
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "UPDATE students SET is_active = ? WHERE student_id = ?", (int(active), student_id)
            )
        if cursor.rowcount == 0:
            raise NotFoundError(f"student_id {student_id} does not exist")

    def get(self, student_id: str) -> Student:
        student_id = clean_student_id(student_id)
        rows = self._db.query(_SELECT + " WHERE student_id = ?", (student_id,))
        if not rows:
            raise NotFoundError(f"student_id {student_id} does not exist")
        return Student.from_row(rows[0])

    def all(self, *, active_only: bool = False) -> list[Student]:
        return [Student.from_row(row) for row in self._rows(active_only)]

    def to_dataframe(self, *, active_only: bool = False) -> pl.DataFrame:
        """The table as a Polars dataframe with proper types (Date, Enum, Boolean, Datetime)."""
        df = pl.DataFrame(self._rows(active_only), schema=_DATAFRAME_RAW_SCHEMA, orient="row")
        return df.with_columns(
            pl.col("dob").str.to_date("%Y-%m-%d"),
            pl.col("sex").cast(pl.Enum([s.value for s in Sex])),
            pl.col("is_active").cast(pl.Boolean),
            pl.col("created_at").str.to_datetime("%Y-%m-%dT%H:%M:%SZ"),
        )

    # -- private helpers --
    def _rows(self, active_only: bool) -> list[tuple]:
        where = " WHERE is_active = 1" if active_only else ""
        return self._db.query(_SELECT + where + " ORDER BY student_id")

    @staticmethod
    def _refuse_same_person(conn: sqlite3.Connection, student: Student) -> None:
        """Same name + same DOB under another ID is the same person, ignoring capitals/accents.

        Done in Python because SQLite's lower() cannot fold 'É' to 'é'. Only students born on
        the same day are compared, so this stays tiny. Same ID is left to the primary key.
        """
        wanted = (student.first_name.casefold(), student.last_name.casefold())
        for student_id, first, last in conn.execute(
            "SELECT student_id, first_name, last_name FROM students WHERE dob = ?",
            (student.dob.isoformat(),),
        ):
            if student_id != student.student_id and (first.casefold(), last.casefold()) == wanted:
                raise DuplicateError(f"{student.full_name} with this DOB is already in the database")

    @staticmethod
    def _translate(exc: sqlite3.IntegrityError, student: Student) -> Exception:
        """Turn a raw database error into a clear project error.

        Uses SQLite's own error code rather than searching the message text.
        """
        code = exc.sqlite_errorname
        if code == "SQLITE_CONSTRAINT_PRIMARYKEY":
            return DuplicateError(f"student_id {student.student_id} already exists")
        if code == "SQLITE_CONSTRAINT_UNIQUE":
            return DuplicateError(f"{student.full_name} with this DOB is already in the database")
        if code == "SQLITE_CONSTRAINT_TRIGGER":
            return ValidationError(str(exc))  # the trigger's own message, e.g. 'dob must be in the past'
        return ValidationError(f"database refused this student: {exc}")