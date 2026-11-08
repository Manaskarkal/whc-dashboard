"""fitness_tests.py - the catalogue of fitness tests (what can be measured, and in what unit).

    Category:                 the allowed buckets (Strength, Power, Speed, ...)
    Unit:                     the allowed units (Seconds, Newtons, ...)
    FitnessTest:              one validated test (cannot be edited once created)
    DEFAULT_TESTS:            the standard 35-test catalogue, loaded by load_catalogue()
    FitnessTestRepository:    saves and loads tests (the only code that touches the table)

This is a small REFERENCE table: a handful of rows that rarely change. Later, test_results
rows will point at `test_id`, so a test_id is never reused, moved or deleted.

test_id numbering (the first digit tells you the bucket):
      1-99   Strength            400s  Aerobic Endurance
    101-199  Power               500s  Anaerobic Endurance
    201-299  Speed               600s  Muscular Endurance
    301-399  Agility             700s  Flexibility
                                 800s  Reaction
    IDs start at 1 inside each block (a block's own round number - 0, 100, 200 - is not used,
    so no ID is ever 0, which is falsy in Python and easy to lose in an `if test_id:` check).

Business rules enforced (each one has tests):
    1. test_id sits inside its category's block (a Power test cannot be numbered 14).
    2. test_name is unique, ignoring capitals, and made of plain characters only.
    3. category and unit must come from the allowed lists.
    4. test_id, category and unit can never change (results depend on them).
       Only the name can be corrected, using rename().
    5. Tests cannot be deleted.

Where each rule lives: Python gives friendly messages for typed input; SQL repeats the rules
so bad data cannot get in even if Python is bypassed (same approach as students.py).
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum

import polars as pl

from whc.database import Database, Repository
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.students import parse_enum, require_text

# ---- Settings (change rules here) --------------------------------------------------
IDS_PER_CATEGORY = 100  # each category owns a block of 100 numbers
MAX_TEST_ID = 999  # the highest ID the database accepts (10 blocks: 0-99 ... 900-999)
NAME_MAX_LENGTH = 80
# Letters, digits, spaces and . ' ( ) / : -   (enough for "5-0-5 Agility Test Left" and
# "1.6k/Mile Run"). Must start with a letter or digit. Always used with fullmatch().
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .'()/:-]*")

# NOTE on changing settings: like students.py, the SQL below is only written when the table
# is FIRST created. Changing NAME_MAX_LENGTH or MAX_TEST_ID later does not alter an existing
# whc.db table. Rebuild the table (or the file) if you change them.


class Category(StrEnum):
    """The standard buckets. Add a bucket here AND give it a free block in _ID_START."""

    STRENGTH = "Strength"
    POWER = "Power"
    SPEED = "Speed"
    AGILITY = "Agility"
    AEROBIC_ENDURANCE = "Aerobic Endurance"
    ANAEROBIC_ENDURANCE = "Anaerobic Endurance"
    MUSCULAR_ENDURANCE = "Muscular Endurance"
    FLEXIBILITY = "Flexibility"
    REACTION = "Reaction"

    @property
    def id_start(self) -> int:
        """First number of this category's block: Strength 0, Power 100, Speed 200, ..."""
        return _ID_START[self]


# Written out in full (not worked out from the enum's order) so that re-ordering or inserting
# a category above can never silently renumber tests that results already point at.
_ID_START: dict[Category, int] = {
    Category.STRENGTH: 0,
    Category.POWER: 100,
    Category.SPEED: 200,
    Category.AGILITY: 300,
    Category.AEROBIC_ENDURANCE: 400,
    Category.ANAEROBIC_ENDURANCE: 500,
    Category.MUSCULAR_ENDURANCE: 600,
    Category.FLEXIBILITY: 700,
    Category.REACTION: 800,
}


class Unit(StrEnum):
    """How a result is written. 'Shuttles:Levels' results look like '16:3' (shuttle:level)."""

    CENTIMETRES = "Centimetres"
    METRES = "Metres"
    SECONDS = "Seconds"
    NEWTONS = "Newtons"
    KILOGRAMS = "Kilograms"
    WATTS_PER_KILOGRAM = "Watts/Kilogram"
    REPS = "Reps"
    UNITLESS = "Unitless"
    SHUTTLES_LEVELS = "Shuttles:Levels"


# ---- Small pure helpers ------------------------------------------------------------
def _tidy_name(text: object) -> str:
    """Trim, collapse repeated whitespace (incl. newlines) and use one form for accents."""
    return unicodedata.normalize("NFC", " ".join(require_text(text, "test_name").split()))


def _check_test_id(test_id: object) -> None:
    """A real int (not True, not '101'), so lookups never see odd types."""
    if isinstance(test_id, bool) or not isinstance(test_id, int):
        raise ValidationError("test_id must be a whole number")


# ---- The FitnessTest (value object) ------------------------------------------------
@dataclass(frozen=True, slots=True)
class FitnessTest:
    """One fitness test. Frozen: once created it cannot be changed. Invalid tests cannot exist."""

    test_id: int
    test_name: str
    category: Category
    unit: Unit

    def __post_init__(self) -> None:
        """Rules that must hold for EVERY test, however it was created. Types are checked first."""
        _check_test_id(self.test_id)
        if not 1 <= self.test_id <= MAX_TEST_ID or self.test_id % IDS_PER_CATEGORY == 0:
            raise ValidationError(
                f"test_id must be between 1 and {MAX_TEST_ID} and cannot be a multiple of "
                f"{IDS_PER_CATEGORY}"
            )
        # fullmatch (not match + "$"): "$" would also accept a trailing newline.
        if (
            not isinstance(self.test_name, str)
            or len(self.test_name) > NAME_MAX_LENGTH
            or not NAME_PATTERN.fullmatch(self.test_name)
            or self.test_name != self.test_name.strip()
            or "  " in self.test_name
            or not any(ch.isalpha() for ch in self.test_name)
        ):
            raise ValidationError(
                f"test_name must be 1-{NAME_MAX_LENGTH} characters: letters, digits, single "
                "spaces and . ' ( ) / : - only, starting with a letter or digit, with at "
                "least one letter"
            )
        if not isinstance(self.category, Category):
            raise ValidationError("category must be a Category value")
        if not isinstance(self.unit, Unit):
            raise ValidationError("unit must be a Unit value")
        start = self.category.id_start
        if self.test_id // IDS_PER_CATEGORY * IDS_PER_CATEGORY != start:
            raise ValidationError(
                f"test_id {self.test_id} does not belong to {self.category.value}: "
                f"its IDs run {start + 1}-{start + IDS_PER_CATEGORY - 1}"
            )

    @classmethod
    def new(cls, test_id: int, test_name: str, category: str, unit: str) -> FitnessTest:
        """Build a test from raw typed input (cleans the name, matches category and unit)."""
        return cls(
            test_id=test_id,
            test_name=_tidy_name(test_name),
            category=parse_enum(Category, category, "category"),
            unit=parse_enum(Unit, unit, "unit"),
        )

    @classmethod
    def from_row(cls, row: tuple) -> FitnessTest:
        """Build a test from a database row.

        Unpacks all 4 columns on purpose, so a changed _SELECT fails loudly.
        """
        test_id, test_name, category, unit = row
        return cls(test_id, test_name, Category(category), Unit(unit))


# ---- The standard catalogue --------------------------------------------------------
# IDs are written out by hand and must never change once results exist. To add a test, give
# it the next free number in its block (or use FitnessTestRepository.next_id). Building these
# objects at import time means a typo in this list stops the program immediately.
_S, _P, _SP, _AG = Category.STRENGTH, Category.POWER, Category.SPEED, Category.AGILITY
_AE, _AN, _ME = Category.AEROBIC_ENDURANCE, Category.ANAEROBIC_ENDURANCE, Category.MUSCULAR_ENDURANCE
_FL, _RE = Category.FLEXIBILITY, Category.REACTION

DEFAULT_TESTS: tuple[FitnessTest, ...] = (
    # Strength (1-99)
    FitnessTest(1, "Isometric Mid-Thigh Pull Peak", _S, Unit.NEWTONS),
    FitnessTest(2, "Isometric Mid-Thigh Pull Relative", _S, Unit.UNITLESS),
    FitnessTest(3, "Nordic Hamstring Left", _S, Unit.NEWTONS),
    FitnessTest(4, "Nordic Hamstring Right", _S, Unit.NEWTONS),
    FitnessTest(5, "Adduction Left", _S, Unit.NEWTONS),
    FitnessTest(6, "Adduction Right", _S, Unit.NEWTONS),
    FitnessTest(7, "Abduction Left", _S, Unit.NEWTONS),
    FitnessTest(8, "Abduction Right", _S, Unit.NEWTONS),
    FitnessTest(9, "1RM Back Squat", _S, Unit.KILOGRAMS),
    FitnessTest(10, "1RM Bench Press", _S, Unit.KILOGRAMS),
    FitnessTest(11, "1RM Trap Bar Deadlift", _S, Unit.KILOGRAMS),
    FitnessTest(12, "3RM Back Squat", _S, Unit.KILOGRAMS),
    FitnessTest(13, "3RM Bench Press", _S, Unit.KILOGRAMS),
    FitnessTest(14, "3RM Trap Bar Deadlift", _S, Unit.KILOGRAMS),
    # Power (101-199)
    FitnessTest(101, "Counter Movement Jump Height", _P, Unit.CENTIMETRES),
    FitnessTest(102, "Counter Movement Jump Power", _P, Unit.WATTS_PER_KILOGRAM),
    FitnessTest(103, "Counter Movement Jump Relative Strength Index (RSI)", _P, Unit.UNITLESS),
    FitnessTest(104, "Hop Jump Relative Strength Index (RSI)", _P, Unit.UNITLESS),
    FitnessTest(105, "Standing Long Jump (Broad Jump)", _P, Unit.METRES),
    # Speed (201-299)
    FitnessTest(201, "5 Metre Sprint", _SP, Unit.SECONDS),
    FitnessTest(202, "10 Metre Sprint", _SP, Unit.SECONDS),
    FitnessTest(203, "20 Metre Sprint", _SP, Unit.SECONDS),
    # Agility (301-399)
    FitnessTest(301, "5-0-5 Agility Test Left", _AG, Unit.SECONDS),
    FitnessTest(302, "5-0-5 Agility Test Right", _AG, Unit.SECONDS),
    FitnessTest(303, "NBA Lane Agility Test", _AG, Unit.SECONDS),
    # Aerobic Endurance (401-499)
    FitnessTest(401, "YOYO Intermittent Recovery Test 1 (IRT1)", _AE, Unit.SHUTTLES_LEVELS),
    FitnessTest(402, "20m Beep Test (Shuttle Run)", _AE, Unit.SHUTTLES_LEVELS),
    FitnessTest(403, "Cooper 12-Minute Run", _AE, Unit.METRES),
    FitnessTest(404, "1.6k/Mile Run", _AE, Unit.SECONDS),
    FitnessTest(405, "2k Run", _AE, Unit.SECONDS),
    # Anaerobic Endurance (501-599)
    FitnessTest(501, "YOYO Intermittent Recovery Test 2 (IRT2)", _AN, Unit.SHUTTLES_LEVELS),
    # Muscular Endurance (601-699)
    FitnessTest(601, "Push Up Test (1-Minute)", _ME, Unit.REPS),
    FitnessTest(602, "Pull Up Test (1-Minute)", _ME, Unit.REPS),
    # Flexibility (701-799)
    FitnessTest(701, "Sit-and-Reach Test", _FL, Unit.CENTIMETRES),
    # Reaction (801-899)
    FitnessTest(801, "Reaction Time Ruler Drop Test", _RE, Unit.CENTIMETRES),
)

# ---- The table (SQL) ---------------------------------------------------------------
# Categories and units are small LOOKUP TABLES filled from the enums above every start-up
# (same idea as sports/grades in enrolments.py): adding a unit later is an INSERT, not a
# table rebuild.
#
# WITHOUT ROWID makes test_id an ordinary primary key. Without it, SQLite treats an
# INTEGER PRIMARY KEY as the hidden row number and quietly invents a value when NULL is
# inserted, which would slip past the "ID must match its category" trigger.
#
# Triggers are DROPPED and re-created every start-up so they always match the settings above.
_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS test_categories (
    name     TEXT PRIMARY KEY,
    id_start INTEGER NOT NULL UNIQUE
             CHECK (id_start BETWEEN 0 AND {MAX_TEST_ID} AND id_start % {IDS_PER_CATEGORY} = 0)
) STRICT;

CREATE TABLE IF NOT EXISTS test_units (name TEXT PRIMARY KEY) STRICT;

CREATE TABLE IF NOT EXISTS fitness_tests (
    test_id    INTEGER PRIMARY KEY
               CHECK (test_id BETWEEN 1 AND {MAX_TEST_ID} AND test_id % {IDS_PER_CATEGORY} <> 0),
    test_name  TEXT NOT NULL
               CHECK (test_name = trim(test_name, char(32, 9, 10, 13))
                      AND length(test_name) BETWEEN 1 AND {NAME_MAX_LENGTH}
                      AND test_name GLOB '[A-Za-z0-9]*'
                      AND test_name GLOB '*[A-Za-z]*'
                      AND test_name NOT GLOB '*[^A-Za-z0-9 .''()/:-]*'
                      AND test_name NOT LIKE '%  %'),
    category   TEXT NOT NULL REFERENCES test_categories(name),
    unit       TEXT NOT NULL REFERENCES test_units(name)
) STRICT, WITHOUT ROWID;

-- Backstop for exact (capital-insensitive) duplicate names.
CREATE UNIQUE INDEX IF NOT EXISTS ux_fitness_tests_name ON fitness_tests (lower(test_name));
CREATE INDEX IF NOT EXISTS ix_fitness_tests_category ON fitness_tests (category);

DROP TRIGGER IF EXISTS fitness_tests_id_in_category_block;
CREATE TRIGGER fitness_tests_id_in_category_block BEFORE INSERT ON fitness_tests
WHEN EXISTS (SELECT 1 FROM test_categories
             WHERE name = NEW.category AND id_start <> NEW.test_id / {IDS_PER_CATEGORY} * {IDS_PER_CATEGORY})
-- (an unknown category is left to the foreign key, which reports the real problem)
BEGIN SELECT RAISE(ABORT, 'test_id does not belong to the block of this category'); END;

DROP TRIGGER IF EXISTS fitness_tests_lock_identity;
CREATE TRIGGER fitness_tests_lock_identity
BEFORE UPDATE OF test_id, category, unit ON fitness_tests
BEGIN SELECT RAISE(ABORT, 'test_id, category and unit are locked'); END;

DROP TRIGGER IF EXISTS fitness_tests_no_delete;
CREATE TRIGGER fitness_tests_no_delete BEFORE DELETE ON fitness_tests
BEGIN SELECT RAISE(ABORT, 'fitness tests cannot be deleted'); END;
"""

_SELECT = "SELECT test_id, test_name, category, unit FROM fitness_tests"
_INSERT = "INSERT INTO fitness_tests (test_id, test_name, category, unit) VALUES (?, ?, ?, ?)"

_DATAFRAME_RAW_SCHEMA = {
    "test_id": pl.Int64, "test_name": pl.String, "category": pl.String, "unit": pl.String,
}


# ---- The repository ----------------------------------------------------------------
class FitnessTestRepository(Repository):
    """Saves and loads fitness tests. The rest of the project never writes SQL for this table."""

    def __init__(self, db: Database) -> None:
        super().__init__(db)  # creates the tables and triggers
        self._fill_lookup_tables()

    @property
    def schema_sql(self) -> str:
        return _SCHEMA_SQL

    def add(self, test: FitnessTest) -> None:
        """Save one new test. Raises DuplicateError if its ID or name already exists."""
        self.add_many([test])

    def add_many(self, tests: Iterable[FitnessTest]) -> None:
        """Save several tests all-or-nothing: if one is refused, none are saved."""
        batch = list(tests)
        for item in batch:
            if not isinstance(item, FitnessTest):
                raise ValidationError("only FitnessTest objects can be added")
        current: FitnessTest | None = None
        try:
            with self._db.transaction() as conn:
                for current in batch:
                    conn.execute(_INSERT, (current.test_id, current.test_name,
                                           current.category.value, current.unit.value))
        except sqlite3.IntegrityError as exc:
            raise self._translate(exc, current) from None

    def load_catalogue(self) -> None:
        """Make sure every test in DEFAULT_TESTS is saved (safe to repeat; keeps existing rows).

        A row that already has a catalogue test_id but a different category or unit is a
        real conflict (results would be mislabelled), so it stops with an error instead of
        being overwritten. A renamed test is fine: renames are allowed.
        """
        saved = {t.test_id: t for t in self.all()}
        missing: list[FitnessTest] = []
        for default in DEFAULT_TESTS:
            have = saved.get(default.test_id)
            if have is None:
                missing.append(default)
            elif (have.category, have.unit) != (default.category, default.unit):
                raise ValidationError(
                    f"test_id {default.test_id} is already used by '{have.test_name}' "
                    f"({have.category.value}, {have.unit.value}); the catalogue expects "
                    f"'{default.test_name}' ({default.category.value}, {default.unit.value})"
                )
        self.add_many(missing)

    def next_id(self, category: str | Category) -> int:
        """The next free test_id in a category's block (for adding a new test)."""
        cat = parse_enum(Category, category, "category")
        first, last = cat.id_start + 1, cat.id_start + IDS_PER_CATEGORY - 1
        (top,) = self._db.query(
            "SELECT MAX(test_id) FROM fitness_tests WHERE test_id BETWEEN ? AND ?", (first, last)
        )[0]
        following = first if top is None else top + 1
        if following > last:
            raise ValidationError(f"{cat.value} has no free test_id left ({first}-{last} are used)")
        return following

    def rename(self, test_id: int, new_name: str) -> None:
        """Correct a test's name. The only change allowed on an existing test."""
        _check_test_id(test_id)
        new = replace(self.get(test_id), test_name=_tidy_name(new_name))  # re-runs every name rule
        try:
            with self._db.transaction() as conn:
                conn.execute("UPDATE fitness_tests SET test_name = ? WHERE test_id = ?",
                             (new.test_name, test_id))
        except sqlite3.IntegrityError as exc:
            raise self._translate(exc, new) from None

    def get(self, test_id: int) -> FitnessTest:
        _check_test_id(test_id)
        rows = self._db.query(_SELECT + " WHERE test_id = ?", (test_id,))
        if not rows:
            raise NotFoundError(f"test_id {test_id} does not exist")
        return FitnessTest.from_row(rows[0])

    def get_by_name(self, test_name: str) -> FitnessTest:
        """Find a test by name, ignoring capitals and extra spaces."""
        name = _tidy_name(test_name)
        rows = self._db.query(_SELECT + " WHERE lower(test_name) = lower(?)", (name,))
        if not rows:
            raise NotFoundError(f"no fitness test is named '{name}'")
        return FitnessTest.from_row(rows[0])

    def all(self, *, category: str | Category | None = None) -> list[FitnessTest]:
        return [FitnessTest.from_row(row) for row in self._rows(category)]

    def to_dataframe(self, *, category: str | Category | None = None) -> pl.DataFrame:
        """The table as a Polars dataframe with proper types (Int64, Enum)."""
        df = pl.DataFrame(self._rows(category), schema=_DATAFRAME_RAW_SCHEMA, orient="row")
        return df.with_columns(
            pl.col("category").cast(pl.Enum([c.value for c in Category])),
            pl.col("unit").cast(pl.Enum([u.value for u in Unit])),
        )

    # -- private helpers --
    def _rows(self, category: str | Category | None) -> list[tuple]:
        """The ONE place that builds the SELECT, so every read orders and filters the same way."""
        if category is None:
            return self._db.query(_SELECT + " ORDER BY test_id")
        cat = parse_enum(Category, category, "category")
        return self._db.query(_SELECT + " WHERE category = ? ORDER BY test_id", (cat.value,))

    def _fill_lookup_tables(self) -> None:
        """Make sure every Category and Unit in the enums exists in the database (safe to repeat)."""
        with self._db.transaction() as conn:
            conn.executemany("INSERT OR IGNORE INTO test_categories (name, id_start) VALUES (?, ?)",
                             [(c.value, c.id_start) for c in Category])
            conn.executemany("INSERT OR IGNORE INTO test_units (name) VALUES (?)",
                             [(u.value,) for u in Unit])
        # INSERT OR IGNORE would hide a category whose block was changed in the code but not
        # in an older whc.db; that would number tests in the wrong block. Stop loudly instead.
        stored = dict(self._db.query("SELECT name, id_start FROM test_categories"))
        for cat in Category:
            if stored[cat.value] != cat.id_start:
                raise RuntimeError(
                    f"data/whc.db stores {cat.value} IDs starting at {stored[cat.value]}, but the "
                    f"code says {cat.id_start}. Renumbering tests would break results; "
                    "restore the old block in _ID_START (or rebuild the database)."
                )

    @staticmethod
    def _translate(exc: sqlite3.IntegrityError, test: FitnessTest | None) -> Exception:
        """Turn a raw database error into a clear project error (uses SQLite's error code)."""
        code = exc.sqlite_errorname
        if code == "SQLITE_CONSTRAINT_PRIMARYKEY" and test is not None:
            return DuplicateError(f"test_id {test.test_id} already exists")
        if code == "SQLITE_CONSTRAINT_UNIQUE" and test is not None:
            return DuplicateError(f"a test named '{test.test_name}' already exists")
        if code == "SQLITE_CONSTRAINT_TRIGGER":
            return ValidationError(str(exc))  # the rule's own message
        return ValidationError(f"database refused this fitness test: {exc}")
