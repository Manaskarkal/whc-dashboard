"""fitness_tests.py - the catalogue of fitness tests (what can be measured, in what unit, and
what a believable result looks like).

    Category:                 the allowed buckets (Strength, Power, Speed, ...)
    Unit:                     the allowed units (Seconds, Newtons, Level:Shuttle, ...)
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

min_value / max_value are a PLAUSIBILITY range, wide enough for school-age athletes up to
elite adults. Their job is to catch typos (a 20 m sprint of 35 seconds, a squat of 1000 kg),
not to judge performance. Both ends are inclusive. They can be tuned later with set_range().

Level:Shuttle tests (Yo-Yo, beep test) are different: one result is TWO whole numbers, the
level first and then the shuttle within that level ("16:3" = level 16, shuttle 3). A single
min/max cannot describe a pair, so these tests have NO min/max here. The limits for each part
belong with the stored results (the future test_results table, as two integer columns).

Business rules enforced (each one has tests):
    1. test_id sits inside its category's block (a Power test cannot be numbered 14).
    2. test_name is unique, ignoring capitals, and made of plain characters only.
    3. category and unit must come from the allowed lists.
    4. A test measured in a normal unit has BOTH min_value and max_value, with min below max.
       A Level:Shuttle test has NEITHER.
    5. test_id, category and unit can never change (results depend on them).
       The name can be corrected with rename() and the range tuned with set_range().
    6. Tests cannot be deleted.

Where each rule lives: Python gives friendly messages for typed input; SQL repeats the rules
so bad data cannot get in even if Python is bypassed (same approach as students.py).
"""

from __future__ import annotations

import math
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
VALUE_LIMIT = 1_000_000  # min_value / max_value must sit between -VALUE_LIMIT and +VALUE_LIMIT
# Letters, digits, spaces and . ' ( ) / : -   (enough for "5-0-5 Agility Test Left" and
# "1.6k/Mile Run"). Must start with a letter or digit. Always used with fullmatch().
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .'()/:-]*")

# NOTE on changing settings: like students.py, the SQL below is only written when the table
# is FIRST created. Changing NAME_MAX_LENGTH, MAX_TEST_ID or VALUE_LIMIT later does not alter
# an existing whc.db table. Rebuild the table (see FitnessTestRepository.drop_tables).


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
    """How a result is measured. 'Level:Shuttle' is a pair: level FIRST, then shuttle."""

    CENTIMETRES = "Centimetres"
    METRES = "Metres"
    SECONDS = "Seconds"
    NEWTONS = "Newtons"
    KILOGRAMS = "Kilograms"
    WATTS_PER_KILOGRAM = "Watts/Kilogram"
    REPS = "Reps"
    UNITLESS = "Unitless"
    LEVEL_SHUTTLE = "Level:Shuttle"


# ---- Small pure helpers ------------------------------------------------------------
def _tidy_name(text: object) -> str:
    """Trim, collapse repeated whitespace (incl. newlines) and use one form for accents."""
    return unicodedata.normalize("NFC", " ".join(require_text(text, "test_name").split()))


def _check_test_id(test_id: object) -> None:
    """A real int (not True, not '101'), so lookups never see odd types."""
    if isinstance(test_id, bool) or not isinstance(test_id, int):
        raise ValidationError("test_id must be a whole number")


def _is_plain_number(value: object) -> bool:
    """A real, finite int or float. Not True/False (they are ints in Python), text, NaN or inf."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_range(unit: Unit, min_value: float | None, max_value: float | None) -> None:
    """The min/max rule: both or neither, depending on the unit."""
    if unit is Unit.LEVEL_SHUTTLE:
        if min_value is not None or max_value is not None:
            raise ValidationError(
                f"{unit.value} tests have no min_value/max_value: a result is a level and a "
                "shuttle, not one number"
            )
        return
    if min_value is None or max_value is None:
        raise ValidationError(
            f"min_value and max_value are both required for tests measured in {unit.value}"
        )
    if not _is_plain_number(min_value) or not _is_plain_number(max_value):
        raise ValidationError("min_value and max_value must be numbers (not text or NaN)")
    if abs(min_value) > VALUE_LIMIT or abs(max_value) > VALUE_LIMIT:
        raise ValidationError(
            f"min_value and max_value must be between -{VALUE_LIMIT:,} and {VALUE_LIMIT:,}"
        )
    if min_value >= max_value:
        raise ValidationError("min_value must be below max_value")


# ---- The FitnessTest (value object) ------------------------------------------------
@dataclass(frozen=True, slots=True)
class FitnessTest:
    """One fitness test. Frozen: once created it cannot be changed. Invalid tests cannot exist."""

    test_id: int
    test_name: str
    category: Category
    unit: Unit
    min_value: float | None = None
    max_value: float | None = None

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
        _check_range(self.unit, self.min_value, self.max_value)

    @classmethod
    def new(
        cls,
        test_id: int,
        test_name: str,
        category: str,
        unit: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> FitnessTest:
        """Build a test from raw typed input (cleans the name, matches category and unit)."""
        return cls(
            test_id=test_id,
            test_name=_tidy_name(test_name),
            category=parse_enum(Category, category, "category"),
            unit=parse_enum(Unit, unit, "unit"),
            min_value=min_value,
            max_value=max_value,
        )

    @classmethod
    def from_row(cls, row: tuple) -> FitnessTest:
        """Build a test from a database row.

        Unpacks all 6 columns on purpose, so a changed _SELECT fails loudly.
        """
        test_id, test_name, category, unit, min_value, max_value = row
        return cls(test_id, test_name, Category(category), Unit(unit), min_value, max_value)

    def check_value(self, value: float) -> None:
        """Raise ValidationError unless `value` is a plausible result (inclusive range).

        For the future test_results table. A Level:Shuttle test has no single-number range,
        so it is refused here (check its level and shuttle separately).
        """
        if self.min_value is None or self.max_value is None:
            raise ValidationError(
                f"'{self.test_name}' is measured as {self.unit.value}, a pair of numbers, "
                "so it has no single min/max to check against"
            )
        if not _is_plain_number(value):
            raise ValidationError(f"{self.test_name}: result must be a number (not text or NaN)")
        if not self.min_value <= value <= self.max_value:
            raise ValidationError(
                f"{self.test_name}: {value:g} is outside the allowed range "
                f"{self.min_value:g} to {self.max_value:g} {self.unit.value}"
            )


# ---- The standard catalogue --------------------------------------------------------
# IDs are written out by hand and must never change once results exist. To add a test, give
# it the next free number in its block (or use FitnessTestRepository.next_id). Building these
# objects at import time means a typo in this list stops the program immediately.
# Arguments: id, name, category, unit, min_value, max_value (no range for Level:Shuttle).
_S, _P, _SP, _AG = Category.STRENGTH, Category.POWER, Category.SPEED, Category.AGILITY
_AE, _AN, _ME = Category.AEROBIC_ENDURANCE, Category.ANAEROBIC_ENDURANCE, Category.MUSCULAR_ENDURANCE
_FL, _RE = Category.FLEXIBILITY, Category.REACTION

DEFAULT_TESTS: tuple[FitnessTest, ...] = (
    # Strength (1-99)
    FitnessTest(1, "Isometric Mid-Thigh Pull Peak", _S, Unit.NEWTONS, 300, 7000),
    FitnessTest(2, "Isometric Mid-Thigh Pull Relative", _S, Unit.UNITLESS, 1, 7),  # x bodyweight
    FitnessTest(3, "Nordic Hamstring Left", _S, Unit.NEWTONS, 50, 800),
    FitnessTest(4, "Nordic Hamstring Right", _S, Unit.NEWTONS, 50, 800),
    FitnessTest(5, "Adduction Left", _S, Unit.NEWTONS, 50, 800),
    FitnessTest(6, "Adduction Right", _S, Unit.NEWTONS, 50, 800),
    FitnessTest(7, "Abduction Left", _S, Unit.NEWTONS, 50, 800),
    FitnessTest(8, "Abduction Right", _S, Unit.NEWTONS, 50, 800),
    FitnessTest(9, "1RM Back Squat", _S, Unit.KILOGRAMS, 20, 350),
    FitnessTest(10, "1RM Bench Press", _S, Unit.KILOGRAMS, 20, 250),
    FitnessTest(11, "1RM Trap Bar Deadlift", _S, Unit.KILOGRAMS, 30, 400),
    FitnessTest(12, "3RM Back Squat", _S, Unit.KILOGRAMS, 20, 320),
    FitnessTest(13, "3RM Bench Press", _S, Unit.KILOGRAMS, 20, 230),
    FitnessTest(14, "3RM Trap Bar Deadlift", _S, Unit.KILOGRAMS, 30, 380),
    # Power (101-199)
    FitnessTest(101, "Counter Movement Jump Height", _P, Unit.CENTIMETRES, 5, 90),
    FitnessTest(102, "Counter Movement Jump Power", _P, Unit.WATTS_PER_KILOGRAM, 15, 80),
    # RSI here = jump height / contact time (typical results about 0.3-0.5).
    FitnessTest(103, "Counter Movement Jump Relative Strength Index (RSI)", _P, Unit.UNITLESS, 0.1, 1.2),
    FitnessTest(104, "Hop Jump Relative Strength Index (RSI)", _P, Unit.UNITLESS, 0.3, 5),
    FitnessTest(105, "Standing Long Jump (Broad Jump)", _P, Unit.METRES, 0.8, 3.6),
    # Speed (201-299)
    FitnessTest(201, "5 Metre Sprint", _SP, Unit.SECONDS, 0.8, 3.5),
    FitnessTest(202, "10 Metre Sprint", _SP, Unit.SECONDS, 1.2, 6.0),
    FitnessTest(203, "20 Metre Sprint", _SP, Unit.SECONDS, 2.2, 8.0),
    # Agility (301-399)
    FitnessTest(301, "5-0-5 Agility Test Left", _AG, Unit.SECONDS, 2.0, 8.0),
    FitnessTest(302, "5-0-5 Agility Test Right", _AG, Unit.SECONDS, 2.0, 8.0),
    FitnessTest(303, "NBA Lane Agility Test", _AG, Unit.SECONDS, 8, 20),
    # Aerobic Endurance (401-499)
    FitnessTest(401, "YOYO Intermittent Recovery Test 1 (IRT1)", _AE, Unit.LEVEL_SHUTTLE),
    FitnessTest(402, "20m Beep Test (Shuttle Run)", _AE, Unit.LEVEL_SHUTTLE),
    FitnessTest(403, "Cooper 12-Minute Run", _AE, Unit.METRES, 1000, 4500),
    FitnessTest(404, "1.6k/Mile Run", _AE, Unit.SECONDS, 220, 1200),
    FitnessTest(405, "2k Run", _AE, Unit.SECONDS, 300, 1500),
    # Anaerobic Endurance (501-599)
    FitnessTest(501, "YOYO Intermittent Recovery Test 2 (IRT2)", _AN, Unit.LEVEL_SHUTTLE),
    # Muscular Endurance (601-699)
    FitnessTest(601, "Push Up Test (1-Minute)", _ME, Unit.REPS, 0, 120),
    FitnessTest(602, "Pull Up Test (1-Minute)", _ME, Unit.REPS, 0, 60),
    # Flexibility (701-799)
    FitnessTest(701, "Sit-and-Reach Test", _FL, Unit.CENTIMETRES, -30, 50),
    # Reaction (801-899)
    FitnessTest(801, "Reaction Time Ruler Drop Test", _RE, Unit.CENTIMETRES, 1, 40),
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
# min_value/max_value are REAL and nullable; the named CHECK says which units need them
# (see _check_range). SQLite stores NaN as NULL, so NaN is refused the same way as a gap.
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
    unit       TEXT NOT NULL REFERENCES test_units(name),
    min_value  REAL,
    max_value  REAL,
    CONSTRAINT range_matches_unit CHECK (
        (unit = '{Unit.LEVEL_SHUTTLE.value}' AND min_value IS NULL AND max_value IS NULL)
        OR (unit <> '{Unit.LEVEL_SHUTTLE.value}'
            AND min_value IS NOT NULL AND max_value IS NOT NULL
            AND min_value < max_value
            AND min_value >= -{VALUE_LIMIT} AND max_value <= {VALUE_LIMIT})
    )
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

_COLUMNS = ("test_id", "test_name", "category", "unit", "min_value", "max_value")
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM fitness_tests"
_INSERT = f"INSERT INTO fitness_tests ({', '.join(_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?)"

_DATAFRAME_RAW_SCHEMA = {
    "test_id": pl.Int64, "test_name": pl.String, "category": pl.String, "unit": pl.String,
    "min_value": pl.Float64, "max_value": pl.Float64,
}


# ---- The repository ----------------------------------------------------------------
class FitnessTestRepository(Repository):
    """Saves and loads fitness tests. The rest of the project never writes SQL for this table."""

    def __init__(self, db: Database) -> None:
        super().__init__(db)  # creates the tables and triggers
        self._check_table_is_current()
        self._fill_lookup_tables()

    @property
    def schema_sql(self) -> str:
        return _SCHEMA_SQL

    @staticmethod
    def drop_tables(db: Database) -> None:
        """Delete the three fitness tables so they are rebuilt (empty) next start-up.

        Use ONCE after the table layout changes. This removes every row, including any test
        you added yourself, but never touches students or enrolments. Safe only while no
        other table (e.g. a future test_results) points at fitness_tests.
        """
        with db.transaction() as conn:
            for table in ("fitness_tests", "test_categories", "test_units"):  # child first
                conn.execute(f"DROP TABLE IF EXISTS {table}")

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
                                           current.category.value, current.unit.value,
                                           current.min_value, current.max_value))
        except sqlite3.IntegrityError as exc:
            raise self._translate(exc, current) from None

    def load_catalogue(self) -> None:
        """Make sure every test in DEFAULT_TESTS is saved (safe to repeat; keeps existing rows).

        A row that already has a catalogue test_id but a different category or unit is a
        real conflict (results would be mislabelled), so it stops with an error instead of
        being overwritten. A renamed test or a tuned min/max is NOT a conflict: those are
        allowed to change, and reloading never puts the catalogue's values back.
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
        """Correct a test's name."""
        _check_test_id(test_id)
        new = replace(self.get(test_id), test_name=_tidy_name(new_name))  # re-runs every name rule
        try:
            with self._db.transaction() as conn:
                conn.execute("UPDATE fitness_tests SET test_name = ? WHERE test_id = ?",
                             (new.test_name, test_id))
        except sqlite3.IntegrityError as exc:
            raise self._translate(exc, new) from None

    def set_range(self, test_id: int, min_value: float, max_value: float) -> None:
        """Tune a test's plausible range. Refused for Level:Shuttle tests (they have none)."""
        _check_test_id(test_id)
        new = replace(self.get(test_id), min_value=min_value, max_value=max_value)  # re-runs rules
        try:
            with self._db.transaction() as conn:
                conn.execute("UPDATE fitness_tests SET min_value = ?, max_value = ? "
                             "WHERE test_id = ?", (new.min_value, new.max_value, test_id))
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
        """The table as a Polars dataframe with proper types (Int64, Enum, Float64)."""
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

    def _check_table_is_current(self) -> None:
        """Stop with a clear message if whc.db still has an older fitness_tests table.

        CREATE TABLE IF NOT EXISTS cannot add columns or change rules on an existing table,
        so an old table would otherwise fail later with a confusing 'no such column' error.
        """
        columns = tuple(row[1] for row in self._db.query("PRAGMA table_info(fitness_tests)"))
        if columns != _COLUMNS:
            raise RuntimeError(
                "data/whc.db has an old fitness_tests table (it has no min_value/max_value). "
                "Rebuild just the fitness tables (students and enrolments are not touched) with:\n"
                '  uv run python -c "from whc.database import Database; '
                'from whc.fitness_tests import FitnessTestRepository as R; '
                'db = Database(); R.drop_tables(db); db.close()"\n'
                "then run main.py again."
            )

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