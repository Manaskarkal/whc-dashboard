"""fitness_tests.py - the catalogue of fitness tests (what can be measured, in what unit,
which direction is better, and what a believable result looks like).

    Category:                 default seeded buckets (extra categories via add_category())
    Unit:                     default seeded units (extra units via add_unit())
    ScoreDirection:           'higher', 'lower', or 'neutral' (so new tests work in analytics automatically)
    FitnessTest:              one validated test (cannot be edited once created)
    DEFAULT_TESTS:            the standard 37-test catalogue, loaded by load_catalogue()
    DEFAULT_LEVEL_SHUTTLE_RULES: official shuttle limits for Yo-Yo IRT1, Yo-Yo IRT2, 20m Beep Test
    FitnessTestRepository:    saves and loads tests (the only code that touches the table)

test_id numbering (each category owns a block of 100 numbers; MAX_TEST_ID = 9999 -> 100 blocks):
      1-99   Strength            400s  Aerobic Endurance
    101-199  Power               500s  Anaerobic Endurance
    201-299  Speed               600s  Muscular Endurance
    301-399  Agility             700s  Flexibility
    801-899  Reaction            900s  Anthropometrics
   1001+     Future categories added via add_category() (zero code edits required)
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TypeGuard

import polars as pl

from whc.database import Database, Repository
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.students import parse_enum, require_text

# ---- Settings (change rules here) --------------------------------------------------
IDS_PER_CATEGORY = 100  # each category owns a block of 100 numbers
MAX_TEST_ID = 9999  # 100 blocks: 0-99 ... 9900-9999
NAME_MAX_LENGTH = 80
LOOKUP_MAX_LENGTH = 50
VALUE_LIMIT = 1_000_000  # min_value / max_value must sit between -VALUE_LIMIT and +VALUE_LIMIT
# Letters, digits, spaces and . ' ( ) / : -   (enough for "5-0-5 Agility Test Left" and
# "1.6k/Mile Run"). Must start with a letter or digit. Always used with fullmatch().
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .'()/:-]*")
LOOKUP_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9 /:'-]*")


class ScoreDirection(StrEnum):
    """Whether higher or lower scores represent better performance."""

    HIGHER = "higher"
    LOWER = "lower"
    NEUTRAL = "neutral"


class Category(StrEnum):
    """Default seeded categories. Extra categories can be added at runtime via add_category()."""

    STRENGTH = "Strength"
    POWER = "Power"
    SPEED = "Speed"
    AGILITY = "Agility"
    AEROBIC_ENDURANCE = "Aerobic Endurance"
    ANAEROBIC_ENDURANCE = "Anaerobic Endurance"
    MUSCULAR_ENDURANCE = "Muscular Endurance"
    FLEXIBILITY = "Flexibility"
    REACTION = "Reaction"
    ANTHROPOMETRICS = "Anthropometrics"

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
    Category.ANTHROPOMETRICS: 900,
}

_DEFAULT_CATEGORY_START: dict[str, int] = {c.value: c.id_start for c in Category}


class Unit(StrEnum):
    """Default seeded units. Extra units can be added at runtime via add_unit()."""

    CENTIMETRES = "Centimetres"
    METRES = "Metres"
    SECONDS = "Seconds"
    NEWTONS = "Newtons"
    KILOGRAMS = "Kilograms"
    WATTS_PER_KILOGRAM = "Watts/Kilogram"
    REPS = "Reps"
    UNITLESS = "Unitless"
    LEVEL_SHUTTLE = "Level:Shuttle"


# Official Level:Shuttle protocols (test_id -> {valid_level: max_shuttle})
DEFAULT_LEVEL_SHUTTLE_RULES: dict[int, dict[int, int]] = {
    # 401: Yo-Yo Intermittent Recovery Test 1 (IRT1)
    401: {
        5: 1,
        9: 1,
        11: 2,
        12: 3,
        13: 4,
        **{lvl: 8 for lvl in range(14, 24)},
    },
    # 402: 20m Beep Test (Leger 1988 protocol)
    402: {
        1: 7, 2: 8, 3: 8, 4: 9, 5: 9, 6: 10, 7: 10,
        8: 11, 9: 11, 10: 11, 11: 12, 12: 12, 13: 13, 14: 13,
        15: 13, 16: 14, 17: 14, 18: 15, 19: 15, 20: 16, 21: 16,
    },
    # 501: Yo-Yo Intermittent Recovery Test 2 (IRT2)
    501: {
        11: 1,
        15: 1,
        17: 2,
        18: 3,
        19: 4,
        **{lvl: 8 for lvl in range(20, 30)},
    },
}


# ---- Small pure helpers ------------------------------------------------------------
def _tidy_name(text: object) -> str:
    """Trim, collapse repeated whitespace (incl. newlines) and use one form for accents."""
    return unicodedata.normalize("NFC", " ".join(require_text(text, "test_name").split()))


def _tidy_lookup(text: object, label: str, defaults: type[StrEnum]) -> str:
    """Clean a lookup name (category or unit); matches default enum values case-insensitively."""
    raw = unicodedata.normalize("NFC", " ".join(require_text(text, label).split()))
    for member in defaults:
        if member.value.casefold() == raw.casefold():
            return member.value
    if not raw or len(raw) > LOOKUP_MAX_LENGTH or not LOOKUP_PATTERN.fullmatch(raw):
        raise ValidationError(
            f"{label} must be 1-{LOOKUP_MAX_LENGTH} characters: letters, digits, single spaces "
            "and / : ' - only, starting with a letter"
        )
    return raw


def _check_test_id(test_id: object) -> None:
    """A real int (not True, not '101'), so lookups never see odd types."""
    if isinstance(test_id, bool) or not isinstance(test_id, int):
        raise ValidationError("test_id must be a whole number")


def _is_plain_number(value: object) -> TypeGuard[int | float]:
    """A real, finite int or float. Not True/False (they are ints in Python), text, NaN or inf."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_range(unit: str, min_value: float | None, max_value: float | None) -> None:
    """The min/max rule: both or neither, depending on the unit."""
    if unit == Unit.LEVEL_SHUTTLE.value:
        if min_value is not None or max_value is not None:
            raise ValidationError(
                f"{unit} tests have no min_value/max_value: a result is a level and a "
                "shuttle, not one number"
            )
        return
    if min_value is None or max_value is None:
        raise ValidationError(
            f"min_value and max_value are both required for tests measured in {unit}"
        )
    if not _is_plain_number(min_value) or not _is_plain_number(max_value):
        raise ValidationError("min_value and max_value must be numbers (not text or NaN)")
    if abs(min_value) > VALUE_LIMIT or abs(max_value) > VALUE_LIMIT:
        raise ValidationError(
            f"min_value and max_value must be between -{VALUE_LIMIT:,} and {VALUE_LIMIT:,}"
        )
    if min_value >= max_value:
        raise ValidationError("min_value must be below max_value")


def _sql_list(values: list[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


# ---- The FitnessTest (value object) ------------------------------------------------
@dataclass(frozen=True, slots=True)
class FitnessTest:
    """One fitness test. Frozen: once created it cannot be changed. Invalid tests cannot exist."""

    test_id: int
    test_name: str
    category: str
    unit: str
    min_value: float | None = None
    max_value: float | None = None
    score_direction: ScoreDirection = ScoreDirection.HIGHER

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
        if (
            not isinstance(self.category, str)
            or len(self.category) > LOOKUP_MAX_LENGTH
            or not LOOKUP_PATTERN.fullmatch(self.category)
            or self.category != self.category.strip()
            or "  " in self.category
        ):
            raise ValidationError("category must be a valid category name")
        if (
            not isinstance(self.unit, str)
            or len(self.unit) > LOOKUP_MAX_LENGTH
            or not LOOKUP_PATTERN.fullmatch(self.unit)
            or self.unit != self.unit.strip()
            or "  " in self.unit
        ):
            raise ValidationError("unit must be a valid unit name")
        if not isinstance(self.score_direction, ScoreDirection):
            raise ValidationError("score_direction must be a ScoreDirection value")
        # For default categories, check the block immediately in Python (SQLite checks all categories)
        if self.category in _DEFAULT_CATEGORY_START:
            start = _DEFAULT_CATEGORY_START[self.category]
            if self.test_id // IDS_PER_CATEGORY * IDS_PER_CATEGORY != start:
                raise ValidationError(
                    f"test_id {self.test_id} does not belong to {self.category}: "
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
        *,
        score_direction: str | ScoreDirection = ScoreDirection.HIGHER,
    ) -> FitnessTest:
        """Build a test from raw typed input (cleans the name, matches category/unit/direction)."""
        direction = (
            score_direction
            if isinstance(score_direction, ScoreDirection)
            else parse_enum(ScoreDirection, score_direction, "score_direction")
        )
        return cls(
            test_id=test_id,
            test_name=_tidy_name(test_name),
            category=_tidy_lookup(category, "category", Category),
            unit=_tidy_lookup(unit, "unit", Unit),
            min_value=min_value,
            max_value=max_value,
            score_direction=direction,
        )

    @classmethod
    def from_row(cls, row: tuple) -> FitnessTest:
        """Build a test from a database row (unpacks all 7 columns on purpose)."""
        test_id, test_name, category, unit, min_value, max_value, score_direction = row
        return cls(
            test_id,
            test_name,
            category,
            unit,
            min_value,
            max_value,
            ScoreDirection(score_direction),
        )

    def check_value(self, value: float) -> None:
        """Raise ValidationError unless `value` is a plausible result (inclusive range)."""
        if self.min_value is None or self.max_value is None:
            raise ValidationError(
                f"'{self.test_name}' is measured as {self.unit}, a pair of numbers, "
                "so it has no single min/max to check against"
            )
        if not _is_plain_number(value):
            raise ValidationError(f"{self.test_name}: result must be a number (not text or NaN)")
        if not self.min_value <= value <= self.max_value:
            raise ValidationError(
                f"{self.test_name}: {value:g} is outside the allowed range "
                f"{self.min_value:g} to {self.max_value:g} {self.unit}"
            )


# ---- The standard catalogue --------------------------------------------------------
_S, _P, _SP, _AG = Category.STRENGTH, Category.POWER, Category.SPEED, Category.AGILITY
_AE, _AN, _ME = Category.AEROBIC_ENDURANCE, Category.ANAEROBIC_ENDURANCE, Category.MUSCULAR_ENDURANCE
_FL, _RE, _ANT = Category.FLEXIBILITY, Category.REACTION, Category.ANTHROPOMETRICS
_HI, _LO, _NEU = ScoreDirection.HIGHER, ScoreDirection.LOWER, ScoreDirection.NEUTRAL

DEFAULT_TESTS: tuple[FitnessTest, ...] = (
    # Strength (1-99)
    FitnessTest(1, "Isometric Mid-Thigh Pull Peak", _S, Unit.NEWTONS, 300, 7000, _HI),
    FitnessTest(2, "Isometric Mid-Thigh Pull Relative", _S, Unit.UNITLESS, 1, 7, _HI),
    FitnessTest(3, "Nordic Hamstring Left", _S, Unit.NEWTONS, 50, 800, _HI),
    FitnessTest(4, "Nordic Hamstring Right", _S, Unit.NEWTONS, 50, 800, _HI),
    FitnessTest(5, "Adduction Left", _S, Unit.NEWTONS, 50, 800, _HI),
    FitnessTest(6, "Adduction Right", _S, Unit.NEWTONS, 50, 800, _HI),
    FitnessTest(7, "Abduction Left", _S, Unit.NEWTONS, 50, 800, _HI),
    FitnessTest(8, "Abduction Right", _S, Unit.NEWTONS, 50, 800, _HI),
    FitnessTest(9, "1RM Back Squat", _S, Unit.KILOGRAMS, 20, 350, _HI),
    FitnessTest(10, "1RM Bench Press", _S, Unit.KILOGRAMS, 20, 250, _HI),
    FitnessTest(11, "1RM Trap Bar Deadlift", _S, Unit.KILOGRAMS, 30, 400, _HI),
    FitnessTest(12, "3RM Back Squat", _S, Unit.KILOGRAMS, 20, 320, _HI),
    FitnessTest(13, "3RM Bench Press", _S, Unit.KILOGRAMS, 20, 230, _HI),
    FitnessTest(14, "3RM Trap Bar Deadlift", _S, Unit.KILOGRAMS, 30, 380, _HI),
    # Power (101-199)
    FitnessTest(101, "Counter Movement Jump Height", _P, Unit.CENTIMETRES, 5, 90, _HI),
    FitnessTest(102, "Counter Movement Jump Power", _P, Unit.WATTS_PER_KILOGRAM, 15, 80, _HI),
    FitnessTest(103, "Counter Movement Jump Relative Strength Index (RSI)", _P, Unit.UNITLESS, 0.1, 1.2, _HI),
    FitnessTest(104, "Hop Jump Relative Strength Index (RSI)", _P, Unit.UNITLESS, 0.3, 5, _HI),
    FitnessTest(105, "Standing Long Jump (Broad Jump)", _P, Unit.METRES, 0.8, 3.6, _HI),
    # Speed (201-299) - lower seconds is faster/better
    FitnessTest(201, "5 Metre Sprint", _SP, Unit.SECONDS, 0.8, 3.5, _LO),
    FitnessTest(202, "10 Metre Sprint", _SP, Unit.SECONDS, 1.2, 6.0, _LO),
    FitnessTest(203, "20 Metre Sprint", _SP, Unit.SECONDS, 2.2, 8.0, _LO),
    # Agility (301-399) - lower seconds is faster/better
    FitnessTest(301, "5-0-5 Agility Test Left", _AG, Unit.SECONDS, 2.0, 8.0, _LO),
    FitnessTest(302, "5-0-5 Agility Test Right", _AG, Unit.SECONDS, 2.0, 8.0, _LO),
    FitnessTest(303, "NBA Lane Agility Test", _AG, Unit.SECONDS, 8, 20, _LO),
    # Aerobic Endurance (401-499)
    FitnessTest(401, "YOYO Intermittent Recovery Test 1 (IRT1)", _AE, Unit.LEVEL_SHUTTLE, None, None, _HI),
    FitnessTest(402, "20m Beep Test (Shuttle Run)", _AE, Unit.LEVEL_SHUTTLE, None, None, _HI),
    FitnessTest(403, "Cooper 12-Minute Run", _AE, Unit.METRES, 1000, 4500, _HI),
    FitnessTest(404, "1.6k/Mile Run", _AE, Unit.SECONDS, 220, 1200, _LO),
    FitnessTest(405, "2k Run", _AE, Unit.SECONDS, 300, 1500, _LO),
    # Anaerobic Endurance (501-599)
    FitnessTest(501, "YOYO Intermittent Recovery Test 2 (IRT2)", _AN, Unit.LEVEL_SHUTTLE, None, None, _HI),
    # Muscular Endurance (601-699)
    FitnessTest(601, "Push Up Test (1-Minute)", _ME, Unit.REPS, 0, 120, _HI),
    FitnessTest(602, "Pull Up Test (1-Minute)", _ME, Unit.REPS, 0, 60, _HI),
    # Flexibility (701-799)
    FitnessTest(701, "Sit-and-Reach Test", _FL, Unit.CENTIMETRES, -30, 50, _HI),
    # Reaction (801-899) - smaller ruler drop distance = faster reaction
    FitnessTest(801, "Reaction Time Ruler Drop Test", _RE, Unit.CENTIMETRES, 1, 40, _LO),
    # Anthropometrics (901-999) - neither higher nor lower is inherently "better"
    FitnessTest(901, "Height", _ANT, Unit.CENTIMETRES, 100, 230, _NEU),
    FitnessTest(902, "Weight", _ANT, Unit.KILOGRAMS, 20, 200, _NEU),
)

# ---- The table (SQL) ---------------------------------------------------------------
_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS test_categories (
    name     TEXT PRIMARY KEY
             CHECK (name = trim(name, char(32, 9, 10, 13))
                    AND length(name) BETWEEN 1 AND {LOOKUP_MAX_LENGTH}
                    AND name GLOB '[A-Za-z]*'
                    AND name NOT LIKE '%  %'),
    id_start INTEGER NOT NULL UNIQUE
             CHECK (id_start BETWEEN 0 AND {MAX_TEST_ID} AND id_start % {IDS_PER_CATEGORY} = 0)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_test_categories_name ON test_categories (lower(name));

CREATE TABLE IF NOT EXISTS test_units (
    name TEXT PRIMARY KEY
         CHECK (name = trim(name, char(32, 9, 10, 13))
                AND length(name) BETWEEN 1 AND {LOOKUP_MAX_LENGTH}
                AND name GLOB '[A-Za-z]*'
                AND name NOT LIKE '%  %')
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_test_units_name ON test_units (lower(name));

CREATE TABLE IF NOT EXISTS fitness_tests (
    test_id         INTEGER PRIMARY KEY
                    CHECK (test_id BETWEEN 1 AND {MAX_TEST_ID} AND test_id % {IDS_PER_CATEGORY} <> 0),
    test_name       TEXT NOT NULL
                    CHECK (test_name = trim(test_name, char(32, 9, 10, 13))
                           AND length(test_name) BETWEEN 1 AND {NAME_MAX_LENGTH}
                           AND test_name GLOB '[A-Za-z0-9]*'
                           AND test_name GLOB '*[A-Za-z]*'
                           AND test_name NOT GLOB '*[^A-Za-z0-9 .''()/:-]*'
                           AND test_name NOT LIKE '%  %'),
    category        TEXT NOT NULL REFERENCES test_categories(name),
    unit            TEXT NOT NULL REFERENCES test_units(name),
    min_value       REAL,
    max_value       REAL,
    score_direction TEXT NOT NULL DEFAULT 'higher'
                    CHECK (score_direction IN ({_sql_list([d.value for d in ScoreDirection])})),
    CONSTRAINT range_matches_unit CHECK (
        (unit = '{Unit.LEVEL_SHUTTLE.value}' AND min_value IS NULL AND max_value IS NULL)
        OR (unit <> '{Unit.LEVEL_SHUTTLE.value}'
            AND min_value IS NOT NULL AND max_value IS NOT NULL
            AND min_value < max_value
            AND min_value >= -{VALUE_LIMIT} AND max_value <= {VALUE_LIMIT})
    )
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS level_shuttle_rules (
    test_id     INTEGER NOT NULL REFERENCES fitness_tests(test_id),
    level       INTEGER NOT NULL CHECK (level BETWEEN 1 AND 30),
    max_shuttle INTEGER NOT NULL CHECK (max_shuttle BETWEEN 1 AND 20),
    PRIMARY KEY (test_id, level)
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

_COLUMNS = ("test_id", "test_name", "category", "unit", "min_value", "max_value", "score_direction")
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM fitness_tests"
_INSERT = f"INSERT INTO fitness_tests ({', '.join(_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?, ?)"

_DATAFRAME_RAW_SCHEMA = {
    "test_id": pl.Int64, "test_name": pl.String, "category": pl.String, "unit": pl.String,
    "min_value": pl.Float64, "max_value": pl.Float64, "score_direction": pl.String,
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
        """Delete the fitness tables so they are rebuilt (empty) next start-up."""
        with db.transaction() as conn:
            for table in ("level_shuttle_rules", "fitness_tests", "test_categories", "test_units"):
                conn.execute(f"DROP TABLE IF EXISTS {table}")

    def categories(self) -> dict[str, int]:
        """Return `{category_name: id_start}` from the live lookup table, ordered by id_start."""
        return dict(self._db.query("SELECT name, id_start FROM test_categories ORDER BY id_start"))

    def units(self) -> list[str]:
        """Return all units from the live lookup table, sorted."""
        return [row[0] for row in self._db.query("SELECT name FROM test_units ORDER BY name")]

    def add_category(self, name: str, id_start: int | None = None) -> tuple[str, int]:
        """Add a new test category to the database (auto-assigns next 100-block if omitted)."""
        clean = _tidy_lookup(name, "category", Category)
        if id_start is None:
            (top_start,) = self._db.query("SELECT MAX(id_start) FROM test_categories")[0]
            id_start = 0 if top_start is None else int(top_start) + IDS_PER_CATEGORY
        if (
            isinstance(id_start, bool)
            or not isinstance(id_start, int)
            or not 0 <= id_start <= MAX_TEST_ID
            or id_start % IDS_PER_CATEGORY != 0
        ):
            raise ValidationError(
                f"id_start must be a multiple of {IDS_PER_CATEGORY} between 0 and {MAX_TEST_ID}"
            )
        try:
            with self._db.transaction() as conn:
                conn.execute(
                    "INSERT INTO test_categories (name, id_start) VALUES (?, ?)", (clean, id_start)
                )
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorname in ("SQLITE_CONSTRAINT_PRIMARYKEY", "SQLITE_CONSTRAINT_UNIQUE"):
                raise DuplicateError(
                    f"category '{clean}' or block {id_start} already exists"
                ) from None
            raise ValidationError(f"database refused category '{clean}': {exc}") from None
        return clean, id_start

    def add_unit(self, name: str) -> str:
        """Add a new measurement unit to the database (no code changes needed)."""
        clean = _tidy_lookup(name, "unit", Unit)
        try:
            with self._db.transaction() as conn:
                conn.execute("INSERT INTO test_units (name) VALUES (?)", (clean,))
        except sqlite3.IntegrityError as exc:
            if exc.sqlite_errorname in ("SQLITE_CONSTRAINT_PRIMARYKEY", "SQLITE_CONSTRAINT_UNIQUE"):
                raise DuplicateError(f"unit '{clean}' already exists") from None
            raise ValidationError(f"database refused unit '{clean}': {exc}") from None
        return clean

    def set_level_shuttle_rules(self, test_id: int, rules: dict[int, int]) -> None:
        """Register or replace the valid (level -> max_shuttle) protocol table for a Level:Shuttle test."""
        test = self.get(test_id)
        if test.unit != Unit.LEVEL_SHUTTLE.value:
            raise ValidationError(f"test_id {test_id} is not a {Unit.LEVEL_SHUTTLE.value} test")
        if not rules:
            raise ValidationError("level_shuttle_rules must contain at least one level")
        clean_rows: list[tuple[int, int, int]] = []
        for lvl, max_sh in sorted(rules.items()):
            if isinstance(lvl, bool) or not isinstance(lvl, int) or not 1 <= lvl <= 30:
                raise ValidationError("level must be a whole number between 1 and 30")
            if isinstance(max_sh, bool) or not isinstance(max_sh, int) or not 1 <= max_sh <= 20:
                raise ValidationError("max_shuttle must be a whole number between 1 and 20")
            clean_rows.append((test_id, lvl, max_sh))
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM level_shuttle_rules WHERE test_id = ?", (test_id,))
            conn.executemany(
                "INSERT INTO level_shuttle_rules (test_id, level, max_shuttle) VALUES (?, ?, ?)",
                clean_rows,
            )

    def level_shuttle_rules(self, test_id: int) -> dict[int, int]:
        """Return `{level: max_shuttle}` for a test (empty dict if no protocol rules registered)."""
        _check_test_id(test_id)
        return dict(
            self._db.query(
                "SELECT level, max_shuttle FROM level_shuttle_rules WHERE test_id = ? ORDER BY level",
                (test_id,),
            )
        )

    def all_level_shuttle_rules(self) -> dict[int, dict[int, int]]:
        """Return `{test_id: {level: max_shuttle}}` for all tests with registered protocol rules."""
        rules: dict[int, dict[int, int]] = {}
        for tid, lvl, max_sh in self._db.query(
            "SELECT test_id, level, max_shuttle FROM level_shuttle_rules ORDER BY test_id, level"
        ):
            rules.setdefault(int(tid), {})[int(lvl)] = int(max_sh)
        return rules

    def add(self, test: FitnessTest) -> None:
        """Save one new test. Raises DuplicateError if its ID or name already exists."""
        self.add_many([test])

    def add_many(self, tests: Iterable[FitnessTest]) -> None:
        """Save several tests all-or-nothing: if one is refused, none are saved."""
        batch = list(tests)
        resolved: list[FitnessTest] = []
        for item in batch:
            if not isinstance(item, FitnessTest):
                raise ValidationError("only FitnessTest objects can be added")
            cat = self._resolve_category(item.category)
            unit = self._resolve_unit(item.unit)
            resolved.append(replace(item, category=cat, unit=unit))
        current: FitnessTest | None = None
        try:
            with self._db.transaction() as conn:
                for current in resolved:
                    conn.execute(
                        _INSERT,
                        (
                            current.test_id,
                            current.test_name,
                            current.category,
                            current.unit,
                            current.min_value,
                            current.max_value,
                            current.score_direction.value,
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            raise self._translate(exc, current) from None

    def load_catalogue(self) -> None:
        """Make sure every test in DEFAULT_TESTS and its Level:Shuttle rules are saved."""
        saved = {t.test_id: t for t in self.all()}
        missing: list[FitnessTest] = []
        for default in DEFAULT_TESTS:
            have = saved.get(default.test_id)
            if have is None:
                missing.append(default)
            elif (have.category, have.unit) != (default.category, default.unit):
                raise ValidationError(
                    f"test_id {default.test_id} is already used by '{have.test_name}' "
                    f"({have.category}, {have.unit}); the catalogue expects "
                    f"'{default.test_name}' ({default.category}, {default.unit})"
                )
        self.add_many(missing)
        # Seed official Level:Shuttle protocol tables (idempotent)
        with self._db.transaction() as conn:
            for tid, rules in DEFAULT_LEVEL_SHUTTLE_RULES.items():
                conn.executemany(
                    "INSERT OR IGNORE INTO level_shuttle_rules (test_id, level, max_shuttle) "
                    "VALUES (?, ?, ?)",
                    [(tid, lvl, max_sh) for lvl, max_sh in rules.items()],
                )

    def next_id(self, category: str) -> int:
        """The next free test_id in a category's block (works for default AND custom categories)."""
        cat_name = self._resolve_category(require_text(category, "category").strip())
        (start,) = self._db.query(
            "SELECT id_start FROM test_categories WHERE name = ?", (cat_name,)
        )[0]
        first, last = int(start) + 1, int(start) + IDS_PER_CATEGORY - 1
        (top,) = self._db.query(
            "SELECT MAX(test_id) FROM fitness_tests WHERE test_id BETWEEN ? AND ?", (first, last)
        )[0]
        following = first if top is None else top + 1
        if following > last:
            raise ValidationError(f"{cat_name} has no free test_id left ({first}-{last} are used)")
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
        """Tune a test's plausible range. Refused for Level:Shuttle tests or if existing results fall outside."""
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

    def all(self, *, category: str | None = None) -> list[FitnessTest]:
        return [FitnessTest.from_row(row) for row in self._rows(category)]

    def to_dataframe(self, *, category: str | None = None) -> pl.DataFrame:
        """The table as a Polars dataframe with dynamic Enum/Float64 types."""
        df = pl.DataFrame(self._rows(category), schema=_DATAFRAME_RAW_SCHEMA, orient="row")
        return df.with_columns(
            pl.col("category").cast(pl.Enum(list(self.categories().keys()))),
            pl.col("unit").cast(pl.Enum(self.units())),
            pl.col("score_direction").cast(pl.Enum([d.value for d in ScoreDirection])),
        )

    # -- private helpers --
    def _resolve_category(self, category: str) -> str:
        cleaned = " ".join(require_text(category, "category").split())
        rows = self._db.query(
            "SELECT name FROM test_categories WHERE lower(name) = lower(?)", (cleaned,)
        )
        if not rows:
            allowed = list(self.categories().keys())
            raise ValidationError(f"category must be one of {allowed} (or add it via add_category())")
        return rows[0][0]

    def _resolve_unit(self, unit: str) -> str:
        cleaned = " ".join(require_text(unit, "unit").split())
        rows = self._db.query(
            "SELECT name FROM test_units WHERE lower(name) = lower(?)", (cleaned,)
        )
        if not rows:
            allowed = self.units()
            raise ValidationError(f"unit must be one of {allowed} (or add it via add_unit())")
        return rows[0][0]

    def _rows(self, category: str | None) -> list[tuple]:
        if category is None:
            return self._db.query(_SELECT + " ORDER BY test_id")
        cat = self._resolve_category(category)
        return self._db.query(_SELECT + " WHERE category = ? ORDER BY test_id", (cat,))

    def _check_table_is_current(self) -> None:
        """Stop with a clear message if whc.db still has an older fitness_tests table."""
        columns = tuple(row[1] for row in self._db.query("PRAGMA table_info(fitness_tests)"))
        if columns != _COLUMNS:
            raise RuntimeError(
                "data/whc.db has an old fitness_tests table. Rebuild it cleanly before importing."
            )

    def _fill_lookup_tables(self) -> None:
        """Make sure every default Category and Unit exists in the database (safe to repeat)."""
        with self._db.transaction() as conn:
            conn.executemany("INSERT OR IGNORE INTO test_categories (name, id_start) VALUES (?, ?)",
                             [(c.value, c.id_start) for c in Category])
            conn.executemany("INSERT OR IGNORE INTO test_units (name) VALUES (?)",
                             [(u.value,) for u in Unit])
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
