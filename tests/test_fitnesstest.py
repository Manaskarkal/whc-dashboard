# pyright: reportArgumentType=false
"""Tries to break the fitness_tests table. Run with:  uv run python -m pytest

Sections:
    1. FitnessTest: cleaning and validation (Python side)
    2. ID blocks: every category owns a block of 100 numbers
    3. Repository: saving, loading, duplicates, filtering, next_id, rename, no delete
    4. The database itself refuses bad data, even if Python is bypassed
    5. The standard catalogue (the 35 tests you asked for) and load_catalogue()
    6. Dataframes
    7. Table upkeep: re-running, lookup tables, rebuilt triggers, changed blocks

Every test says what it protects. Where a test uses `match=`, it also proves the RIGHT
rule fired. Where two rules would both be broken by the same row, SQLite does not promise
which message wins, so those cases only check that the row is refused.
"""

import sqlite3
from dataclasses import FrozenInstanceError, replace
from typing import Any

import polars as pl
import pytest

from whc.database import Database
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.fitness_tests import (
    DEFAULT_TESTS,
    IDS_PER_CATEGORY,
    NAME_MAX_LENGTH,
    Category,
    FitnessTest,
    FitnessTestRepository,
    Unit,
)

# A test that passes every rule. Tests copy it and change ONE field to break ONE rule.
GOOD: dict[str, Any] = dict(
    test_id=101, test_name="Counter Movement Jump Height", category="Power", unit="Centimetres"
)

RAW_INSERT = "INSERT INTO fitness_tests (test_id, test_name, category, unit) VALUES (?, ?, ?, ?)"


# ---- Fixtures -----------------------------------------------------------------------
@pytest.fixture
def db():
    with Database(":memory:") as database:  # fresh, empty, in-memory for every test
        yield database


@pytest.fixture
def repo(db):
    return FitnessTestRepository(db)


@pytest.fixture
def loaded(repo):
    """The repository with the standard 35-test catalogue already saved."""
    repo.load_catalogue()
    return repo


@pytest.fixture
def one(repo):
    """The repository with only GOOD saved."""
    repo.add(FitnessTest.new(**GOOD))
    return repo


# ==== 1. FitnessTest: cleaning and validation =============================================
def test_new_cleans_input():
    """new() collapses spaces/newlines in the name and matches category and unit loosely."""
    t = FitnessTest.new(101, "  Counter   Movement\nJump Height ", " pOwEr ", "  centimetres")
    assert t.test_name == "Counter Movement Jump Height"
    assert (t.category, t.unit) == (Category.POWER, Unit.CENTIMETRES)


def test_new_accepts_unit_with_punctuation_in_any_case():
    """'shuttles:levels' and 'watts/kilogram' contain : and / - they must still match the enum."""
    assert FitnessTest.new(**{**GOOD, "unit": "SHUTTLES:levels"}).unit is Unit.SHUTTLES_LEVELS
    assert FitnessTest.new(**{**GOOD, "unit": "watts/KILOGRAM"}).unit is Unit.WATTS_PER_KILOGRAM


@pytest.mark.parametrize("change, message", [
    (dict(test_id="101"), "whole number"),                # text that looks like a number
    (dict(test_id=101.0), "whole number"),                # float
    (dict(test_id=True), "whole number"),                 # True counts as 1 in Python
    (dict(test_id=None), "whole number"),
    (dict(test_id=0), "test_id must be between"),
    (dict(test_id=-101), "test_id must be between"),
    (dict(test_id=100), "multiple of"),                   # a block's own round number is unused
    (dict(test_id=1000), "test_id must be between"),
    (dict(test_name=""), "test_name must be"),
    (dict(test_name="   "), "test_name must be"),         # blank once tidied
    (dict(test_name=None), "test_name must be text"),
    (dict(test_name=42), "test_name must be text"),
    (dict(test_name="123"), "test_name must be"),         # needs at least one letter
    (dict(test_name="(RSI)"), "test_name must be"),       # must start with a letter or digit
    (dict(test_name="Jump; DROP TABLE fitness_tests"), "test_name must be"),
    (dict(test_name="Jump_Height"), "test_name must be"), # underscore is not allowed
    (dict(test_name="Jump\u2013Height"), "test_name must be"),   # en dash, not a hyphen
    (dict(test_name="A" * (NAME_MAX_LENGTH + 1)), "test_name must be"),
    (dict(category="Speedy"), "category must be one of"),
    (dict(category="   "), "category must be one of"),
    (dict(category=None), "category must be one of"),
    (dict(unit="Furlongs"), "unit must be one of"),
    (dict(unit=None), "unit must be one of"),
])
def test_bad_input_refused(change, message):
    """One bad field at a time; each must fail for ITS OWN reason (see `message`)."""
    with pytest.raises(ValidationError, match=message):
        FitnessTest.new(**{**GOOD, **change})


def test_name_length_limit_in_python():
    """80 characters is fine, 81 is refused (the SQL side is tested in section 4)."""
    FitnessTest.new(**{**GOOD, "test_name": "A" * NAME_MAX_LENGTH})
    with pytest.raises(ValidationError, match="test_name must be"):
        FitnessTest.new(**{**GOOD, "test_name": "A" * (NAME_MAX_LENGTH + 1)})


def test_fitness_test_cannot_be_edited():
    """The dataclass is frozen, so a created test can never be changed in memory."""
    with pytest.raises(FrozenInstanceError):
        FitnessTest.new(**GOOD).unit = Unit.SECONDS  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("build, message", [
    (lambda: FitnessTest(101, "Jump", "Power", Unit.SECONDS), "category must be a Category"),
    (lambda: FitnessTest(101, "Jump", Category.POWER, "Seconds"), "unit must be a Unit"),
    (lambda: FitnessTest("101", "Jump", Category.POWER, Unit.SECONDS), "whole number"),
    (lambda: FitnessTest(101, "Jump\n", Category.POWER, Unit.SECONDS), "test_name"),
    (lambda: FitnessTest(101, " Jump", Category.POWER, Unit.SECONDS), "test_name"),
    (lambda: FitnessTest(101, "Jump  Height", Category.POWER, Unit.SECONDS), "test_name"),
    (lambda: FitnessTest(14, "Jump", Category.POWER, Unit.SECONDS), "does not belong to Power"),
], ids=["category-text", "unit-text", "id-text", "name-newline", "name-leading-space",
        "name-double-space", "id-in-wrong-block"])
def test_directly_built_invalid_test_refused(build, message):
    """Building FitnessTest(...) directly skips new(), so __post_init__ must guard every rule.

    The newline case is a regression guard: '$' in a regex would let a trailing newline through.
    """
    with pytest.raises(ValidationError, match=message):
        build()


@pytest.mark.parametrize("name", [
    "5-0-5 Agility Test Left", "1.6k/Mile Run", "YOYO Intermittent Recovery Test 1 (IRT1)",
    "Sit-and-Reach Test", "Coach's Special Test", "Test: Part 2",
])
def test_names_with_allowed_punctuation_are_accepted(name):
    """Proves the name rule is not too strict for the real test names."""
    assert FitnessTest(101, name, Category.POWER, Unit.SECONDS).test_name == name


# ==== 2. ID blocks ==========================================================================
def test_every_category_has_its_own_block():
    """No two categories share a block, and every block starts on a multiple of 100."""
    starts = [c.id_start for c in Category]
    assert len(set(starts)) == len(starts)
    assert all(s % IDS_PER_CATEGORY == 0 for s in starts)


def test_the_blocks_you_asked_for():
    """Strength 0-99, Power 100-199, and so on - written down so a change here is deliberate."""
    assert Category.STRENGTH.id_start == 0
    assert Category.POWER.id_start == 100
    assert Category.SPEED.id_start == 200
    assert Category.AGILITY.id_start == 300
    assert Category.AEROBIC_ENDURANCE.id_start == 400
    assert Category.ANAEROBIC_ENDURANCE.id_start == 500
    assert Category.MUSCULAR_ENDURANCE.id_start == 600
    assert Category.FLEXIBILITY.id_start == 700
    assert Category.REACTION.id_start == 800


@pytest.mark.parametrize("test_id, accepted", [
    (101, True),    # first Power ID
    (199, True),    # last Power ID
    (99, False),    # last Strength ID: not a Power ID
    (100, False),   # Power's own round number is unused
    (200, False),   # Speed's round number
    (201, False),   # first Speed ID: not a Power ID
])
def test_power_block_edges(test_id, accepted):
    """The exact edges of the Power block (101-199), tested against the neighbouring blocks."""
    if accepted:
        FitnessTest(test_id, "Jump", Category.POWER, Unit.SECONDS)
    else:
        with pytest.raises(ValidationError):
            FitnessTest(test_id, "Jump", Category.POWER, Unit.SECONDS)


@pytest.mark.parametrize("category", list(Category))
def test_first_and_last_id_of_every_block(category):
    """Each block accepts start+1 and start+99, and refuses the ID just outside either end."""
    start = category.id_start
    FitnessTest(start + 1, "Jump", category, Unit.SECONDS)
    FitnessTest(start + 99, "Jump", category, Unit.SECONDS)
    for outside in (start - 1, start + 101):
        if 1 <= outside <= 999 and outside % 100 != 0:
            with pytest.raises(ValidationError, match="does not belong"):
                FitnessTest(outside, "Jump", category, Unit.SECONDS)


# ==== 3. Repository =======================================================================
def test_add_and_get_round_trip(one):
    """What we save is exactly what we load back."""
    assert one.get(101) == FitnessTest.new(**GOOD)


def test_get_by_name_ignores_capitals_and_spaces(one):
    assert one.get_by_name("  counter   MOVEMENT jump height ").test_id == 101


def test_get_by_name_missing_raises_not_found(one):
    with pytest.raises(NotFoundError, match="no fitness test is named"):
        one.get_by_name("Vertical Jump")


def test_duplicate_id_refused(one):
    """Same ID, different test -> DuplicateError about the ID (and the original is untouched)."""
    with pytest.raises(DuplicateError, match="test_id 101 already exists"):
        one.add(FitnessTest.new(101, "Something Else", "Power", "Metres"))
    assert one.get(101).test_name == "Counter Movement Jump Height"


def test_duplicate_name_refused_ignoring_capitals(one):
    """New ID, same name in other capitals -> DuplicateError about the name."""
    with pytest.raises(DuplicateError, match="already exists"):
        one.add(FitnessTest.new(102, "COUNTER MOVEMENT JUMP HEIGHT", "Power", "Centimetres"))


def test_same_name_is_refused_even_in_another_category(one):
    """A test name identifies a test on its own, whatever its category."""
    with pytest.raises(DuplicateError, match="already exists"):
        one.add(FitnessTest.new(1, "Counter Movement Jump Height", "Strength", "Newtons"))


def test_different_tests_may_share_a_unit_and_category(one):
    """Proves the unique rules are not too strict (many tests share a category and unit)."""
    one.add(FitnessTest.new(102, "Broad Jump", "Power", "Centimetres"))
    assert len(one.all()) == 2


def test_missing_test_raises_not_found(repo):
    with pytest.raises(NotFoundError, match="test_id 999 does not exist"):
        repo.get(999)


@pytest.mark.parametrize("bad_id", ["abc", None, True, 1.5])
def test_id_argument_must_be_a_whole_number(repo, bad_id):
    with pytest.raises(ValidationError, match="whole number"):
        repo.get(bad_id)


def test_add_refuses_things_that_are_not_fitness_tests(repo):
    """A dict or tuple must give OUR error, not an AttributeError halfway through a save."""
    with pytest.raises(ValidationError, match="only FitnessTest"):
        repo.add(GOOD)
    with pytest.raises(ValidationError, match="only FitnessTest"):
        repo.add_many([FitnessTest.new(**GOOD), ("102", "x", "Power", "Metres")])
    assert repo.all() == []  # the good one in the list was NOT saved either


def test_add_many_is_all_or_nothing(repo):
    """If the second test is refused, the first must not stay saved."""
    first = FitnessTest.new(101, "Jump A", "Power", "Centimetres")
    clash = FitnessTest.new(102, "jump a", "Power", "Centimetres")  # same name as `first`
    with pytest.raises(DuplicateError):
        repo.add_many([first, clash])
    assert repo.all() == []


def test_add_many_with_the_same_id_twice_in_one_batch_is_refused(repo):
    a = FitnessTest.new(101, "Jump A", "Power", "Centimetres")
    b = FitnessTest.new(101, "Jump B", "Power", "Centimetres")
    with pytest.raises(DuplicateError, match="test_id 101"):
        repo.add_many([a, b])
    assert repo.all() == []


def test_add_many_accepts_a_generator_and_an_empty_list(repo):
    repo.add_many([])
    repo.add_many(t for t in [FitnessTest.new(**GOOD)])
    assert len(repo.all()) == 1


def test_filter_by_category(loaded):
    """Needs several categories: with one category, a missing filter would still pass."""
    power = loaded.all(category="power")  # lowercase text on purpose
    assert [t.test_id for t in power] == [101, 102, 103, 104, 105]
    assert {t.category for t in power} == {Category.POWER}
    assert [t.test_id for t in loaded.all(category=Category.REACTION)] == [801]
    assert len(loaded.all()) == 35


def test_filter_by_unknown_category_is_refused(loaded):
    with pytest.raises(ValidationError, match="category must be one of"):
        loaded.all(category="Balance")


def test_results_are_ordered_by_test_id(repo):
    """Saved in a jumbled order, read back in ID order every time."""
    for tid, cat in ((301, "Agility"), (1, "Strength"), (201, "Speed"), (101, "Power")):
        repo.add(FitnessTest.new(tid, f"Test {tid}", cat, "Seconds"))
    assert [t.test_id for t in repo.all()] == [1, 101, 201, 301]


# ---- next_id ----
def test_next_id_starts_each_block_at_one(repo):
    assert repo.next_id("Strength") == 1
    assert repo.next_id("Power") == 101
    assert repo.next_id("reaction") == 801


def test_next_id_follows_the_highest_id_in_the_block(loaded):
    assert loaded.next_id("Strength") == 15
    assert loaded.next_id(Category.POWER) == 106
    assert loaded.next_id("Flexibility") == 702


def test_next_id_does_not_look_at_other_blocks(repo):
    """A test numbered 105 must not push Speed's next ID to 106."""
    repo.add(FitnessTest.new(105, "Jump", "Power", "Metres"))
    assert repo.next_id("Speed") == 201


def test_next_id_can_be_used_to_add_a_test(loaded):
    new_id = loaded.next_id("Power")
    loaded.add(FitnessTest.new(new_id, "Depth Jump", "Power", "Centimetres"))
    assert loaded.get(new_id).test_name == "Depth Jump"
    assert loaded.next_id("Power") == new_id + 1


def test_next_id_refuses_when_the_block_is_full(repo):
    """99 tests fill a block; the 100th must be refused cleanly, not spill into the next block."""
    repo.add_many(FitnessTest(101 + i, f"Jump {chr(65 + i // 26)}{chr(65 + i % 26)}",
                              Category.POWER, Unit.SECONDS) for i in range(99))
    with pytest.raises(ValidationError, match="no free test_id left"):
        repo.next_id("Power")


def test_next_id_unknown_category(repo):
    with pytest.raises(ValidationError, match="category must be one of"):
        repo.next_id("Balance")


# ---- rename ----
def test_rename_changes_only_the_name(one):
    one.rename(101, "  CMJ   Height ")
    after = one.get(101)
    assert after.test_name == "CMJ Height"
    assert (after.category, after.unit, after.test_id) == (Category.POWER, Unit.CENTIMETRES, 101)


def test_rename_can_fix_capitals_of_its_own_name(one):
    """Renaming a test to the same name in other capitals must not clash with ITSELF."""
    one.rename(101, "COUNTER MOVEMENT JUMP HEIGHT")
    assert one.get(101).test_name == "COUNTER MOVEMENT JUMP HEIGHT"


def test_rename_to_another_tests_name_is_refused(loaded):
    with pytest.raises(DuplicateError, match="already exists"):
        loaded.rename(102, "counter movement jump height")
    assert loaded.get(102).test_name == "Counter Movement Jump Power"  # unchanged


def test_rename_refuses_a_bad_name_and_changes_nothing(one):
    with pytest.raises(ValidationError, match="test_name must be"):
        one.rename(101, "Bad;Name")
    with pytest.raises(ValidationError, match="test_name must be text"):
        one.rename(101, None)
    assert one.get(101).test_name == "Counter Movement Jump Height"


def test_rename_missing_test_raises_not_found(repo):
    with pytest.raises(NotFoundError):
        repo.rename(101, "Whatever")


@pytest.mark.parametrize("bad_id", ["101", None, True])
def test_rename_needs_a_whole_number_id(repo, bad_id):
    with pytest.raises(ValidationError, match="whole number"):
        repo.rename(bad_id, "Whatever")


def test_rename_leaves_every_other_test_alone(loaded):
    """If the WHERE clause were lost, every test would get the same name (and clash)."""
    before = {t.test_id: t for t in loaded.all()}
    loaded.rename(101, "CMJ Height")
    after = {t.test_id: t for t in loaded.all()}
    assert after[101].test_name == "CMJ Height"
    assert {k: v for k, v in after.items() if k != 101} == {k: v for k, v in before.items() if k != 101}


def test_tests_cannot_be_deleted(db, one):
    """There is deliberately no delete(), and the database refuses a raw DELETE too."""
    assert not hasattr(one, "delete")
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        with db.transaction() as conn:
            conn.execute("DELETE FROM fitness_tests")
    assert len(one.all()) == 1


# ==== 4. The database itself refuses bad data, even if Python is bypassed ================
@pytest.mark.parametrize("values, message", [
    ((106, "Jump", "Balance", "Seconds"), "FOREIGN KEY"),             # category not in lookup table
    ((106, "Jump", "Power", "Furlongs"), "FOREIGN KEY"),              # unit not in lookup table
    ((None, "Jump", "Power", "Seconds"), "NOT NULL"),                 # NULL ID must not auto-number
    ((106, None, "Power", "Seconds"), "NOT NULL"),
    ((106, "Jump", None, "Seconds"), "NOT NULL"),
    ((106, "Jump", "Power", None), "NOT NULL"),
    (("abc", "Jump", "Power", "Seconds"), "test_id"),                 # STRICT: wrong type
    ((106, 123, "Power", "Seconds"), "test_name"),                    # STRICT: number as name
    ((0, "Jump", "Strength", "Seconds"), "test_id"),                  # 0 is never used
    ((100, "Jump", "Power", "Seconds"), "test_id"),                   # round numbers unused
    ((-5, "Jump", "Strength", "Seconds"), "test_id"),                 # negative
    ((5, "Jump", "Power", "Seconds"), "block of this category"),      # Strength ID, Power category
    ((201, "Jump", "Power", "Seconds"), "block of this category"),    # Speed ID, Power category
    ((1000, "Jump", "Power", "Seconds"), "block of this category"),   # beyond every block
    ((106, "", "Power", "Seconds"), "test_name"),                     # empty name
    ((106, " Jump", "Power", "Seconds"), "test_name"),                # leading space
    ((106, "Jump ", "Power", "Seconds"), "test_name"),                # trailing space
    ((106, "Jump\n", "Power", "Seconds"), "test_name"),               # trailing newline
    ((106, "Jump\tHeight", "Power", "Seconds"), "test_name"),         # tab inside
    ((106, "Jump  Height", "Power", "Seconds"), "test_name"),         # double space
    ((106, "(Jump)", "Power", "Seconds"), "test_name"),               # starts with punctuation
    ((106, "12345", "Power", "Seconds"), "test_name"),                # no letter
    ((106, "Jump_Height", "Power", "Seconds"), "test_name"),          # underscore
    ((106, "Jump;--", "Power", "Seconds"), "test_name"),              # SQL-ish characters
    ((106, "A" * (NAME_MAX_LENGTH + 1), "Power", "Seconds"), "test_name"),  # too long
], ids=["bad-category", "bad-unit", "null-id", "null-name", "null-category", "null-unit",
        "text-id", "numeric-name", "id-zero", "id-100", "id-negative", "strength-id-in-power",
        "speed-id-in-power", "id-1000", "empty-name", "leading-space", "trailing-space",
        "newline", "tab", "double-space", "starts-with-bracket", "no-letter", "underscore",
        "sql-chars", "long-name"])
def test_database_refuses_bad_rows(db, repo, values, message):
    """Raw INSERTs skip every Python check. Each must still be refused, by the right rule."""
    with pytest.raises(sqlite3.IntegrityError, match=message):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, values)


def test_database_accepts_a_name_of_exactly_the_maximum_length(db, repo):
    """Proves the length limit is a limit, not an off-by-one that refuses valid names."""
    with db.transaction() as conn:
        conn.execute(RAW_INSERT, (106, "A" * NAME_MAX_LENGTH, "Power", "Seconds"))


def test_database_refuses_duplicate_names_ignoring_capitals(db, one):
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, (106, "counter movement jump HEIGHT", "Power", "Seconds"))


@pytest.mark.parametrize("column, value", [
    ("test_id", 102), ("category", "Strength"), ("unit", "Seconds"),
])
def test_every_locked_column_is_locked(db, one, column, value):
    """Every column the trigger claims to lock must really refuse to change."""
    with pytest.raises(sqlite3.DatabaseError, match="locked"):
        with db.transaction() as conn:
            conn.execute(f"UPDATE fitness_tests SET {column} = ?", (value,))


def test_name_is_not_locked_but_still_checked(db, one):
    """Renaming at SQL level works, and the name rules still apply to an UPDATE."""
    with db.transaction() as conn:
        conn.execute("UPDATE fitness_tests SET test_name = 'CMJ Height'")
    with pytest.raises(sqlite3.IntegrityError, match="test_name"):
        with db.transaction() as conn:
            conn.execute("UPDATE fitness_tests SET test_name = 'bad;name'")


def test_foreign_keys_protect_the_lookup_tables(db, one):
    """A category or unit that tests use cannot be deleted from its lookup table."""
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with db.transaction() as conn:
            conn.execute("DELETE FROM test_categories WHERE name = 'Power'")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with db.transaction() as conn:
            conn.execute("DELETE FROM test_units WHERE name = 'Centimetres'")


def test_a_refused_raw_insert_leaves_nothing_behind(db, one):
    with pytest.raises(sqlite3.IntegrityError):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, (102, "Fine Name", "Power", "Seconds"))  # valid row...
            conn.execute(RAW_INSERT, (5, "Other Name", "Power", "Seconds"))   # ...then a bad one
    assert [t.test_id for t in one.all()] == [101]  # the valid row was rolled back too


# ==== 5. The standard catalogue =============================================================
def test_catalogue_has_the_35_requested_tests():
    assert len(DEFAULT_TESTS) == 35


def test_catalogue_has_no_duplicate_ids_or_names():
    ids = [t.test_id for t in DEFAULT_TESTS]
    names = [t.test_name.casefold() for t in DEFAULT_TESTS]
    assert len(set(ids)) == len(ids)
    assert len(set(names)) == len(names)


def test_every_catalogue_test_sits_in_its_categorys_block():
    """The headline rule of the numbering scheme, checked over the whole catalogue."""
    for t in DEFAULT_TESTS:
        assert t.test_id // IDS_PER_CATEGORY * IDS_PER_CATEGORY == t.category.id_start, t


def test_every_category_and_unit_is_used_by_the_catalogue():
    """An unused bucket or unit usually means a test was filed under the wrong one."""
    assert {t.category for t in DEFAULT_TESTS} == set(Category)
    assert {t.unit for t in DEFAULT_TESTS} == set(Unit)


# (name, category, unit): the list you asked for, written out independently of the module,
# so a wrong unit or bucket in DEFAULT_TESTS is caught by a second pair of eyes.
EXPECTED = [
    ("Counter Movement Jump Height", "Power", "Centimetres"),
    ("Counter Movement Jump Power", "Power", "Watts/Kilogram"),
    ("Counter Movement Jump Relative Strength Index (RSI)", "Power", "Unitless"),
    ("Hop Jump Relative Strength Index (RSI)", "Power", "Unitless"),
    ("Standing Long Jump (Broad Jump)", "Power", "Metres"),
    ("Isometric Mid-Thigh Pull Peak", "Strength", "Newtons"),
    ("Isometric Mid-Thigh Pull Relative", "Strength", "Unitless"),
    ("YOYO Intermittent Recovery Test 1 (IRT1)", "Aerobic Endurance", "Shuttles:Levels"),
    ("YOYO Intermittent Recovery Test 2 (IRT2)", "Anaerobic Endurance", "Shuttles:Levels"),
    ("20m Beep Test (Shuttle Run)", "Aerobic Endurance", "Shuttles:Levels"),
    ("Cooper 12-Minute Run", "Aerobic Endurance", "Metres"),
    ("1.6k/Mile Run", "Aerobic Endurance", "Seconds"),
    ("2k Run", "Aerobic Endurance", "Seconds"),
    ("5 Metre Sprint", "Speed", "Seconds"),
    ("10 Metre Sprint", "Speed", "Seconds"),
    ("20 Metre Sprint", "Speed", "Seconds"),
    ("5-0-5 Agility Test Left", "Agility", "Seconds"),
    ("5-0-5 Agility Test Right", "Agility", "Seconds"),
    ("NBA Lane Agility Test", "Agility", "Seconds"),
    ("Nordic Hamstring Left", "Strength", "Newtons"),
    ("Nordic Hamstring Right", "Strength", "Newtons"),
    ("Adduction Left", "Strength", "Newtons"),
    ("Adduction Right", "Strength", "Newtons"),
    ("Abduction Left", "Strength", "Newtons"),
    ("Abduction Right", "Strength", "Newtons"),
    ("1RM Back Squat", "Strength", "Kilograms"),
    ("1RM Bench Press", "Strength", "Kilograms"),
    ("1RM Trap Bar Deadlift", "Strength", "Kilograms"),
    ("3RM Back Squat", "Strength", "Kilograms"),
    ("3RM Bench Press", "Strength", "Kilograms"),
    ("3RM Trap Bar Deadlift", "Strength", "Kilograms"),
    ("Push Up Test (1-Minute)", "Muscular Endurance", "Reps"),
    ("Pull Up Test (1-Minute)", "Muscular Endurance", "Reps"),
    ("Sit-and-Reach Test", "Flexibility", "Centimetres"),
    ("Reaction Time Ruler Drop Test", "Reaction", "Centimetres"),
]


def test_catalogue_matches_the_requested_list_exactly():
    got = sorted((t.test_name, t.category.value, t.unit.value) for t in DEFAULT_TESTS)
    assert got == sorted(EXPECTED)


def test_load_catalogue_saves_everything(loaded):
    assert loaded.all() == sorted(DEFAULT_TESTS, key=lambda t: t.test_id)


def test_load_catalogue_is_repeatable_and_does_not_duplicate(loaded):
    loaded.load_catalogue()
    loaded.load_catalogue()
    assert len(loaded.all()) == 35


def test_load_catalogue_keeps_other_tests_and_fills_the_gaps(repo):
    """A database that already has some rows (even your own extra test) just gets the rest."""
    repo.add(FitnessTest.new(106, "Depth Jump", "Power", "Centimetres"))  # my own extra test
    repo.add(DEFAULT_TESTS[0])  # one catalogue test already there
    repo.load_catalogue()
    assert len(repo.all()) == 36
    assert repo.get(106).test_name == "Depth Jump"


def test_load_catalogue_respects_a_rename(loaded):
    """Renaming a catalogue test must not make start-up fail or bring the old name back."""
    loaded.rename(101, "CMJ Height")
    loaded.load_catalogue()
    assert loaded.get(101).test_name == "CMJ Height"
    assert len(loaded.all()) == 35


def test_load_catalogue_refuses_to_overwrite_a_conflicting_row(repo):
    """ID 101 already means something else here: results would be mislabelled, so stop."""
    repo.add(FitnessTest.new(101, "Depth Jump", "Power", "Metres"))  # catalogue says Centimetres
    with pytest.raises(ValidationError, match="test_id 101 is already used"):
        repo.load_catalogue()
    assert len(repo.all()) == 1  # nothing else was loaded either (all-or-nothing)


def test_load_catalogue_refuses_a_name_taken_by_another_id(repo):
    """The catalogue's name is already in use at a different ID: clean DuplicateError, no partial load."""
    repo.add(FitnessTest.new(106, "Counter Movement Jump Height", "Power", "Centimetres"))
    with pytest.raises(DuplicateError, match="already exists"):
        repo.load_catalogue()
    assert len(repo.all()) == 1


# ==== 6. Dataframes =========================================================================
def test_dataframe_types_even_when_empty(repo):
    schema = repo.to_dataframe().schema
    assert schema["test_id"] == pl.Int64
    assert schema["test_name"] == pl.String
    assert isinstance(schema["category"], pl.Enum)
    assert isinstance(schema["unit"], pl.Enum)


def test_dataframe_has_real_values(one):
    """The empty-frame test only checks types; this checks a FILLED frame converts correctly."""
    df = one.to_dataframe()
    assert df["test_id"].to_list() == [101]
    assert df["test_name"].to_list() == ["Counter Movement Jump Height"]
    assert df["category"].to_list() == ["Power"]
    assert df["unit"].to_list() == ["Centimetres"]


def test_dataframe_column_names_and_order(repo):
    assert repo.to_dataframe().columns == ["test_id", "test_name", "category", "unit"]


def test_dataframe_category_filter(loaded):
    df = loaded.to_dataframe(category="Speed")
    assert df["test_id"].to_list() == [201, 202, 203]
    assert len(loaded.to_dataframe()) == 35


def test_dataframe_enum_lists_every_category_even_if_unused(one):
    """Charts and group-bys later need all 9 buckets, not just the ones that have rows."""
    assert one.to_dataframe().schema["category"] == pl.Enum([c.value for c in Category])


# ==== 7. Table upkeep =======================================================================
def test_table_creation_is_repeatable_and_keeps_data(db, one):
    """Building the repository again must not fail AND must not wipe what is already saved."""
    FitnessTestRepository(db)
    FitnessTestRepository(db)
    assert len(FitnessTestRepository(db).all()) == 1


def test_lookup_tables_match_the_enums(db, repo):
    """The database holds exactly the enums' categories (with their blocks) and units."""
    cats = dict(db.query("SELECT name, id_start FROM test_categories"))
    units = {row[0] for row in db.query("SELECT name FROM test_units")}
    assert cats == {c.value: c.id_start for c in Category}
    assert units == {u.value for u in Unit}


def test_a_unit_missing_from_an_old_database_is_added_on_startup(db, repo):
    """The reason lookup tables exist: an older file that lacks a unit picks it up on start-up."""
    with db.transaction() as conn:
        conn.execute("DELETE FROM test_units WHERE name = 'Reps'")  # pretend an old file
    FitnessTestRepository(db)  # start-up runs again
    repo.add(FitnessTest.new(601, "Sit Ups", "Muscular Endurance", "Reps"))  # fails without the refill


def test_a_changed_block_in_an_old_database_stops_start_up(db, repo):
    """If a category's block differs from what the code says, IDs would be renumbered: refuse."""
    with db.transaction() as conn:
        conn.execute("UPDATE test_categories SET id_start = 900 WHERE name = 'Reaction'")
    with pytest.raises(RuntimeError, match="Reaction IDs starting at 900"):
        FitnessTestRepository(db)


def test_the_database_blocks_come_from_the_code_not_from_enum_order(db, repo):
    """Strength is the FIRST enum member and starts at 0; Reaction is LAST and starts at 800."""
    blocks = dict(db.query("SELECT name, id_start FROM test_categories"))
    assert blocks["Strength"] == 0 and blocks["Reaction"] == 800
    assert list(Category)[0] is Category.STRENGTH


def test_triggers_are_rebuilt_on_startup(db, repo):
    """An out-of-date rule is replaced at start-up (CREATE ... IF NOT EXISTS would keep it)."""
    with db.transaction() as conn:  # pretend an old version that never fires is stored in the file
        conn.execute("DROP TRIGGER fitness_tests_id_in_category_block")
        conn.execute("CREATE TRIGGER fitness_tests_id_in_category_block BEFORE INSERT ON "
                     "fitness_tests WHEN 0 BEGIN SELECT 1; END")
    FitnessTestRepository(db)  # start-up runs again
    with pytest.raises(sqlite3.IntegrityError, match="block of this category"):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, (5, "Jump", "Power", "Seconds"))


def test_the_other_triggers_are_rebuilt_too(db, repo):
    """The lock and the no-delete rule also come back if an old file had them disabled."""
    with db.transaction() as conn:
        conn.execute("DROP TRIGGER fitness_tests_no_delete")
        conn.execute("DROP TRIGGER fitness_tests_lock_identity")
    FitnessTestRepository(db)
    repo.add(FitnessTest.new(**GOOD))
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        with db.transaction() as conn:
            conn.execute("DELETE FROM fitness_tests")
    with pytest.raises(sqlite3.DatabaseError, match="locked"):
        with db.transaction() as conn:
            conn.execute("UPDATE fitness_tests SET unit = 'Seconds'")


def test_table_is_strict_and_without_rowid(db, repo):
    """Documents WHY NULL ids are refused: a plain INTEGER PRIMARY KEY would auto-number them."""
    (sql,) = db.query("SELECT sql FROM sqlite_master WHERE name = 'fitness_tests'")[0]
    assert "STRICT" in sql and "WITHOUT ROWID" in sql


def test_replace_cannot_smuggle_in_an_invalid_test(one):
    """dataclasses.replace() re-runs __post_init__, so a copied test is validated again."""
    good = one.get(101)
    with pytest.raises(ValidationError, match="does not belong"):
        replace(good, test_id=5)
    with pytest.raises(ValidationError, match="unit must be a Unit"):
        replace(good, unit="Seconds")
