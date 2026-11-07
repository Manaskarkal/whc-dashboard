# pyright: reportArgumentType=false
"""Tries to break the students table. Run with:  uv run python -m pytest

Sections:
    1. Student: cleaning and validation (Python side)
    2. Ages and dates
    3. Repository: saving, loading, duplicates, active flag, dataframe
    4. The database itself refuses bad data, even if Python is bypassed
    5. Table upkeep: re-running, locked columns, rebuilt triggers

Every test says what it protects. Where a test uses `match=`, it also proves the RIGHT
rule fired - a test that only says "some error happened" can pass for the wrong reason.
"""

import sqlite3
import unicodedata
from dataclasses import FrozenInstanceError
from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

import whc.students as students_module
from whc.database import Database
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.students import Sex, Student, StudentRepository

# A student that passes every rule. Tests copy it and change ONE field to break ONE rule.
GOOD: dict[str, Any] = dict(
    student_id="AAA0001", first_name="Sam", last_name="Lee", dob="14/03/2010", sex="Male"
)

# The raw INSERT used by the "bypass Python" tests (same column order as the tuples below).
RAW_INSERT = (
    "INSERT INTO students (student_id, first_name, last_name, dob, sex, is_active) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


# ---- Fixtures -----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def frozen_today(monkeypatch):
    """Freeze Python's 'today' at 5 Nov 2026 for EVERY test in this file.

    Without this, a test like 'a student born in 2010 is a valid new student' would start
    failing in 2036 when that student turns 26. (The SQL triggers still use the real clock.)
    """
    monkeypatch.setattr(students_module, "_today", lambda: date(2026, 11, 5))


@pytest.fixture
def db():
    with Database(":memory:") as database:  # fresh, empty, in-memory for every test
        yield database


@pytest.fixture
def repo(db):
    return StudentRepository(db)


@pytest.fixture
def sam(repo):
    """The repository with GOOD already saved."""
    repo.add(Student.new(**GOOD))
    return repo


# ==== 1. Student: cleaning and validation ===============================================
def test_new_cleans_input():
    """new() trims, uppercases the ID, collapses spaces, reads the date and the sex."""
    s = Student.new(" aaa0002 ", "  Mary   Ann ", " Van  Der Berg", "02/11/2011", "female")
    assert s.student_id == "AAA0002"
    assert s.first_name == "Mary Ann"
    assert s.last_name == "Van Der Berg"
    assert s.dob == date(2011, 11, 2)
    assert s.sex is Sex.FEMALE
    assert s.is_active is True


@pytest.mark.parametrize("change, message", [
    (dict(student_id="AA1234"), "student_id must be"),            # wrong ID format
    (dict(first_name="S4m"), "first_name must be letters"),       # digit in a name
    (dict(first_name="   "), "first_name must be letters"),       # blank once tidied
    (dict(dob="31/02/2010"), "real date"),                        # not a real date
    (dict(dob="2010-03-14"), "real date"),                        # wrong format
    (dict(dob="01/01/2099"), "in the past"),                      # future
    (dict(dob="01/01/1906"), "dob gives an age"),                 # too old to be a new student
    (dict(sex="M"), "sex must be one of"),                        # not an allowed value
])
def test_bad_input_refused(change, message):
    """One bad field at a time; each must fail for ITS OWN reason (see `message`)."""
    with pytest.raises(ValidationError, match=message):
        Student.new(**{**GOOD, **change})


def test_student_cannot_be_edited():
    """The dataclass is frozen, so a created student can never be changed in memory."""
    with pytest.raises(FrozenInstanceError):
        Student.new(**GOOD).first_name = "Tom"  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("build, message", [
    (lambda: Student("aaa0001", "Sam", "Lee", date(2010, 3, 14), Sex.MALE), "student_id"),
    (lambda: Student("AAA0001", "S4m", "Lee", date(2010, 3, 14), Sex.MALE), "first_name"),
    (lambda: Student("AAA0001", "Sam", "L33", date(2010, 3, 14), Sex.MALE), "last_name"),
    (lambda: Student("AAA0001", "Sam", "Lee", date(2010, 3, 14), "Male"), "sex must be a Sex"),
    (lambda: Student("AAA0001", "Sam", "Lee", date(2099, 1, 1), Sex.MALE), "in the past"),
    (lambda: Student("AAA0001", "Sam", "Lee", date(1899, 12, 31), Sex.MALE), "on or after"),
], ids=["id", "first_name", "last_name", "sex", "future-dob", "dob-before-1900"])
def test_directly_built_invalid_student_refused(build, message):
    """Building Student(...) directly skips new(), so __post_init__ must guard every rule."""
    with pytest.raises(ValidationError, match=message):
        build()


@pytest.mark.parametrize("field, bad", [
    ("student_id", "AAA0001\n"),  # "$" in a regex would let a trailing newline through
    ("first_name", "Sam\n"),      # same hole; SQLite's trim() also ignores newlines
    ("last_name", "Lee\n"),
])
def test_trailing_newline_refused(field, bad):
    """Regression: these used to be accepted (and 'Sam\\n' was even saved to the database)."""
    values = dict(student_id="AAA0001", first_name="Sam", last_name="Lee")
    values[field] = bad
    with pytest.raises(ValidationError):
        Student(dob=date(2010, 3, 14), sex=Sex.MALE, **values)


def test_accent_typed_two_ways_is_the_same_name():
    """'José' can be one character (NFC) or 'e' + a combining accent (NFD). Both must work."""
    nfd = unicodedata.normalize("NFD", "José")
    s = Student.new(**{**GOOD, "first_name": nfd})
    assert s.first_name == unicodedata.normalize("NFC", "José")


@pytest.mark.parametrize("build", [
    lambda: Student("AAA0001", "Sam", "Lee", "2010-03-14", Sex.MALE),             # dob is text
    lambda: Student(123, "Sam", "Lee", date(2010, 3, 14), Sex.MALE),              # id is a number
    lambda: Student("AAA0001", "Sam", "Lee", date(2010, 3, 14), Sex.MALE, "no"),  # is_active text
], ids=["dob-text", "id-number", "is_active-text"])
def test_wrong_types_raise_validation_error(build):
    """A wrong type must give OUR error, not a raw TypeError (or, for is_active, nothing)."""
    with pytest.raises(ValidationError):
        build()


@pytest.mark.parametrize("field", ["student_id", "first_name", "last_name", "dob", "sex"])
def test_none_input_raises_validation_error(field):
    """A blank form field often arrives as None; the caller gets a WhcError, not AttributeError."""
    with pytest.raises(ValidationError):
        Student.new(**{**GOOD, field: None})


# ==== 2. Ages and dates ==================================================================
def test_age_counts_birthday():
    """The day before the 16th birthday is 15; on the birthday it is 16."""
    s = Student.new(**GOOD)
    assert s.age(date(2026, 3, 13)) == 15
    assert s.age(date(2026, 3, 14)) == 16


def test_leap_day_birthday_counts_on_1_march():
    """Documents the rule for 29 Feb babies: they age up on 1 March in non-leap years."""
    s = Student("AAA0001", "Sam", "Lee", date(2008, 2, 29), Sex.MALE)
    assert s.age(date(2025, 2, 28)) == 16
    assert s.age(date(2025, 3, 1)) == 17


def test_age_before_birth_is_an_error():
    """Regression: age() used to quietly return a negative number."""
    with pytest.raises(ValidationError, match="before the date of birth"):
        Student.new(**GOOD).age(date(2000, 1, 1))


@pytest.mark.parametrize("dob, accepted", [
    ("05/11/2022", True),   # turns 4 exactly today  -> youngest allowed
    ("06/11/2022", False),  # turns 4 tomorrow       -> still 3, refused
    ("06/11/2000", True),   # turns 26 tomorrow      -> still 25, oldest allowed
    ("05/11/2000", False),  # turns 26 today         -> refused
])
def test_entry_age_boundaries(dob, accepted):
    """The exact edges of MIN_ENTRY_AGE (4) and MAX_ENTRY_AGE (25), with 'today' frozen."""
    if accepted:
        Student.new(**{**GOOD, "dob": dob})
    else:
        with pytest.raises(ValidationError, match="dob gives an age"):
            Student.new(**{**GOOD, "dob": dob})


def test_name_length_limit_in_python():
    """50 letters is fine, 51 is refused. (The SQL side of this limit is tested in section 4.)"""
    Student.new(**{**GOOD, "first_name": "A" * 50})
    with pytest.raises(ValidationError, match="max 50"):
        Student.new(**{**GOOD, "first_name": "A" * 51})


# ==== 3. Repository =======================================================================
def test_add_and_get_round_trip(repo):
    """What we save is exactly what we load back; lookups ignore capitals."""
    s = Student.new(**GOOD)
    repo.add(s)
    assert repo.get("aaa0001") == s


def test_duplicate_id_refused(sam):
    """Same ID, different person -> DuplicateError about the ID."""
    with pytest.raises(DuplicateError, match="student_id AAA0001 already exists"):
        sam.add(Student.new(**{**GOOD, "first_name": "Tom", "last_name": "Ng"}))


def test_same_person_refused_ignoring_capitals(sam):
    """New ID, but same name (any capitals) and DOB -> it is the same person."""
    with pytest.raises(DuplicateError, match="already in the database"):
        sam.add(Student.new(**{**GOOD, "student_id": "AAA0002", "first_name": "sam", "last_name": "LEE"}))


def test_same_person_refused_ignoring_accented_capitals(repo):
    """SQLite's lower() only knows A-Z, so this has to be caught in Python."""
    repo.add(Student.new("AAA0001", "Émile", "Lee", "14/03/2010", "Male"))
    with pytest.raises(DuplicateError, match="already in the database"):
        repo.add(Student.new("AAA0002", "émile", "Lee", "14/03/2010", "Male"))


def test_same_name_different_dob_is_a_different_person(sam):
    """Two Sam Lees are fine as long as their dates of birth differ."""
    sam.add(Student.new(**{**GOOD, "student_id": "AAA0002", "dob": "15/03/2010"}))
    assert len(sam.all()) == 2


def test_set_active_hides_student_from_active_only(sam):
    sam.set_active("aaa0001", False)  # lowercase on purpose: IDs are normalised
    assert sam.all(active_only=True) == []
    assert len(sam.all()) == 1


def test_set_active_can_reactivate(sam):
    sam.set_active("AAA0001", False)
    sam.set_active("AAA0001", True)
    assert sam.get("AAA0001").is_active is True


def test_set_active_changes_nothing_else(sam):
    before = sam.get("AAA0001")
    sam.set_active("AAA0001", False)
    after = sam.get("AAA0001")
    assert (after.first_name, after.last_name, after.dob, after.sex) == \
           (before.first_name, before.last_name, before.dob, before.sex)


def test_missing_student_raises_not_found(repo):
    with pytest.raises(NotFoundError):
        repo.get("ZZZ9999")
    with pytest.raises(NotFoundError):
        repo.set_active("ZZZ9999", True)


@pytest.mark.parametrize("bad", [5, "no", None])
def test_set_active_needs_a_real_bool(sam, bad):
    """Regression: 5 used to leak a raw sqlite3 error and 'no' a raw ValueError."""
    with pytest.raises(ValidationError, match="True or False"):
        sam.set_active("AAA0001", bad)


def test_dataframe_types_even_when_empty(repo):
    schema = repo.to_dataframe().schema
    assert schema["dob"] == pl.Date
    assert schema["is_active"] == pl.Boolean
    assert isinstance(schema["sex"], pl.Enum)


def test_dataframe_has_real_values(sam):
    """The empty-frame test only checks types; this checks a FILLED frame converts correctly."""
    df = sam.to_dataframe()
    assert df["student_id"].to_list() == ["AAA0001"]
    assert df["dob"].to_list() == [date(2010, 3, 14)]
    assert df["sex"].to_list() == ["Male"]
    assert df["is_active"].to_list() == [True]


# ==== 4. The database itself refuses bad data, even if Python is bypassed ================
@pytest.mark.parametrize("values, message", [
    (("aaa0001", "Sam", "Lee", "2010-03-14", "Male", 1), "student_id"),        # lowercase ID
    (("AAA0001", "Sam", "Lee", "2010-02-31", "Male", 1), "dob"),               # impossible date
    (("AAA0001", "Sam", "Lee", "14/03/2010", "Male", 1), "dob"),               # wrong format
    (("AAA0001", "Sam", "Lee", "2099-01-01", "Male", 1), "in the past"),       # future (trigger)
    (("AAA0001", "Sam", "Lee", "1899-12-31", "Male", 1), "dob"),               # before 1900
    (("AAA0001", "Sam", "Lee", "2010-03-14", "M", 1), "sex"),                  # not allowed
    (("AAA0001", "Sam", "Lee", "2010-03-14", "Male", 2), "is_active"),         # not 0/1
    (("AAA0001", "Sam", "Lee", "2010-03-14", "Male", "yes"), "is_active"),     # STRICT: wrong type
    (("AAA0001", " Sam", "Lee", "2010-03-14", "Male", 1), "first_name"),       # leading space
    (("AAA0001", "Sam\n", "Lee", "2010-03-14", "Male", 1), "first_name"),      # trailing newline
    (("AAA0001", "", "Lee", "2010-03-14", "Male", 1), "first_name"),           # empty
    (("AAA0001", "A" * 51, "Lee", "2010-03-14", "Male", 1), "first_name"),     # too long
    (("AAA0001", "Sam", "L" * 51, "2010-03-14", "Male", 1), "last_name"),      # too long
    (("AAA0001", None, "Lee", "2010-03-14", "Male", 1), "first_name"),         # NULL
], ids=["lower-id", "bad-date", "date-format", "future", "pre-1900", "bad-sex", "active-2",
        "active-text", "leading-space", "newline", "empty-name", "long-first", "long-last", "null-name"])
def test_database_refuses_bad_rows(repo, db, values, message):
    """Raw INSERTs skip every Python check. Each must still be refused, by the right rule."""
    with pytest.raises(sqlite3.IntegrityError, match=message):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, values)


def test_sql_sex_list_is_exactly_the_enum(db, repo):
    """SQL must accept every Sex value AND nothing else (it is generated from the enum)."""
    with db.transaction() as conn:
        for i, sex in enumerate(Sex):  # every enum value is accepted...
            conn.execute(RAW_INSERT, (f"AAA000{i}", "A", f"T{i}", "2010-01-01", sex.value, 1))
    with pytest.raises(sqlite3.IntegrityError, match="sex"):  # ...and an extra value is not
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, ("AAA0009", "A", "Z", "2010-01-01", "Unknown", 1))


def test_database_and_python_agree_on_today(repo, db):
    """Both use LOCAL time. Born today is refused; born yesterday is allowed at database level."""
    today = date.today()
    with pytest.raises(sqlite3.IntegrityError, match="in the past"):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, ("AAA0001", "Sam", "Lee", today.isoformat(), "Male", 1))
    with db.transaction() as conn:
        conn.execute(RAW_INSERT, ("AAA0002", "Tom", "Ng", (today - timedelta(days=1)).isoformat(), "Male", 1))


# ==== 5. Table upkeep ======================================================================
@pytest.mark.parametrize("column, value", [
    ("student_id", "BBB0001"), ("first_name", "Tom"), ("last_name", "Ng"),
    ("dob", "2011-01-01"), ("sex", "Female"), ("created_at", "2000-01-01T00:00:00Z"),
])
def test_every_locked_column_is_locked(db, sam, column, value):
    """Every column the trigger claims to lock must really refuse to change."""
    with pytest.raises(sqlite3.DatabaseError, match="locked"):
        with db.transaction() as conn:
            conn.execute(f"UPDATE students SET {column} = ?", (value,))


def test_students_cannot_be_deleted(db, sam):
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        with db.transaction() as conn:
            conn.execute("DELETE FROM students")


def test_table_creation_is_repeatable_and_keeps_data(db, sam):
    """Building the repository again must not fail AND must not wipe what is already saved."""
    StudentRepository(db)
    StudentRepository(db)
    assert len(StudentRepository(db).all()) == 1


def test_triggers_are_rebuilt_on_startup(db, repo):
    """An out-of-date trigger is replaced at start-up (CREATE ... IF NOT EXISTS would keep it)."""
    with db.transaction() as conn:  # pretend an old version that never fires is stored in the file
        conn.execute("DROP TRIGGER students_dob_in_past")
        conn.execute("CREATE TRIGGER students_dob_in_past BEFORE INSERT ON students "
                     "WHEN 0 BEGIN SELECT 1; END")
    StudentRepository(db)  # start-up runs again
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with pytest.raises(sqlite3.IntegrityError, match="in the past"):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, ("AAA0001", "Sam", "Lee", tomorrow, "Male", 1))