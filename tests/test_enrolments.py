"""Tries to break the enrolments table. Run with: uv run python -m pytest

Sections:
    1. Enrolment: cleaning and validation (Python side)
    2. Repository: saving, loading, filtering, missing things
    3. Business rules (age/grade, one grade per season, no going backwards, active students)
    4. set_grade and delete
    5. The database itself refuses bad data, even if Python is bypassed
    6. Dataframes
    7. Table upkeep: re-running, lookup tables, rebuilt triggers, old tables

Every test says what it protects. `match=` proves the RIGHT rule fired. Where two rules would
both be broken by the same row (e.g. a year far in the past breaks the age AND other rules),
SQLite does not promise which message wins, so those cases check only that it is refused.
"""

import sqlite3
from datetime import date
from typing import Any

import polars as pl
import pytest

import whc.students as students_module
from whc.database import Database
from whc.enrolments import Enrolment, EnrolmentRepository, Grade, Sport
from whc.exceptions import DuplicateError, NotFoundError, ValidationError
from whc.students import Student, StudentRepository

STUDENT: dict[str, Any] = dict(
    student_id="AAA0001", first_name="Sam", last_name="Lee", dob="14/03/2010", sex="Male"
)
# Sam, born 2010, is 15 in the 2025 season, which suits Year 8 (allowed ages for Year 8: 11-16).
GOOD: dict[str, Any] = dict(
    student_id="AAA0001", season_year=2025, sport="Badminton", grade="Year 8"
)
RAW_INSERT = "INSERT INTO enrolments (student_id, season_year, sport, grade) VALUES (?, ?, ?, ?)"


# ---- Fixtures -----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def frozen_today(monkeypatch):
    """Freeze the STUDENT age rules at 5 Nov 2026, so the test students never 'age out'.

    (Enrolment season rules use the real clock, like the database trigger does.)
    """
    monkeypatch.setattr(students_module, "_today", lambda: date(2026, 11, 5))


@pytest.fixture
def db():
    with Database(":memory:") as database:
        StudentRepository(database)  # parent table first
        EnrolmentRepository(database)  # child table second (it would also create the parent)
        yield database


@pytest.fixture
def students(db):
    return StudentRepository(db)


@pytest.fixture
def repo(db):
    return EnrolmentRepository(db)


@pytest.fixture
def student(students):
    """Sam (AAA0001) already saved."""
    students.add(Student.new(**STUDENT))


# ==== 1. Enrolment: cleaning and validation ===============================================
def test_new_cleans_input():
    """new() trims/uppercases the ID and matches sport and grade ignoring capitals and spaces."""
    e = Enrolment.new(" aaa0002 ", 2025, " cross  COUNTRY ", " year   9 ")
    assert (e.student_id, e.season_year) == ("AAA0002", 2025)
    assert (e.sport, e.grade) == (Sport.CROSS_COUNTRY, Grade.YEAR_9)
    assert e.enrolment_id is None


@pytest.mark.parametrize("change, message", [
    (dict(student_id="AA1234"), "student_id must be"),
    (dict(student_id="   "), "student_id must be"),
    (dict(student_id="AAA001"), "student_id must be"),
    (dict(season_year=1999), "season_year must be between"),
    (dict(season_year=date.today().year + 2), "season_year must be between"),
    (dict(season_year="2025"), "whole number"),
    (dict(season_year=2025.0), "whole number"),
    (dict(season_year=True), "whole number"),   # True counts as 1 in Python
    (dict(season_year=None), "whole number"),
    (dict(sport="Bball"), "sport must be one of"),
    (dict(sport="   "), "sport must be one of"),
    (dict(sport=None), "sport must be one of"),
    (dict(grade="Year 13"), "grade must be one of"),
    (dict(grade="   "), "grade must be one of"),
    (dict(grade=None), "grade must be one of"),
])
def test_bad_input_refused(change, message):
    """One bad field at a time; each must fail for ITS OWN reason."""
    with pytest.raises(ValidationError, match=message):
        Enrolment.new(**{**GOOD, **change})


def test_none_student_id_raises_validation_error():
    """A blank form field arrives as None; that must be our error, not AttributeError."""
    with pytest.raises(ValidationError, match="student_id"):
        Enrolment.new(**{**GOOD, "student_id": None})


@pytest.mark.parametrize("student_id", ["aaa0001", "AAA0001\n"])
def test_directly_built_invalid_enrolment_refused(student_id):
    """Building Enrolment(...) directly skips new(); __post_init__ must still refuse.

    The newline case is a regression: '$' in a regex used to let a trailing newline through.
    """
    with pytest.raises(ValidationError, match="student_id"):
        Enrolment(None, student_id, 2025, Sport.BADMINTON, Grade.YEAR_8)


@pytest.mark.parametrize("bad_id", [0, -1, True, "7"])
def test_enrolment_id_must_be_a_positive_int(bad_id):
    with pytest.raises(ValidationError, match="enrolment_id"):
        Enrolment(bad_id, "AAA0001", 2025, Sport.BADMINTON, Grade.YEAR_8)


def test_enrolment_id_cannot_be_supplied_to_add(repo, student):
    """The database owns IDs: an enrolment that already has one cannot be added."""
    with pytest.raises(ValidationError, match="assigned by the database"):
        repo.add(Enrolment(99, "AAA0001", 2025, Sport.BADMINTON, Grade.YEAR_8))


def test_grade_levels_are_numbers():
    """Grade.level feeds the 'grade never goes backwards' rule, so it must be right."""
    assert [g.level for g in Grade] == [7, 8, 9, 10, 11, 12]


# ==== 2. Repository =======================================================================
def test_add_assigns_id_and_get_round_trips(repo, student):
    saved = repo.add(Enrolment.new(**GOOD))
    assert saved.enrolment_id is not None
    assert repo.get(saved.enrolment_id) == saved


def test_id_is_never_reused(repo, student):
    """AUTOINCREMENT: after a delete the next ID is still bigger. Results will point at these IDs."""
    first = repo.add(Enrolment.new(**GOOD))
    first_id = first.enrolment_id
    assert first_id is not None
    repo.delete(first_id)
    second = repo.add(Enrolment.new(**GOOD))
    assert second.enrolment_id is not None and second.enrolment_id > first_id


def test_duplicate_enrolment_refused(repo, student):
    """Same student, year AND sport again (even with another grade) is a duplicate."""
    repo.add(Enrolment.new(**GOOD))
    with pytest.raises(DuplicateError, match="already enrolled"):
        repo.add(Enrolment.new(**{**GOOD, "grade": "Year 9"}))


def test_same_sport_next_year_allowed(repo, student):
    """Proves the unique index is not too strict."""
    repo.add(Enrolment.new(**GOOD))
    repo.add(Enrolment.new(**{**GOOD, "season_year": 2026, "grade": "Year 9"}))


def test_second_sport_same_year_allowed(repo, student):
    """Proves the unique index is not too strict (same grade, different sport)."""
    repo.add(Enrolment.new(**GOOD))
    repo.add(Enrolment.new(**{**GOOD, "sport": "Athletics"}))


def test_missing_student_refused(repo):
    """The foreign key is on, and the error says the STUDENT is missing."""
    with pytest.raises(NotFoundError, match="student_id ZZZ9999 does not exist"):
        repo.add(Enrolment.new("ZZZ9999", 2025, "Badminton", "Year 8"))


def test_enrolments_repository_alone_creates_the_students_table():
    """Regression: building EnrolmentRepository first used to fail later with 'no such table'."""
    with Database(":memory:") as database:
        repo = EnrolmentRepository(database)  # note: no StudentRepository created by us
        with pytest.raises(NotFoundError):  # a clean 'student does not exist', not an SQL crash
            repo.add(Enrolment.new(**GOOD))


@pytest.fixture
def two_students(students, student):
    """Sam (born 2010) and Tom (born 2011)."""
    students.add(Student.new("AAA0002", "Tom", "Ng", "14/03/2011", "Male"))


def test_filters_by_student_and_season(repo, two_students):
    """Needs TWO students: with one student, a missing student filter would still pass."""
    repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 8"))
    repo.add(Enrolment.new("AAA0001", 2026, "Badminton", "Year 9"))
    repo.add(Enrolment.new("AAA0001", 2026, "Athletics", "Year 9"))
    repo.add(Enrolment.new("AAA0002", 2025, "Badminton", "Year 7"))
    repo.add(Enrolment.new("AAA0002", 2026, "Athletics", "Year 8"))

    sam = repo.for_student("aaa0001")  # lowercase on purpose
    assert [e.season_year for e in sam] == [2025, 2026, 2026]
    assert {e.student_id for e in sam} == {"AAA0001"}  # no Tom leaked in
    assert [e.sport.value for e in repo.for_student("AAA0002", season_year=2026)] == ["Athletics"]
    assert len(repo.all()) == 5
    assert len(repo.all(season_year=2026)) == 3
    assert len(repo.to_dataframe(season_year=2025)) == 2


def test_missing_ids_raise_not_found(repo):
    with pytest.raises(NotFoundError):
        repo.get(999)
    with pytest.raises(NotFoundError):
        repo.delete(999)
    with pytest.raises(NotFoundError):
        repo.set_grade(999, "Year 9")


@pytest.mark.parametrize("bad_id", ["abc", None, True, 1.5])
def test_id_arguments_must_be_whole_numbers(repo, bad_id):
    with pytest.raises(ValidationError, match="whole number"):
        repo.get(bad_id)


# ==== 3. Business rules ====================================================================
# Allowed "season age" for Year N is N+3 ... N+8, where season age = season_year - birth_year.
# For Year 8 in 2025 that is birth years 2009 (age 16) down to 2014 (age 11).
@pytest.mark.parametrize("birth_year, accepted", [
    (2014, True),   # season age 11 = Year 8 + 3 -> youngest allowed
    (2015, False),  # season age 10 -> too young for Year 8
    (2009, True),   # season age 16 = Year 8 + 8 -> oldest allowed
    (2008, False),  # season age 17 -> too old for Year 8
])
def test_grade_age_band_edges(repo, students, birth_year, accepted):
    """The exact edges of the grade/age rule (AGE_OFFSET_MIN and AGE_OFFSET_MAX)."""
    students.add(Student.new("BBB0001", "Ann", "Kim", f"01/06/{birth_year}", "Female"))
    enrolment = Enrolment.new("BBB0001", 2025, "Badminton", "Year 8")
    if accepted:
        repo.add(enrolment)
    else:
        with pytest.raises(ValidationError, match="does not fit the student age"):
            repo.add(enrolment)


def test_grade_must_be_plausible_for_age(repo, student):
    """Born 2010: Year 12 in 2011 would be a 1-year-old in Year 12; Year 8 in 2023 (13) is fine."""
    with pytest.raises(ValidationError, match="does not fit the student age"):
        repo.add(Enrolment.new("AAA0001", 2011, "Badminton", "Year 12"))
    repo.add(Enrolment.new("AAA0001", 2023, "Badminton", "Year 8"))


def test_cannot_enrol_in_birth_year(repo, student):
    """The old rule only refused years BEFORE birth, so an age-0 student could be enrolled."""
    with pytest.raises(ValidationError, match="does not fit the student age"):
        repo.add(Enrolment.new("AAA0001", 2010, "Badminton", "Year 7"))


def test_one_grade_per_student_per_season(repo, student):
    """Badminton in Year 8 and Athletics in Year 9 in the same season is impossible."""
    repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 8"))
    with pytest.raises(ValidationError, match="different grade in this season"):
        repo.add(Enrolment.new("AAA0001", 2025, "Athletics", "Year 9"))


def test_other_students_may_have_other_grades_in_the_same_season(repo, two_students):
    """The one-grade rule is per student, not per season."""
    repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 8"))
    repo.add(Enrolment.new("AAA0002", 2025, "Badminton", "Year 7"))


def test_grade_cannot_go_backwards(repo, student):
    """Year 10 in 2024 then Year 7 in 2025 (both fit Sam's age, so only this rule can fire)."""
    repo.add(Enrolment.new("AAA0001", 2024, "Badminton", "Year 10"))
    with pytest.raises(ValidationError, match="backwards"):
        repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 7"))


def test_grade_cannot_go_backwards_when_back_filling(repo, student):
    """Adding an EARLIER season later must be checked too: Year 8 in 2023 after Year 10 in 2024."""
    repo.add(Enrolment.new("AAA0001", 2024, "Badminton", "Year 10"))
    with pytest.raises(ValidationError, match="backwards"):
        repo.add(Enrolment.new("AAA0001", 2023, "Badminton", "Year 11"))


def test_repeating_a_year_is_allowed(repo, student):
    """Same grade two seasons running is a repeated year, not 'backwards'."""
    repo.add(Enrolment.new("AAA0001", 2024, "Badminton", "Year 9"))
    repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 9"))


def test_inactive_student_cannot_get_new_enrolments(students, repo, student):
    students.set_active("AAA0001", False)
    with pytest.raises(ValidationError, match="not active"):
        repo.add(Enrolment.new(**GOOD))


def test_reactivated_student_can_enrol_again(students, repo, student):
    students.set_active("AAA0001", False)
    students.set_active("AAA0001", True)
    repo.add(Enrolment.new(**GOOD))


def test_inactive_student_keeps_old_enrolments(students, repo, student):
    """Deactivating hides nothing: history stays readable."""
    repo.add(Enrolment.new(**GOOD))
    students.set_active("AAA0001", False)
    assert len(repo.for_student("AAA0001")) == 1


# ==== 4. set_grade and delete ===============================================================
def test_set_grade_corrects_the_whole_season(repo, student):
    """A grade belongs to the student's SEASON, so all that season's enrolments change together."""
    a = repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 8"))
    b = repo.add(Enrolment.new("AAA0001", 2025, "Athletics", "Year 8"))
    assert a.enrolment_id is not None and b.enrolment_id is not None
    repo.set_grade(a.enrolment_id, "Year 9")
    assert repo.get(a.enrolment_id).grade is Grade.YEAR_9
    assert repo.get(b.enrolment_id).grade is Grade.YEAR_9  # no season is ever left with two grades


def test_set_grade_leaves_other_seasons_and_students_alone(repo, two_students):
    """If the WHERE clause were lost, every row's grade would change."""
    a = repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 8"))
    b = repo.add(Enrolment.new("AAA0001", 2026, "Badminton", "Year 9"))
    c = repo.add(Enrolment.new("AAA0002", 2025, "Badminton", "Year 8"))
    assert a.enrolment_id and b.enrolment_id and c.enrolment_id
    repo.set_grade(a.enrolment_id, "Year 7")
    assert repo.get(a.enrolment_id).grade is Grade.YEAR_7
    assert repo.get(b.enrolment_id).grade is Grade.YEAR_9
    assert repo.get(c.enrolment_id).grade is Grade.YEAR_8


def test_set_grade_accepts_a_grade_value(repo, student):
    saved = repo.add(Enrolment.new(**GOOD))
    assert saved.enrolment_id is not None
    repo.set_grade(saved.enrolment_id, Grade.YEAR_9)
    assert repo.get(saved.enrolment_id).grade is Grade.YEAR_9


def test_set_grade_with_a_bad_grade_is_refused_before_any_lookup(repo):
    with pytest.raises(ValidationError, match="grade must be one of"):
        repo.set_grade(999, "Year 13")


def test_set_grade_still_obeys_the_age_rule(repo, students):
    """Ann (born 2014) is 11 in 2025: fine for Year 8, but too young for Year 9 (needs 12-17)."""
    students.add(Student.new("BBB0001", "Ann", "Kim", "01/06/2014", "Female"))
    saved = repo.add(Enrolment.new("BBB0001", 2025, "Badminton", "Year 8"))
    assert saved.enrolment_id is not None
    with pytest.raises(ValidationError, match="does not fit the student age"):
        repo.set_grade(saved.enrolment_id, "Year 9")
    assert repo.get(saved.enrolment_id).grade is Grade.YEAR_8  # unchanged


def test_set_grade_cannot_go_backwards(repo, student):
    repo.add(Enrolment.new("AAA0001", 2024, "Badminton", "Year 9"))
    later = repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 9"))
    assert later.enrolment_id is not None
    with pytest.raises(ValidationError, match="backwards"):
        repo.set_grade(later.enrolment_id, "Year 7")  # fits the age, but is lower than 2024's Year 9


def test_failed_set_grade_changes_nothing(repo, student):
    """The update is all-or-nothing: a refused grade must leave BOTH enrolments untouched."""
    a = repo.add(Enrolment.new("AAA0001", 2024, "Badminton", "Year 9"))
    b = repo.add(Enrolment.new("AAA0001", 2025, "Badminton", "Year 9"))
    c = repo.add(Enrolment.new("AAA0001", 2025, "Athletics", "Year 9"))
    assert b.enrolment_id and c.enrolment_id and a.enrolment_id
    with pytest.raises(ValidationError):
        repo.set_grade(b.enrolment_id, "Year 7")
    assert repo.get(b.enrolment_id).grade is Grade.YEAR_9
    assert repo.get(c.enrolment_id).grade is Grade.YEAR_9


def test_delete_allowed_while_unreferenced(repo, student):
    saved = repo.add(Enrolment.new(**GOOD))
    assert saved.enrolment_id is not None
    repo.delete(saved.enrolment_id)
    with pytest.raises(NotFoundError):
        repo.get(saved.enrolment_id)


# ==== 5. The database itself refuses bad data, even if Python is bypassed ==================
@pytest.mark.parametrize("values, message", [
    (("AAA0001", 2025, "Bball", "Year 8"), "FOREIGN KEY"),         # sport not in lookup table
    (("AAA0001", 2025, "Badminton", "Year 13"), "FOREIGN KEY"),    # grade not in lookup table
    (("ZZZ9999", 2025, "Badminton", "Year 8"), "FOREIGN KEY"),     # missing student
    ((None, 2025, "Badminton", "Year 8"), "NOT NULL"),             # required field
    (("AAA0001", "abc", "Badminton", "Year 8"), "season_year"),    # STRICT: wrong type
    # Two rules break at once in the next three rows, so only 'refused' is checked:
    (("AAA0001", 2099, "Badminton", "Year 8"), None),              # far future (and too old an age)
    (("AAA0001", 2005, "Badminton", "Year 8"), None),              # before the student was born
    (("AAA0001", 1999, "Badminton", "Year 8"), None),              # before MIN_SEASON_YEAR
], ids=["bad-sport", "bad-grade", "missing-student", "null", "text-year", "future", "pre-birth", "pre-2000"])
def test_database_refuses_bad_rows(db, student, values, message):
    with pytest.raises(sqlite3.IntegrityError, match=message):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, values)


def test_database_refuses_future_season_on_its_own(db):
    """Isolates the 'future' trigger: the grade FITS the age, so only the year is wrong.

    The student row is inserted raw so the test never depends on today's entry-age rule.
    """
    year = date.today().year + 2  # one year beyond MAX_YEARS_AHEAD
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO students (student_id, first_name, last_name, dob, sex) "
            "VALUES ('CCC0001', 'Zed', 'Ray', ?, 'Male')", (f"{year - 14}-06-01",))
    with pytest.raises(sqlite3.IntegrityError, match="future"):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, ("CCC0001", year, "Badminton", "Year 9"))  # age 14 fits Year 9


@pytest.mark.parametrize("column, value", [
    ("enrolment_id", 99), ("student_id", "BBB0001"), ("season_year", 2024),
    ("sport", "Athletics"), ("created_at", "2000-01-01T00:00:00Z"),
])
def test_every_locked_column_is_locked(repo, db, student, column, value):
    """Every column the trigger claims to lock must really refuse to change."""
    repo.add(Enrolment.new(**GOOD))
    with pytest.raises(sqlite3.DatabaseError, match="locked"):
        with db.transaction() as conn:
            conn.execute(f"UPDATE enrolments SET {column} = ?", (value,))


# ==== 6. Dataframes =========================================================================
def test_dataframe_types_even_when_empty(repo):
    schema = repo.to_dataframe().schema
    assert schema["season_year"] == pl.Int64
    assert isinstance(schema["sport"], pl.Enum)
    assert isinstance(schema["grade"], pl.Enum)


def test_dataframe_uses_enum_values(repo, student):
    repo.add(Enrolment.new(**GOOD))
    df = repo.to_dataframe()
    assert df["sport"].to_list() == ["Badminton"]
    assert df["grade"].to_list() == ["Year 8"]
    assert isinstance(df.schema["sport"], pl.Enum)
    assert isinstance(df.schema["grade"], pl.Enum)


# ==== 7. Table upkeep =======================================================================
def test_table_creation_is_repeatable_and_keeps_data(db, repo, student):
    """Building the repository again must not fail AND must not wipe what is already saved."""
    repo.add(Enrolment.new(**GOOD))
    EnrolmentRepository(db)
    EnrolmentRepository(db)
    assert len(EnrolmentRepository(db).all()) == 1


def test_every_sport_round_trips(repo, students, student):
    """Every Sport can be saved and loaded (same student, season and grade, 13 sports)."""
    for sport in Sport:
        repo.add(Enrolment.new("AAA0001", 2025, sport.value, "Year 8"))
    assert [e.sport for e in repo.all()] == sorted(Sport, key=lambda s: s.value)


def test_every_grade_round_trips(repo, student):
    """Every Grade can be saved: Sam goes Year 7 -> 12 across 2021-2026 (a realistic path)."""
    for grade, year in zip(Grade, range(2021, 2027)):
        repo.add(Enrolment.new("AAA0001", year, "Badminton", grade.value))
    assert [e.grade for e in repo.for_student("AAA0001")] == list(Grade)


def test_lookup_tables_match_the_enums(db, repo):
    """The database holds exactly the enum's sports and grades (with their levels)."""
    sports = {row[0] for row in db.query("SELECT name FROM sports")}
    grades = dict(db.query("SELECT name, level FROM grades"))
    assert sports == {s.value for s in Sport}
    assert grades == {g.value: g.level for g in Grade}


def test_a_sport_missing_from_an_old_database_is_added_on_startup(db, repo, student):
    """The reason lookup tables exist: an older file that lacks a sport picks it up on start-up."""
    with db.transaction() as conn:
        conn.execute("DELETE FROM sports WHERE name = 'Tennis'")  # pretend an old file
    EnrolmentRepository(db)  # start-up runs again
    repo.add(Enrolment.new("AAA0001", 2025, "Tennis", "Year 8"))  # would fail without the refill


def test_triggers_are_rebuilt_on_startup(db, repo, students, student):
    """An out-of-date rule is replaced at start-up (CREATE ... IF NOT EXISTS would keep it)."""
    students.set_active("AAA0001", False)
    with db.transaction() as conn:  # pretend an old version that never fires is stored in the file
        conn.execute("DROP TRIGGER enrolments_student_active")
        conn.execute("CREATE TRIGGER enrolments_student_active BEFORE INSERT ON enrolments "
                     "WHEN 0 BEGIN SELECT 1; END")
    EnrolmentRepository(db)  # start-up runs again
    with pytest.raises(sqlite3.IntegrityError, match="not active"):
        with db.transaction() as conn:
            conn.execute(RAW_INSERT, ("AAA0001", 2025, "Badminton", "Year 8"))


def test_an_old_enrolments_table_gives_a_clear_error():
    """A whc.db from before the lookup tables cannot be upgraded silently; say so plainly."""
    with Database(":memory:") as database:
        StudentRepository(database)
        with database.transaction() as conn:
            conn.execute(
                "CREATE TABLE enrolments (enrolment_id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "student_id TEXT NOT NULL REFERENCES students(student_id), "
                "season_year INTEGER NOT NULL, sport TEXT NOT NULL, grade TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT 'x')"
            )
        with pytest.raises(RuntimeError, match="old enrolments table"):
            EnrolmentRepository(database)