"""cleaner.py - Phase 1b: STRICT zero-assumption cleaning of raw_test/ssp_data.xlsx.

Core Engineering Principle:
    * NEVER modify the raw workbook (read-only).
    * ZERO assumptions, ZERO guessing, ZERO inference, ZERO placeholder dates, ZERO merging.
    * If a row has EVEN ONE malformed, missing, ambiguous, or out-of-range field,
      REJECT THE ENTIRE ROW and record the exact data-entry mistake so our future
      dashboard UI can make that mistake impossible.
    * Retain ONLY 100% spotless, validated records.

Outputs written to `data/cleaned/` (gitignored):
    1. students_clean.csv           100% valid legacy students (dob blank — no assumed DOBs)
    2. enrolments_clean.csv         100% valid enrolments for spotless records
    3. test_results_clean.csv       100% spotless long-format legacy test results
                                    (season_year + semester only — NO fabricated dates!)
    4. rejected_data_lessons.csv    Every rejected row/item with the mistake category and the
                                    future dashboard safeguard that prevents it
"""

from __future__ import annotations

import csv
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from whc.enrolments import Enrolment, Sport
from whc.exceptions import ValidationError
from whc.fitness_tests import DEFAULT_LEVEL_SHUTTLE_RULES, DEFAULT_TESTS, FitnessTest, Unit
from whc.students import ID_PATTERN, Sex, Student
from whc.test_results import LEVEL_MAX, LEVEL_MIN, SHUTTLE_MAX, SHUTTLE_MIN, TestResult

_XLSX_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

# Map SSP AMS Database column letters -> FitnessTest.test_id
COLUMN_TO_TEST_ID: dict[str, int] = {
    "H": 902,   # Weight (Kilograms)
    "I": 901,   # Height (Centimetres)
    "J": 101,   # Counter Movement Jump Height (Centimetres)
    "K": 102,   # Counter Movement Jump Power (Watts/Kilogram)
    "L": 103,   # Counter Movement Jump Relative Strength Index (RSI) (Unitless)
    "M": 104,   # Hop Jump Relative Strength Index (RSI) (Unitless)
    "N": 2,     # Isometric Mid-Thigh Pull Relative (Unitless)
    "O": 1,     # Isometric Mid-Thigh Pull Peak (Newtons)
    "P": 401,   # YOYO Intermittent Recovery Test 1 (IRT1) (Level:Shuttle)
    "Q": 201,   # 5 Metre Sprint (Seconds)
    "R": 202,   # 10 Metre Sprint (Seconds)
    "S": 203,   # 20 Metre Sprint (Seconds)
    "T": 301,   # 5-0-5 Agility Test Left (Seconds)
    "U": 302,   # 5-0-5 Agility Test Right (Seconds)
    "V": 303,   # NBA Lane Agility Test (Seconds)
    "W": 3,     # Nordic Hamstring Left (Newtons)
    "X": 4,     # Nordic Hamstring Right (Newtons)
    "Y": 5,     # Adduction Left (Newtons)
    "Z": 6,     # Adduction Right (Newtons)
    "AA": 13,   # 3RM Bench Press (Kilograms)
    "AB": 14,   # 3RM Trap Bar Deadlift (Kilograms)
}

TESTS_BY_ID: dict[int, FitnessTest] = {t.test_id: t for t in DEFAULT_TESTS}

# Schema label mapping (the entire Basketball cohort in the workbook uses the header 'Basket Ball')
SPORT_SCHEMA_MAP: dict[str, str] = {
    "basket ball": "Basketball",
}

ROSTER_YEAR = 2026

_PLACEHOLDER_NAMES = {"", "student id code not entered", "#n/a", "0"}
_LEVEL_SHUTTLE_RE = re.compile(r"^\s*(\d+)\s*[:.]\s*(\d+)\s*$")

YOYO_IRT1_MAX_SHUTTLES: dict[int, int] = DEFAULT_LEVEL_SHUTTLE_RULES[401]
IMTP_REL_MAX_TOLERANCE = 0.15  # strict 15% filter for cleaning legacy raw data


# ---- Output records ----------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CleanStudentRow:
    student_id: str
    first_name: str
    last_name: str
    dob: str
    sex: str
    is_active: bool


@dataclass(frozen=True, slots=True)
class CleanEnrolmentRow:
    student_id: str
    season_year: int
    sport: str
    grade: str


@dataclass(frozen=True, slots=True)
class CleanTestResultRow:
    student_id: str
    season_year: int
    semester: int
    test_id: int
    test_name: str
    value: float | None
    level: int | None
    shuttle: int | None
    source_row: int


@dataclass(frozen=True, slots=True)
class RejectedRecord:
    sheet: str
    excel_row: int
    student_id: str
    mistake_category: str
    raw_detail: str
    reason: str
    future_safeguard: str


@dataclass(frozen=True, slots=True)
class CleaningReport:
    students: tuple[CleanStudentRow, ...]
    enrolments: tuple[CleanEnrolmentRow, ...]
    test_results: tuple[CleanTestResultRow, ...]
    rejected: tuple[RejectedRecord, ...]


# ---- Pure helper functions ---------------------------------------------------------
def _col_to_idx(col_letters: str) -> int:
    idx = 0
    for ch in col_letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _read_xlsx_sheets(path: Path) -> dict[str, dict[int, dict[str, str]]]:
    """Read an .xlsx workbook strictly read-only using Python's standard library."""
    with zipfile.ZipFile(path, "r") as zf:
        wb_xml = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_xml}

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_xml = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_xml.findall("s:si", _XLSX_NS):
                text = "".join(t.text or "" for t in si.iter(f"{{{_XLSX_NS['s']}}}t"))
                shared_strings.append(text)

        sheets: dict[str, dict[int, dict[str, str]]] = {}
        for sh in wb_xml.findall(".//s:sheet", _XLSX_NS):
            name = sh.attrib["name"]
            target = rel_map[sh.attrib[_REL_ATTR]]
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            sh_xml = ET.fromstring(zf.read(target))
            rows: dict[int, dict[str, str]] = {}
            for row_el in sh_xml.findall(".//s:sheetData/s:row", _XLSX_NS):
                rnum = int(row_el.attrib["r"])
                row_data: dict[str, str] = {}
                for cell in row_el.findall("s:c", _XLSX_NS):
                    m = re.match(r"([A-Z]+)", cell.attrib["r"])
                    if not m:
                        continue
                    col = m.group(1)
                    ctype = cell.attrib.get("t")
                    v_el = cell.find("s:v", _XLSX_NS)
                    is_el = cell.find("s:is", _XLSX_NS)
                    val: str | None = None
                    if ctype == "s" and v_el is not None and v_el.text is not None:
                        val = shared_strings[int(v_el.text)]
                    elif ctype == "inlineStr" and is_el is not None:
                        val = "".join(t.text or "" for t in is_el.iter(f"{{{_XLSX_NS['s']}}}t"))
                    elif v_el is not None and v_el.text is not None:
                        val = v_el.text
                    if val is not None:
                        row_data[col] = val
                if row_data:
                    rows[rnum] = row_data
            sheets[name] = rows
        return sheets


def repair_mojibake(text: str) -> str:
    """Lossless byte-level repair of UTF-8 text accidentally decoded as Latin-1."""
    try:
        repaired = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return unicodedata.normalize("NFC", repaired) if repaired != text else text


def _parse_int_cell(raw: str) -> int | None:
    text = raw.strip()
    if not text:
        return None
    try:
        num = float(text)
    except ValueError:
        return None
    return int(num) if num.is_integer() else None


def _normalize_sport(raw: str) -> str | None:
    cleaned = " ".join(raw.split())
    if not cleaned:
        return None
    lower = cleaned.casefold()
    if lower in SPORT_SCHEMA_MAP:
        return SPORT_SCHEMA_MAP[lower]
    for s in Sport:
        if s.value.casefold() == lower:
            return s.value
    return None


def _normalize_sex(raw: str) -> str | None:
    cleaned = " ".join(raw.split())
    if not cleaned:
        return None
    for s in Sex:
        if s.value.casefold() == cleaned.casefold():
            return s.value
    return None


def _format_surname(last_name: str) -> str:
    cleaned = " ".join(last_name.split())
    if not cleaned.isupper():
        return cleaned

    def _cap_word(word: str) -> str:
        parts = re.split(r"([-'])", word)
        out: list[str] = []
        for p in parts:
            if p in ("-", "'"):
                out.append(p)
            elif p.startswith("MC") and len(p) > 2:
                out.append("Mc" + p[2:].capitalize())
            else:
                out.append(p.capitalize())
        return "".join(out)

    return " ".join(_cap_word(w) for w in cleaned.split(" "))


def _parse_strict_test_cell(
    raw: str, test: FitnessTest
) -> tuple[tuple[float | None, int | None, int | None] | None, str | None]:
    """Strictly parse and validate ONE test cell — NO comma-fixing, NO guessing."""
    text = raw.strip()
    if test.unit == Unit.LEVEL_SHUTTLE.value:
        m = _LEVEL_SHUTTLE_RE.fullmatch(text)
        if not m:
            return None, f"{test.test_name}: '{raw}' is not valid level.shuttle / level:shuttle format"
        lvl, sh = int(m.group(1)), int(m.group(2))
        if not (LEVEL_MIN <= lvl <= LEVEL_MAX and SHUTTLE_MIN <= sh <= SHUTTLE_MAX):
            return (
                None,
                f"{test.test_name}: {lvl}:{sh} outside allowed range ({LEVEL_MIN}-{LEVEL_MAX}:{SHUTTLE_MIN}-{SHUTTLE_MAX})",
            )
        if test.test_id == 401:
            max_sh = YOYO_IRT1_MAX_SHUTTLES.get(lvl)
            if max_sh is None or sh > max_sh:
                return (
                    None,
                    f"{test.test_name}: {lvl}.{sh} is impossible in official Yo-Yo IRT1 protocol "
                    f"(level {lvl} max shuttles = {max_sh if max_sh is not None else 'invalid level'})",
                )
        return (None, lvl, sh), None

    try:
        num = float(text)
    except ValueError:
        return None, f"{test.test_name}: '{raw}' is not a valid decimal number"

    try:
        test.check_value(num)
    except ValidationError as exc:
        return None, str(exc)

    return (round(num, 4), None, None), None


# ---- Strict zero-assumption cleaning pipeline --------------------------------------
def clean_workbook(xlsx_path: Path | str) -> CleaningReport:
    """Strictly clean raw_test/ssp_data.xlsx with ZERO assumptions and ZERO fabricated dates.

    Retains only 100% spotless records; rejects anything even slightly ambiguous or malformed.
    """
    path = Path(xlsx_path)
    sheets = _read_xlsx_sheets(path)
    rejected: list[RejectedRecord] = []

    # ------------------------------------------------------------------
    # STEP 1: Authoritative "All Students" roster
    # ------------------------------------------------------------------
    students_by_id: dict[str, CleanStudentRow] = {}
    roster_year_level: dict[str, int] = {}

    for rnum, row in sorted(sheets.get("All Students", {}).items()):
        col_a = row.get("A", "").strip()
        col_b = row.get("B", "").strip()
        last_raw = row.get("C", "").strip()
        first_raw = row.get("D", "").strip()
        sex_raw = row.get("E", "").strip()
        yl_raw = row.get("H", "").strip()

        if col_a.upper() == "SUSSI ID" or last_raw.casefold() == "last name":
            rejected.append(
                RejectedRecord(
                    sheet="All Students",
                    excel_row=rnum,
                    student_id="",
                    mistake_category="misplaced_header_row",
                    raw_detail=col_a,
                    reason="Header row was sorted alphabetically into the middle of All Students",
                    future_safeguard="Database tables have fixed schemas; users can never sort headers into data rows",
                )
            )
            continue

        sid = col_a.upper()
        if not ID_PATTERN.fullmatch(sid):
            rejected.append(
                RejectedRecord(
                    sheet="All Students",
                    excel_row=rnum,
                    student_id=col_a,
                    mistake_category="invalid_student_id",
                    raw_detail=f"A={col_a!r}, B={col_b!r}",
                    reason="Student ID in All Students does not match AAA0000 format",
                    future_safeguard="Student form + SQL CHECK strictly enforce AAA0000 format",
                )
            )
            continue

        if col_b and col_b.upper() != sid:
            rejected.append(
                RejectedRecord(
                    sheet="All Students",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="mismatched_roster_ids",
                    raw_detail=f"SUSSI ID={col_a!r} vs Import Identifier={col_b!r} ({first_raw} {last_raw})",
                    reason="SUSSI ID (Col A) and Import Identifier (Col B) disagree on this roster row",
                    future_safeguard="Single authoritative student_id primary key locked at creation",
                )
            )
            continue

        first_clean = repair_mojibake(first_raw)
        last_clean = _format_surname(repair_mojibake(last_raw))
        sex_clean = _normalize_sex(sex_raw)
        m_yl = re.fullmatch(r"Year\s+(\d+)", yl_raw, flags=re.IGNORECASE)

        if sex_clean is None or not m_yl:
            rejected.append(
                RejectedRecord(
                    sheet="All Students",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="invalid_roster_metadata",
                    raw_detail=f"sex={sex_raw!r}, year_level={yl_raw!r}",
                    reason="Missing or invalid Sex or Year Level on All Students row",
                    future_safeguard="Dropdown-only Sex and Grade selectors with NOT NULL database constraints",
                )
            )
            continue

        try:
            validated = Student.legacy(
                student_id=sid,
                first_name=first_clean,
                last_name=last_clean,
                sex=sex_clean,
                is_active=True,
            )
        except ValidationError as exc:
            rejected.append(
                RejectedRecord(
                    sheet="All Students",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="student_validation_error",
                    raw_detail=f"{first_raw} {last_raw} ({sex_raw})",
                    reason=str(exc),
                    future_safeguard="Strict Student.new + SQLite CHECK validation on every entry",
                )
            )
            continue

        students_by_id[validated.student_id] = CleanStudentRow(
            student_id=validated.student_id,
            first_name=validated.first_name,
            last_name=validated.last_name,
            dob="",
            sex=validated.sex.value,
            is_active=True,
        )
        roster_year_level[validated.student_id] = int(m_yl.group(1))

    # ------------------------------------------------------------------
    # STEP 2: Reject stray scratchpad rows on Sheet4
    # ------------------------------------------------------------------
    for rnum, row in sorted(sheets.get("Sheet4", {}).items()):
        raw_desc = ", ".join(f"{k}={v}" for k, v in sorted(row.items(), key=lambda kv: _col_to_idx(kv[0])))
        rejected.append(
            RejectedRecord(
                sheet="Sheet4",
                excel_row=rnum,
                student_id="",
                mistake_category="stray_scratch_sheet",
                raw_detail=raw_desc,
                reason="Unstructured scratchpad note with no student ID, year, semester, or sport",
                future_safeguard="All data enters structured forms linked to a student & enrolment; no free-form scratch sheets",
            )
        )

    # ------------------------------------------------------------------
    # STEP 3: Strictly screen every row of "SSP AMS Database"
    # ------------------------------------------------------------------
    # Keyed by (student_id, year, semester) to match the normalized 3NF schema:
    # if a student has conflicting/split rows in the same (year, semester), REJECT ALL!
    # session_candidates: (sid, year, sem) -> list[(rnum, sport, grade_int, parsed_scores_dict)]
    session_candidates: dict[
        tuple[str, int, int],
        list[tuple[int, str, int, dict[int, tuple[float | None, int | None, int | None]]]],
    ] = defaultdict(list)

    enrolments_map: dict[tuple[str, int, str], int] = {}

    for rnum, row in sorted(sheets.get("SSP AMS Database", {}).items()):
        if rnum == 1:
            continue

        sid_raw = row.get("A", "").strip()
        name_raw = row.get("B", "").strip()
        grade_raw = row.get("C", "").strip()
        year_raw = row.get("D", "").strip()
        sem_raw = row.get("E", "").strip()
        sex_raw = row.get("F", "").strip()
        sport_raw = row.get("G", "").strip()

        raw_tests = {
            col: row[col].strip()
            for col in COLUMN_TO_TEST_ID
            if col in row and row[col].strip() != ""
        }

        if (
            not sid_raw
            and name_raw.casefold() in _PLACEHOLDER_NAMES
            and not any([grade_raw, year_raw, sem_raw, sex_raw, sport_raw])
            and not raw_tests
        ):
            continue

        # 3a. Strict Student ID format
        sid = sid_raw.upper()
        if not ID_PATTERN.fullmatch(sid):
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid_raw,
                    mistake_category="invalid_or_mangled_student_id",
                    raw_detail=f"sid={sid_raw!r}, name={name_raw!r}, tests={len(raw_tests)}",
                    reason="Student ID is blank, mangled, or auto-converted to an Excel date serial",
                    future_safeguard="Searchable student dropdown bound to student_id — coaches never type raw IDs or trigger Excel date auto-formatting",
                )
            )
            continue

        # 3b. Student MUST exist in the authoritative All Students roster
        if sid not in students_by_id:
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="student_not_in_authoritative_roster",
                    raw_detail=f"name={name_raw!r}, yr={year_raw!r}, sp={sport_raw!r}, tests={len(raw_tests)}",
                    reason="Student ID is not in the authoritative All Students roll (former/graduated or unverified student)",
                    future_safeguard="Foreign key requires a verified Student profile before any enrolment or test result can be saved",
                )
            )
            continue

        # 3c. Sex must not contradict All Students
        if sex_raw:
            sex_norm = _normalize_sex(sex_raw)
            if sex_norm != students_by_id[sid].sex:
                rejected.append(
                    RejectedRecord(
                        sheet="SSP AMS Database",
                        excel_row=rnum,
                        student_id=sid,
                        mistake_category="sex_mismatch",
                        raw_detail=f"SSP sex={sex_raw!r} vs Roster sex={students_by_id[sid].sex!r}",
                        reason="Sex on SSP row contradicts authoritative All Students roster",
                        future_safeguard="Sex is stored once on the Student profile — never re-typed per testing session",
                    )
                )
                continue

        # 3d. Strict required metadata: Year, Grade (7-12), Sport
        grade_int = _parse_int_cell(grade_raw)
        year_int = _parse_int_cell(year_raw)
        sem_int = _parse_int_cell(sem_raw)
        sport_clean = _normalize_sport(sport_raw)

        if year_int is None:
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="missing_year",
                    raw_detail=f"gr={grade_raw!r}, yr={year_raw!r}, sem={sem_raw!r}, sp={sport_raw!r}, tests={len(raw_tests)}",
                    reason="Year column is blank (e.g. forgot to drag-fill formula/value in Excel)",
                    future_safeguard="Required test_date picker (dd/mm/yyyy) — blank years are impossible",
                )
            )
            continue

        if grade_int is None or not (7 <= grade_int <= 12) or sport_clean is None:
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="missing_or_invalid_enrolment_metadata",
                    raw_detail=f"gr={grade_raw!r}, yr={year_raw!r}, sp={sport_raw!r}, tests={len(raw_tests)}",
                    reason="Missing or invalid Grade (7-12) or Sport",
                    future_safeguard="Strict dropdown selectors for Sport and Grade backed by SQLite lookup tables",
                )
            )
            continue

        # 3e. Grade MUST match the authoritative 2026 roster progression
        expected_grade = roster_year_level[sid] - (ROSTER_YEAR - year_int)
        if grade_int != expected_grade:
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="grade_contradicts_roster",
                    raw_detail=f"yr={year_int}, SSP grade={grade_int}, expected Year {expected_grade}, tests={len(raw_tests)}",
                    reason=f"Grade {grade_int} in {year_int} contradicts expected Year {expected_grade} from {ROSTER_YEAR} All Students roll",
                    future_safeguard="Enrolment triggers enforce one grade per season and prevent backwards grade jumps",
                )
            )
            continue

        try:
            Enrolment.new(sid, year_int, sport_clean, f"Year {grade_int}")
        except ValidationError as exc:
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="enrolment_validation_error",
                    raw_detail=f"yr={year_int}, gr={grade_int}, sp={sport_clean}",
                    reason=str(exc),
                    future_safeguard="Enrolment.new + SQLite triggers reject invalid enrolments at entry time",
                )
            )
            continue

        if not raw_tests:
            enrolments_map[(sid, year_int, sport_clean)] = grade_int
            continue

        # 3f. Any row WITH test results MUST have an explicit Semester (1 or 2)
        if sem_int not in (1, 2):
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="missing_semester",
                    raw_detail=f"yr={year_int}, sem={sem_raw!r}, sp={sport_clean}, tests={len(raw_tests)}",
                    reason="Row has test results but Semester is blank or invalid",
                    future_safeguard="Every future test result requires a real test_date (dd/mm/yyyy); Semester is derived automatically",
                )
            )
            continue

        # 3g. ALL-OR-NOTHING row validation
        parsed_scores: dict[int, tuple[float | None, int | None, int | None]] = {}
        row_errors: list[str] = []
        for col, raw_val in raw_tests.items():
            tid = COLUMN_TO_TEST_ID[col]
            score, err = _parse_strict_test_cell(raw_val, TESTS_BY_ID[tid])
            if err is not None:
                row_errors.append(f"{col}='{raw_val}' ({err})")
            else:
                assert score is not None
                parsed_scores[tid] = score

        if row_errors:
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="invalid_or_out_of_range_test_value",
                    raw_detail="; ".join(row_errors),
                    reason=f"Row rejected completely because {len(row_errors)} test cell(s) failed strict range/format checks",
                    future_safeguard="Schema-driven numeric inputs with live min/max & protocol guardrails + SQLite range/protocol triggers refuse typos before save",
                )
            )
            continue

        # 3h. Sports-science cross-test physics checks
        physics_errors: list[str] = []
        if 1 in parsed_scores and 2 in parsed_scores and 902 in parsed_scores:
            peak = parsed_scores[1][0]
            rel = parsed_scores[2][0]
            wt = parsed_scores[902][0]
            assert peak is not None and rel is not None and wt is not None
            calc_rel = peak / (wt * 9.81)
            if abs(calc_rel - rel) / max(rel, 0.01) > IMTP_REL_MAX_TOLERANCE:
                physics_errors.append(
                    f"IMTP Peak={peak:g}N & Weight={wt:g}kg implies Rel={calc_rel:.2f} vs recorded Rel={rel:g}"
                )
        if 201 in parsed_scores and 202 in parsed_scores:
            s5, s10 = parsed_scores[201][0], parsed_scores[202][0]
            assert s5 is not None and s10 is not None
            if s5 >= s10:
                physics_errors.append(f"5m Sprint ({s5:g}s) >= 10m Sprint ({s10:g}s)")
        if 202 in parsed_scores and 203 in parsed_scores:
            s10, s20 = parsed_scores[202][0], parsed_scores[203][0]
            assert s10 is not None and s20 is not None
            if s10 >= s20:
                physics_errors.append(f"10m Sprint ({s10:g}s) >= 20m Sprint ({s20:g}s)")
        if 201 in parsed_scores and 203 in parsed_scores:
            s5, s20 = parsed_scores[201][0], parsed_scores[203][0]
            assert s5 is not None and s20 is not None
            if s5 >= s20:
                physics_errors.append(f"5m Sprint ({s5:g}s) >= 20m Sprint ({s20:g}s)")

        if physics_errors:
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=rnum,
                    student_id=sid,
                    mistake_category="physics_contradiction",
                    raw_detail="; ".join(physics_errors),
                    reason="Row rejected because cross-test scores contradict physical laws (e.g. IMTP Peak vs Relative/Weight)",
                    future_safeguard="SQLite trigger enforces 5m < 10m < 20m sprint splits; UI warns & asks confirmation when IMTP Relative deviates >20%",
                )
            )
            continue

        session_candidates[(sid, year_int, sem_int)].append((rnum, sport_clean, grade_int, parsed_scores))

    # ------------------------------------------------------------------
    # STEP 4: Resolve duplicate sessions per (student_id, year, semester) with ZERO assumptions
    #   - 1 row -> spotless, retain
    #   - Multiple 100% identical score dicts -> exact duplicate, retain 1 copy (and all valid enrolments)
    #   - Multiple NON-IDENTICAL score dicts -> split/conflicting session, REJECT ALL ROWS in group
    # ------------------------------------------------------------------
    clean_results: list[CleanTestResultRow] = []

    for (sid, yr, sem), entries in sorted(session_candidates.items()):
        first_rnum, _first_sp, _first_grade, first_scores = entries[0]

        if len(entries) > 1 and not all(sc == first_scores for _, _, _, sc in entries[1:]):
            row_nums = [r for r, _, _, _ in entries]
            sports_seen = sorted({sp for _, sp, _, _ in entries})
            rejected.append(
                RejectedRecord(
                    sheet="SSP AMS Database",
                    excel_row=row_nums[0],
                    student_id=sid,
                    mistake_category="duplicate_conflicting_or_split_session",
                    raw_detail=f"rows={row_nums}, yr={yr}, sem={sem}, sports={sports_seen}",
                    reason=(
                        f"Multiple non-identical rows ({row_nums}) exist for the same student/year/semester "
                        "without exact testing dates — rejected to avoid guessing or merging"
                    ),
                    future_safeguard="Unique constraint on (student_id, test_id, test_date) with mandatory real testing dates eliminates ambiguous sessions",
                )
            )
            continue

        # Spotless session! Register enrolment(s) & all validated legacy test results (NO fake dates!)
        for _, sp, gr, _ in entries:
            enrolments_map[(sid, yr, sp)] = gr

        for tid, (val, lvl, sh) in sorted(first_scores.items()):
            test = TESTS_BY_ID[tid]
            TestResult.legacy(
                student_id=sid,
                test_id=tid,
                season_year=yr,
                semester=sem,
                value=val,
                level=lvl,
                shuttle=sh,
            )
            clean_results.append(
                CleanTestResultRow(
                    student_id=sid,
                    season_year=yr,
                    semester=sem,
                    test_id=tid,
                    test_name=test.test_name,
                    value=val,
                    level=lvl,
                    shuttle=sh,
                    source_row=first_rnum,
                )
            )

    clean_students = tuple(students_by_id[sid] for sid in sorted(students_by_id))
    clean_enrolments = tuple(
        CleanEnrolmentRow(student_id=sid, season_year=yr, sport=sp, grade=f"Year {gr}")
        for (sid, yr, sp), gr in sorted(enrolments_map.items())
    )

    return CleaningReport(
        students=clean_students,
        enrolments=clean_enrolments,
        test_results=tuple(clean_results),
        rejected=tuple(rejected),
    )


def write_cleaned_outputs(
    report: CleaningReport, output_dir: Path | str = Path("data/cleaned")
) -> dict[str, Path]:
    """Write all four strict Phase 1b CSV outputs to `output_dir`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("cleaning_log.csv", "quarantine_report.csv"):
        stale_path = out / stale
        if stale_path.exists():
            stale_path.unlink()

    paths: dict[str, Path] = {}

    p_students = out / "students_clean.csv"
    with p_students.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "first_name", "last_name", "dob", "sex", "is_active"])
        for s in report.students:
            w.writerow([s.student_id, s.first_name, s.last_name, s.dob, s.sex, s.is_active])
    paths["students"] = p_students

    p_enrolments = out / "enrolments_clean.csv"
    with p_enrolments.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "season_year", "sport", "grade"])
        for e in report.enrolments:
            w.writerow([e.student_id, e.season_year, e.sport, e.grade])
    paths["enrolments"] = p_enrolments

    p_results = out / "test_results_clean.csv"
    with p_results.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "student_id", "season_year", "semester", "test_id", "test_name",
            "value", "level", "shuttle", "source_row",
        ])
        for r in report.test_results:
            w.writerow([
                r.student_id, r.season_year, r.semester, r.test_id, r.test_name,
                "" if r.value is None else r.value,
                "" if r.level is None else r.level,
                "" if r.shuttle is None else r.shuttle,
                r.source_row,
            ])
    paths["test_results"] = p_results

    p_rejected = out / "rejected_data_lessons.csv"
    with p_rejected.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sheet", "excel_row", "student_id", "mistake_category",
            "raw_detail", "reason", "future_safeguard",
        ])
        for rec in report.rejected:
            w.writerow([
                rec.sheet, rec.excel_row, rec.student_id, rec.mistake_category,
                rec.raw_detail, rec.reason, rec.future_safeguard,
            ])
    paths["rejected"] = p_rejected

    return paths


def main() -> None:
    raw_path = Path("raw_test/ssp_data.xlsx")
    report = clean_workbook(raw_path)
    written = write_cleaned_outputs(report)
    print("Phase 1b STRICT Zero-Assumption Cleaning Complete (Zero Fabricated Dates)")
    print(f"  Spotless students retained:     {len(report.students)}")
    print(f"  Spotless enrolments retained:   {len(report.enrolments)}")
    print(f"  Spotless test results retained: {len(report.test_results)}")
    print(f"  Rejected rows/groups (lessons): {len(report.rejected)}")
    for label, p in written.items():
        print(f"  -> {label}: {p}")


if __name__ == "__main__":
    main()
