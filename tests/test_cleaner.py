"""tests/test_cleaner.py - unit tests for the strict zero-assumption Phase 1b cleaner."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from whc.cleaner import (
    clean_workbook,
    repair_mojibake,
    write_cleaned_outputs,
)


def _make_synthetic_xlsx(path: Path, sheets: dict[str, dict[int, dict[str, str]]]) -> Path:
    """Create a minimal valid .xlsx workbook in memory for isolated unit testing."""
    sheet_names = list(sheets.keys())
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        ct_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '<Default Extension="xml" ContentType="application/xml"/>',
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        ]
        for idx in range(1, len(sheet_names) + 1):
            ct_parts.append(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        ct_parts.append("</Types>")
        zf.writestr("[Content_Types].xml", "".join(ct_parts))

        wb_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>',
        ]
        rel_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        ]
        for idx, name in enumerate(sheet_names, start=1):
            wb_parts.append(f'<sheet name="{name}" sheetId="{idx}" r:id="rId{idx}"/>')
            rel_parts.append(
                f'<Relationship Id="rId{idx}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{idx}.xml"/>'
            )
        wb_parts.append("</sheets></workbook>")
        rel_parts.append("</Relationships>")
        zf.writestr("xl/workbook.xml", "".join(wb_parts))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(rel_parts))

        for idx, name in enumerate(sheet_names, start=1):
            root = ET.Element(
                "worksheet",
                {"xmlns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"},
            )
            sd = ET.SubElement(root, "sheetData")
            for rnum, row_dict in sorted(sheets[name].items()):
                r_el = ET.SubElement(sd, "row", {"r": str(rnum)})
                for col, val in row_dict.items():
                    c_el = ET.SubElement(r_el, "c", {"r": f"{col}{rnum}", "t": "inlineStr"})
                    is_el = ET.SubElement(c_el, "is")
                    t_el = ET.SubElement(is_el, "t")
                    t_el.text = val
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", ET.tostring(root, encoding="utf-8"))
    return path


def test_repair_mojibake_fixes_utf8_in_latin1() -> None:
    assert repair_mojibake("InÃ¨s") == "Inès"
    assert repair_mojibake("Adam") == "Adam"


def test_strict_cleaner_keeps_only_spotless_rows_and_rejects_all_imperfections(tmp_path: Path) -> None:
    xlsx = _make_synthetic_xlsx(
        tmp_path / "sample.xlsx",
        {
            "All Students": {
                1: {"A": "AAA0001", "B": "AAA0001", "C": "SMITH", "D": "InÃ¨s", "E": "Female", "H": "Year 10"},
                2: {"A": "SUSSI ID", "B": "Import Identifier", "C": "Last Name", "D": "First Name", "E": "Gender", "H": "Year Level"},
                3: {"A": "BBB0002", "B": "BBB0002", "C": "JONES", "D": "Alex", "E": "Male", "H": "Year 9"},
                # Mismatched Col A vs Col B -> rejected!
                4: {"A": "CCC0003", "B": "ZZZ9999", "C": "CORBETT", "D": "Max", "E": "Male", "H": "Year 10"},
            },
            "Sheet4": {
                104: {"B": "rundel", "C": "165.0"},
            },
            "SSP AMS Database": {
                1: {"A": "Student ID Code", "B": "Name", "C": "Grade Level", "D": "Year", "E": "Semester", "F": "Sex", "G": "Sport"},
                # 100% spotless row -> RETAINED
                2: {"A": "AAA0001", "B": "SMITH, Ines", "C": "10", "D": "2026", "E": "1", "F": "Female", "G": "Basket Ball", "H": "60.5", "P": "15.1"},
                # Exact copy-paste duplicate of row 2 -> deduplicated cleanly (1 copy kept)
                3: {"A": "AAA0001", "B": "SMITH, Ines", "C": "10", "D": "2026", "E": "1", "F": "Female", "G": "Basket Ball", "H": "60.5", "P": "15.1"},
                # Blank Year -> REJECTED (no inferring!)
                4: {"A": "AAA0001", "B": "SMITH, Ines", "C": "10", "D": "", "E": "2", "F": "Female", "G": "AFL", "H": "61.0"},
                # Mangled ID -> REJECTED (no guessing!)
                5: {"A": "MAYO", "B": "Student ID code not entered", "C": "8", "D": "2026", "E": "1", "F": "", "G": "AFL", "H": "40.5"},
                # Comma decimal typo -> ENTIRE ROW REJECTED (even though Weight 62.0 is valid!)
                6: {"A": "AAA0001", "B": "SMITH, Ines", "C": "10", "D": "2026", "E": "2", "F": "Female", "G": "Soccer", "H": "62.0", "K": "45,03"},
                # Split/conflicting duplicate session for BBB0002 -> ENTIRE GROUP REJECTED (no merging!)
                7: {"A": "BBB0002", "B": "JONES, Alex", "C": "9", "D": "2026", "E": "2", "F": "Male", "G": "Soccer", "H": "65.0"},
                8: {"A": "BBB0002", "B": "JONES, Alex", "C": "9", "D": "2026", "E": "2", "F": "Male", "G": "Soccer", "H": "66.0", "I": "175.0"},
                # Student not in All Students -> REJECTED even if Name & Sex are present
                9: {"A": "NEW0001", "B": "Koa Van Kol", "C": "8", "D": "2026", "E": "2", "F": "Female", "G": "AFL", "H": "55.0"},
                # Missing Semester -> REJECTED
                10: {"A": "AAA0001", "B": "SMITH, Ines", "C": "9", "D": "2025", "E": "", "F": "Female", "G": "Netball", "I": "168.0"},
                # Impossible Yo-Yo IRT1 shuttle .9 -> ENTIRE ROW REJECTED
                11: {"A": "AAA0001", "B": "SMITH, Ines", "C": "10", "D": "2026", "E": "2", "F": "Female", "G": "Netball", "H": "60.0", "P": "16.9"},
                # IMTP Peak vs Relative physics contradiction -> ENTIRE ROW REJECTED
                12: {"A": "BBB0002", "B": "JONES, Alex", "C": "9", "D": "2026", "E": "1", "F": "Male", "G": "AFL", "H": "60.0", "N": "2.3", "O": "2371.0"},
            },
        },
    )

    report = clean_workbook(xlsx)

    # Only AAA0001 and BBB0002 survive in students (header row 2 & mismatched ID row 4 rejected)
    assert {s.student_id for s in report.students} == {"AAA0001", "BBB0002"}

    # ONLY the 2 spotless tests from Row 2 (deduplicated with Row 3) survive — with NO fabricated dates!
    res_keys = {(r.student_id, r.season_year, r.semester, r.test_id, r.value, r.level, r.shuttle) for r in report.test_results}
    assert res_keys == {
        ("AAA0001", 2026, 1, 902, 60.5, None, None),
        ("AAA0001", 2026, 1, 401, None, 15, 1),
    }

    # Verify all mistake categories are captured with future safeguards
    cats = {r.mistake_category for r in report.rejected}
    assert {
        "misplaced_header_row",
        "mismatched_roster_ids",
        "stray_scratch_sheet",
        "missing_year",
        "invalid_or_mangled_student_id",
        "invalid_or_out_of_range_test_value",
        "physics_contradiction",
        "duplicate_conflicting_or_split_session",
        "student_not_in_authoritative_roster",
        "missing_semester",
    } <= cats
    assert all(r.future_safeguard for r in report.rejected)

    # Verify write_cleaned_outputs writes all 4 CSVs cleanly
    out_dir = tmp_path / "cleaned"
    written = write_cleaned_outputs(report, out_dir)
    assert set(written) == {"students", "enrolments", "test_results", "rejected"}
    for p in written.values():
        assert p.exists()
        with p.open("r", encoding="utf-8") as f:
            assert len(list(csv.reader(f))) >= 2


def test_real_raw_workbook_strict_clean_is_read_only(tmp_path: Path) -> None:
    raw = Path("raw_test/ssp_data.xlsx")
    if not raw.exists():
        pytest.skip("raw_test/ssp_data.xlsx not present")
    before_bytes = raw.read_bytes()
    report = clean_workbook(raw)
    assert raw.read_bytes() == before_bytes  # guaranteed untouched

    # Retained records must be non-empty and write cleanly to all 4 CSV outputs
    assert len(report.students) > 0
    assert len(report.test_results) > 0
    assert len(report.rejected) >= 0
    written = write_cleaned_outputs(report, tmp_path / "strict_cleaned")
    assert all(p.exists() for p in written.values())
