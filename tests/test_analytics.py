"""test_analytics.py - Phase 3: Comprehensive, 100% synthetic, CI-safe analytics tests.

Verifies:
  * Hand-worked statistical oracles (Hazen mid-rank percentiles with ties, ddof=1 SD,
    linear quantiles, z-scores, normal-CDF composite index).
  * Basis selection (`latest` default vs `best` vs `first`) preventing testing-frequency bias.
  * Cohort defaults (latest season, Grade+Sex matched, `cohort_label`, `enrolled_n` &
    `coverage_pct`, small-cohort flag without blanking numbers).
  * Raw-change athlete trends (`Improved`/`Unchanged`/`Declined`, direction-aligned
    `improvement_pct`, divide-by-zero protection, PB tracking, `chronology_uncertain`).
  * Safeguarding (`Height`/`Weight` blocked from leaderboards, excluded from percentiles
    and radar profiles; raw `dob` stripped from analytics outputs).
  * Level:Shuttle forward cumulative shuttles, inverse `"level:shuttle"` formatting, and
    unconfigured ordinal-only handling.
  * Left/Right asymmetry formula, per-unit thresholds (`5%/10%` Agility vs `10%/15%`
    Strength), dynamic custom L/R discovery, and same-session pairing.
  * Relative strength (`N/kg`, `kg/kg BW`, `imtp_bodyweight_multiples`).
  * Multi-sport deduplication, empty-frame safety, and `WhcService` RBAC + cache invalidation.
"""

from __future__ import annotations

import math
import statistics
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest
from argon2 import PasswordHasher

from whc.analytics import (
    asymmetry_report,
    athlete_category_profile,
    athlete_percentiles,
    athlete_personal_bests,
    athlete_trends,
    build_enriched_results,
    build_shuttle_lookup,
    cohort_benchmarks,
    compare_athletes,
    dashboard_overview,
    discover_lr_pairs,
    format_shuttle_score,
    relative_strength_metrics,
    select_by_basis,
    squad_leaderboard,
)
from whc.database import Database
from whc.exceptions import AuthenticationError, ValidationError
from whc.fitness_tests import DEFAULT_LEVEL_SHUTTLE_RULES, DEFAULT_TESTS
from whc.services import WhcService

FAST_HASHER = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)


# ---- Synthetic fixture builders ----------------------------------------------------
def _tests_df(extra_rows: list[tuple] | None = None) -> pl.DataFrame:
    rows = [
        (t.test_id, t.test_name, t.category, t.unit, t.min_value, t.max_value, t.score_direction.value)
        for t in DEFAULT_TESTS
    ]
    if extra_rows:
        rows.extend(extra_rows)
    return pl.DataFrame(
        rows,
        schema={
            "test_id": pl.Int64,
            "test_name": pl.String,
            "category": pl.String,
            "unit": pl.String,
            "min_value": pl.Float64,
            "max_value": pl.Float64,
            "score_direction": pl.String,
        },
        orient="row",
    )


def _students_df(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "student_id": pl.String,
            "first_name": pl.String,
            "last_name": pl.String,
            "dob": pl.Date,
            "sex": pl.String,
            "is_active": pl.Boolean,
        },
        orient="row",
    )


def _enrolments_df(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "enrolment_id": pl.Int64,
            "student_id": pl.String,
            "season_year": pl.Int64,
            "sport": pl.String,
            "grade": pl.String,
        },
        orient="row",
    )


def _results_df(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "result_id": pl.Int64,
            "student_id": pl.String,
            "test_id": pl.Int64,
            "test_date": pl.Date,
            "season_year": pl.Int64,
            "semester": pl.Int64,
            "value": pl.Float64,
            "level": pl.Int64,
            "shuttle": pl.Int64,
        },
        orient="row",
    )


@pytest.fixture
def sample_dataset() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """5 Year 9 Female athletes + 1 Year 10 Male + 1 untested Year 9 Female."""
    students = _students_df(
        [
            ("AAA0001", "Alice", "One", date(2011, 5, 10), "Female", True),
            ("AAA0002", "Beth", "Two", date(2011, 6, 15), "Female", True),
            ("AAA0003", "Cara", "Three", None, "Female", True),  # legacy no DOB
            ("AAA0004", "Dana", "Four", date(2011, 2, 1), "Female", True),
            ("AAA0005", "Ella", "Five", date(2011, 8, 20), "Female", True),
            ("AAA0006", "Untested", "Six", date(2011, 3, 3), "Female", True),
            ("BBB0001", "Ben", "Male", date(2010, 4, 4), "Male", True),
        ]
    )
    # AAA0001 is multi-sport (Basketball + Soccer in 2026)
    enrolments = _enrolments_df(
        [
            (1, "AAA0001", 2026, "Basketball", "Year 9"),
            (2, "AAA0001", 2026, "Soccer", "Year 9"),
            (3, "AAA0002", 2026, "Basketball", "Year 9"),
            (4, "AAA0003", 2026, "Basketball", "Year 9"),
            (5, "AAA0004", 2026, "Soccer", "Year 9"),
            (6, "AAA0005", 2026, "Netball", "Year 9"),
            (7, "AAA0006", 2026, "Basketball", "Year 9"),
            (8, "BBB0001", 2026, "AFL", "Year 10"),
        ]
    )
    # Hand-worked CMJ Height (101, higher): [10, 20, 20, 30, 40] for AAA0001..AAA0005
    # Hand-worked 20m Sprint (203, lower): [4.0, 3.6, 3.6, 3.2, 3.0] for AAA0001..AAA0005
    results = _results_df(
        [
            # CMJ Height (101) — 2026 Sem 1
            (1, "AAA0001", 101, date(2026, 3, 1), 2026, 1, 10.0, None, None),
            (2, "AAA0002", 101, date(2026, 3, 1), 2026, 1, 20.0, None, None),
            (3, "AAA0003", 101, None, 2026, 1, 20.0, None, None),
            (4, "AAA0004", 101, date(2026, 3, 1), 2026, 1, 30.0, None, None),
            (5, "AAA0005", 101, date(2026, 3, 1), 2026, 1, 40.0, None, None),
            (6, "BBB0001", 101, date(2026, 3, 1), 2026, 1, 65.0, None, None),
            # 20m Sprint (203, lower is better)
            (7, "AAA0001", 203, date(2026, 3, 1), 2026, 1, 4.0, None, None),
            (8, "AAA0002", 203, date(2026, 3, 1), 2026, 1, 3.6, None, None),
            (9, "AAA0003", 203, None, 2026, 1, 3.6, None, None),
            (10, "AAA0004", 203, date(2026, 3, 1), 2026, 1, 3.2, None, None),
            (11, "AAA0005", 203, date(2026, 3, 1), 2026, 1, 3.0, None, None),
            # Anthropometrics (901 Height, 902 Weight — neutral)
            (12, "AAA0005", 901, date(2026, 3, 1), 2026, 1, 170.0, None, None),
            (13, "AAA0005", 902, date(2026, 3, 1), 2026, 1, 60.0, None, None),
            # IMTP Peak (1) for relative strength
            (14, "AAA0005", 1, date(2026, 3, 1), 2026, 1, 1800.0, None, None),
        ]
    )
    return students, enrolments, _tests_df(), results


# ---- 1. Hand-worked statistical oracle tests ---------------------------------------
def test_hand_worked_hazen_percentiles_and_quantiles(sample_dataset) -> None:
    students, enrolments, tests, results = sample_dataset
    enriched = build_enriched_results(
        results, students, enrolments, tests, DEFAULT_LEVEL_SHUTTLE_RULES
    )

    # Privacy check: dob must be stripped, age_at_test computed accurately
    assert "dob" not in enriched.columns
    a1_age = enriched.filter(pl.col("student_id") == "AAA0001")["age_at_test"][0]
    a3_age = enriched.filter(pl.col("student_id") == "AAA0003")["age_at_test"][0]
    assert a1_age == 14  # born 2011-05-10, tested 2026-03-01 -> 14
    assert a3_age is None  # legacy student with no DOB/date

    # Multi-sport deduplication: AAA0001 counted ONCE with "Basketball, Soccer"
    cmj_rows = enriched.filter((pl.col("student_id") == "AAA0001") & (pl.col("test_id") == 101))
    assert cmj_rows.height == 1
    assert cmj_rows["sports"][0] == "Basketball, Soccer"

    # Cohort benchmarks for 2026 Year 9 Female CMJ: [10, 20, 20, 30, 40]
    bench = cohort_benchmarks(
        enriched,
        enrolments,
        students,
        DEFAULT_LEVEL_SHUTTLE_RULES,
        season_year=2026,
        grade="Year 9",
        sex="Female",
        test_id=101,
    )
    assert bench.height == 1
    row = bench.row(0, named=True)
    assert row["cohort_n"] == 5
    assert row["enrolled_n"] == 6  # includes untested AAA0006
    assert row["coverage_pct"] == pytest.approx(83.3, abs=0.1)
    assert row["small_cohort"] is False
    assert row["mean"] == pytest.approx(24.0)
    assert row["median"] == pytest.approx(20.0)
    assert row["std"] == pytest.approx(statistics.stdev([10.0, 20.0, 20.0, 30.0, 40.0]))
    assert row["p25"] == pytest.approx(20.0)
    assert row["p75"] == pytest.approx(30.0)
    assert row["best_value"] == 40.0
    assert row["worst_value"] == 10.0

    # Hand-worked Hazen mid-rank percentiles for [10, 20, 20, 30, 40]:
    # ranks asc avg = [1, 2.5, 2.5, 4, 5] -> Hazen = [10.0, 40.0, 40.0, 70.0, 90.0]
    expected_pcts = {
        "AAA0001": (10.0, 5),
        "AAA0002": (40.0, 3),
        "AAA0003": (40.0, 3),
        "AAA0004": (70.0, 2),
        "AAA0005": (90.0, 1),
    }
    for sid, (exp_pct, exp_rank) in expected_pcts.items():
        p = athlete_percentiles(enriched, enrolments, students, sid)
        cmj = p.filter(pl.col("test_id") == 101).row(0, named=True)
        assert cmj["percentile"] == pytest.approx(exp_pct)
        assert cmj["rank"] == exp_rank
        assert cmj["cohort_n"] == 5
        assert "Year 9 Female" in cmj["cohort_label"]
        # 20m Sprint (lower is better): AAA0005 (3.0s) is #1 (90th pct), AAA0001 (4.0s) is #5 (10th pct)
        spr = p.filter(pl.col("test_id") == 203).row(0, named=True)
        assert spr["percentile"] == pytest.approx(exp_pct)
        assert spr["rank"] == exp_rank

    # Z-score & normal-CDF composite check for AAA0005 (CMJ=40, mean=24, std=11.401754)
    p5 = athlete_percentiles(enriched, enrolments, students, "AAA0005")
    cmj5 = p5.filter(pl.col("test_id") == 101).row(0, named=True)
    exp_z = (40.0 - 24.0) / statistics.stdev([10.0, 20.0, 20.0, 30.0, 40.0])
    assert cmj5["z_score"] == pytest.approx(exp_z, abs=0.002)

    # Category radar profile
    cat = athlete_category_profile(p5, tests)
    power_row = cat.filter(pl.col("category") == "Power").row(0, named=True)
    expected_comp = 0.5 * (1.0 + math.erf(round(exp_z, 3) / math.sqrt(2.0))) * 100.0
    assert power_row["composite_index"] == pytest.approx(expected_comp, abs=0.05)
    assert power_row["representative_test_id"] == 101
    assert power_row["representative_percentile"] == pytest.approx(90.0)
    assert power_row["tests_completed"] == 1


# ---- 2. Basis selection (latest vs best vs first) ----------------------------------
def test_basis_latest_avoids_best_of_window_bias(sample_dataset) -> None:
    students, enrolments, tests, results = sample_dataset
    # Give AAA0001 a lucky spike in Sem 1 (50.0) then regress to 15.0 in Sem 2
    extra = _results_df(
        [
            (100, "AAA0001", 101, date(2026, 4, 1), 2026, 1, 50.0, None, None),
            (101, "AAA0001", 101, date(2026, 9, 1), 2026, 2, 15.0, None, None),
        ]
    )
    enriched = build_enriched_results(
        pl.concat([results, extra]), students, enrolments, tests, DEFAULT_LEVEL_SHUTTLE_RULES
    )

    latest = select_by_basis(enriched.filter((pl.col("student_id") == "AAA0001") & (pl.col("test_id") == 101)), "latest")
    best = select_by_basis(enriched.filter((pl.col("student_id") == "AAA0001") & (pl.col("test_id") == 101)), "best")
    first = select_by_basis(enriched.filter((pl.col("student_id") == "AAA0001") & (pl.col("test_id") == 101)), "first")
    assert latest["comparable_value"][0] == 15.0
    assert best["comparable_value"][0] == 50.0
    assert first["comparable_value"][0] == 10.0

    # Under default basis="latest", AAA0001 is ranked 5th (15.0), NOT 1st (50.0)
    lb_latest = squad_leaderboard(
        enriched, enrolments, students, 101, grade="Year 9", sex="Female"
    )
    assert lb_latest.filter(pl.col("student_id") == "AAA0001")["rank"][0] == 5

    lb_best = squad_leaderboard(
        enriched, enrolments, students, 101, grade="Year 9", sex="Female", basis="best"
    )
    assert lb_best.filter(pl.col("student_id") == "AAA0001")["rank"][0] == 1

    with pytest.raises(ValidationError):
        select_by_basis(enriched, "median")  # type: ignore[arg-type]


# ---- 3. Small cohort flag (still computes percentiles, warns via small_cohort=True) -
def test_small_cohort_computes_percentile_and_sets_flag(sample_dataset) -> None:
    students, enrolments, tests, results = sample_dataset
    enriched = build_enriched_results(
        results, students, enrolments, tests, DEFAULT_LEVEL_SHUTTLE_RULES
    )
    # BBB0001 is the ONLY Year 10 Male (cohort_n = 1 < 5)
    p = athlete_percentiles(enriched, enrolments, students, "BBB0001")
    row = p.filter(pl.col("test_id") == 101).row(0, named=True)
    assert row["cohort_n"] == 1
    assert row["small_cohort"] is True
    assert row["percentile"] == pytest.approx(50.0)
    assert row["z_score"] is None  # cannot divide by 0 with n=1


# ---- 4. Raw-change trends, div-by-zero, chronology uncertainty, PBs ----------------
def test_athlete_trends_raw_change_div_zero_and_chronology() -> None:
    students = _students_df([("AAA0001", "Alex", "Test", date(2011, 1, 1), "Male", True)])
    enrolments = _enrolments_df(
        [
            (1, "AAA0001", 2025, "AFL", "Year 8"),
            (2, "AAA0001", 2026, "AFL", "Year 9"),
        ]
    )
    results = _results_df(
        [
            # 20m Sprint (lower): 3.20 -> 3.19 (raw Improved!) -> 3.19 (Unchanged) -> 3.25 (Declined)
            (1, "AAA0001", 203, None, 2025, 1, 3.20, None, None),
            (2, "AAA0001", 203, date(2026, 3, 1), 2026, 1, 3.19, None, None),
            (3, "AAA0001", 203, None, 2026, 1, 3.19, None, None),  # legacy + dated in same sem -> uncertain!
            (4, "AAA0001", 203, date(2026, 9, 1), 2026, 2, 3.25, None, None),
            # Push Up (601, higher): 0 -> 10 (prev == 0 must NOT crash / inf)
            (5, "AAA0001", 601, date(2026, 3, 1), 2026, 1, 0.0, None, None),
            (6, "AAA0001", 601, date(2026, 9, 1), 2026, 2, 10.0, None, None),
            # Weight (902, neutral): 60 -> 62 (change_status=None, is_pb=None)
            (7, "AAA0001", 902, date(2026, 3, 1), 2026, 1, 60.0, None, None),
            (8, "AAA0001", 902, date(2026, 9, 1), 2026, 2, 62.0, None, None),
        ]
    )
    enriched = build_enriched_results(
        results, students, enrolments, _tests_df(), DEFAULT_LEVEL_SHUTTLE_RULES
    )

    spr = athlete_trends(enriched, "AAA0001", test_id=203)
    assert spr["change_status"].to_list() == [None, "Improved", "Unchanged", "Declined"]
    assert spr["improvement_pct"][1] > 0  # faster sprint -> positive improvement_pct
    assert spr["improvement_pct"][3] < 0  # slower sprint -> negative improvement_pct
    assert spr["is_pb"].to_list() == [False, True, True, False]
    assert spr["chronology_uncertain"].to_list() == [False, True, True, False]

    push = athlete_trends(enriched, "AAA0001", test_id=601)
    assert push["pct_from_prev"][1] is None  # div-by-zero guarded
    assert push["improvement_pct"][1] is None
    assert push["change_status"][1] == "Improved"

    wt = athlete_trends(enriched, "AAA0001", test_id=902)
    assert wt["change_status"].to_list() == [None, None]
    assert wt["is_pb"].to_list() == [None, None]

    pbs = athlete_personal_bests(enriched, "AAA0001")
    wt_pb = pbs.filter(pl.col("test_id") == 902).row(0, named=True)
    assert wt_pb["pb_value"] is None
    assert wt_pb["latest_value"] == 62.0
    spr_pb = pbs.filter(pl.col("test_id") == 203).row(0, named=True)
    assert spr_pb["pb_value"] == 3.19
    assert spr_pb["is_latest_pb"] is False


# ---- 5. Safeguarding (Height / Weight excluded from leaderboards & percentiles) ----
def test_neutral_tests_safeguarding(sample_dataset) -> None:
    students, enrolments, tests, results = sample_dataset
    enriched = build_enriched_results(
        results, students, enrolments, tests, DEFAULT_LEVEL_SHUTTLE_RULES
    )
    with pytest.raises(ValidationError, match="neutral"):
        squad_leaderboard(enriched, enrolments, students, 901, tests_df=tests)
    with pytest.raises(ValidationError, match="neutral"):
        squad_leaderboard(enriched, enrolments, students, 902, tests_df=tests)

    p5 = athlete_percentiles(enriched, enrolments, students, "AAA0005")
    assert 901 not in p5["test_id"].to_list()
    assert 902 not in p5["test_id"].to_list()

    cat5 = athlete_category_profile(p5, tests)
    assert "Anthropometrics" not in cat5["category"].to_list()


# ---- 6. Level:Shuttle cumulative shuttles, inverse format, unconfigured ordinal ----
def test_level_shuttle_cumulative_inverse_and_unconfigured() -> None:
    # Verify Beep Test (402) 10:11 < 11:1 (naive level + shuttle/10 would break here!)
    lookup = build_shuttle_lookup(DEFAULT_LEVEL_SHUTTLE_RULES)
    sh_10_11 = lookup.filter((pl.col("test_id") == 402) & (pl.col("level") == 10) & (pl.col("shuttle") == 11))["cumulative_shuttles"][0]
    sh_11_1 = lookup.filter((pl.col("test_id") == 402) & (pl.col("level") == 11) & (pl.col("shuttle") == 1))["cumulative_shuttles"][0]
    assert sh_11_1 == sh_10_11 + 1.0
    assert format_shuttle_score(402, sh_10_11, DEFAULT_LEVEL_SHUTTLE_RULES) == "10:11"
    assert format_shuttle_score(402, sh_11_1, DEFAULT_LEVEL_SHUTTLE_RULES) == "11:1"

    # Yo-Yo IRT1 (401): 5:1 -> 1, 9:1 -> 2, 11:2 -> 4
    assert format_shuttle_score(401, 1.0, DEFAULT_LEVEL_SHUTTLE_RULES) == "5:1"
    assert format_shuttle_score(401, 2.0, DEFAULT_LEVEL_SHUTTLE_RULES) == "9:1"
    assert format_shuttle_score(401, 4.0, DEFAULT_LEVEL_SHUTTLE_RULES) == "11:2"

    # Unconfigured custom Level:Shuttle test (450) -> is_ordinal_only=True, mean/std=None
    custom_tests = _tests_df([(450, "Custom Shuttle", "Aerobic Endurance", "Level:Shuttle", None, None, "higher")])
    students = _students_df(
        [
            ("AAA0001", "A", "One", None, "Male", True),
            ("AAA0002", "B", "Two", None, "Male", True),
        ]
    )
    enrolments = _enrolments_df(
        [(1, "AAA0001", 2026, "AFL", "Year 9"), (2, "AAA0002", 2026, "AFL", "Year 9")]
    )
    results = _results_df(
        [
            (1, "AAA0001", 401, None, 2026, 1, None, 14, 2),
            (2, "AAA0002", 401, None, 2026, 1, None, 14, 4),
            (3, "AAA0001", 450, None, 2026, 1, None, 5, 1),
            (4, "AAA0002", 450, None, 2026, 1, None, 6, 2),
        ]
    )
    enriched = build_enriched_results(
        results, students, enrolments, custom_tests, DEFAULT_LEVEL_SHUTTLE_RULES
    )
    bench = cohort_benchmarks(
        enriched, enrolments, students, DEFAULT_LEVEL_SHUTTLE_RULES, season_year=2026
    )
    yoyo = bench.filter(pl.col("test_id") == 401).row(0, named=True)
    assert yoyo["is_ordinal_only"] is False
    assert yoyo["median_display"] == "14:3"
    assert yoyo["mean"] is not None

    custom = bench.filter(pl.col("test_id") == 450).row(0, named=True)
    assert custom["is_ordinal_only"] is True
    assert custom["mean"] is None
    assert custom["std"] is None


# ---- 7. Asymmetry (per-unit thresholds, dynamic custom L/R, same-session) ----------
def test_asymmetry_per_unit_thresholds_and_dynamic_discovery() -> None:
    custom_tests = _tests_df(
        [
            (150, "Single Leg Hop Left", "Power", "Centimetres", 50.0, 300.0, "higher"),
            (151, "Single Leg Hop Right", "Power", "Centimetres", 50.0, 300.0, "higher"),
        ]
    )
    pairs = discover_lr_pairs(custom_tests)
    assert any(p[0] == "Single Leg Hop" for p in pairs)

    students = _students_df([("AAA0001", "Sam", "Runner", date(2011, 1, 1), "Male", True)])
    enrolments = _enrolments_df([(1, "AAA0001", 2026, "Soccer", "Year 9")])
    results = _results_df(
        [
            # 5-0-5 Agility (Seconds, lower): L=2.50, R=2.66 -> asym = 0.16/2.66 = 6.02% -> Moderate (>=5%)
            (1, "AAA0001", 301, date(2026, 3, 1), 2026, 1, 2.50, None, None),
            (2, "AAA0001", 302, date(2026, 3, 1), 2026, 1, 2.66, None, None),
            # 5-0-5 on different date should NOT cross-pair with 2026-03-01
            (3, "AAA0001", 301, date(2026, 4, 1), 2026, 1, 2.40, None, None),
            # Nordic Hamstring (Newtons, higher): L=300, R=320 -> asym = 20/320 = 6.25% -> Normal (<10%)
            (4, "AAA0001", 3, date(2026, 3, 1), 2026, 1, 300.0, None, None),
            (5, "AAA0001", 4, date(2026, 3, 1), 2026, 1, 320.0, None, None),
            # Single Leg Hop: L=200, R=160 -> asym = 40/200 = 20.0% -> Elevated (>=15%)
            (6, "AAA0001", 150, date(2026, 3, 1), 2026, 1, 200.0, None, None),
            (7, "AAA0001", 151, date(2026, 3, 1), 2026, 1, 160.0, None, None),
            # Legacy 5-0-5 session (test_date=None) MUST pair on (year, semester)
            (8, "AAA0001", 301, None, 2026, 2, 2.50, None, None),
            (9, "AAA0001", 302, None, 2026, 2, 2.80, None, None),
        ]
    )
    enriched = build_enriched_results(
        results, students, enrolments, custom_tests, DEFAULT_LEVEL_SHUTTLE_RULES
    )
    rep = asymmetry_report(enriched, custom_tests)
    assert rep.height == 4  # 3 dated pairs + 1 legacy pair; 2026-04-01 unpaired Left excluded

    legacy_agil = rep.filter(
        (pl.col("pair_name") == "5-0-5 Agility Test") & (pl.col("semester") == 2)
    ).row(0, named=True)
    assert legacy_agil["asymmetry_flag"] == "Elevated"  # 0.30/2.80 = 10.71% >= 10%
    assert legacy_agil["period_label"] == "2026 Sem 2"

    agil = rep.filter(
        (pl.col("pair_name") == "5-0-5 Agility Test") & (pl.col("semester") == 1)
    ).row(0, named=True)
    assert agil["asymmetry_pct"] == pytest.approx(6.02, abs=0.02)
    assert agil["better_side"] == "Left"  # lower seconds is better
    assert agil["asymmetry_flag"] == "Moderate"

    nordic = rep.filter(pl.col("pair_name") == "Nordic Hamstring").row(0, named=True)
    assert nordic["asymmetry_pct"] == pytest.approx(6.25, abs=0.02)
    assert nordic["better_side"] == "Right"  # higher Newtons is better
    assert nordic["asymmetry_flag"] == "Normal"

    hop = rep.filter(pl.col("pair_name") == "Single Leg Hop").row(0, named=True)
    assert hop["asymmetry_pct"] == pytest.approx(20.0)
    assert hop["better_side"] == "Left"
    assert hop["asymmetry_flag"] == "Elevated"


# ---- 8. Relative strength, compare_athletes, overview & empty frames ---------------
def test_relative_strength_compare_overview_and_empty(sample_dataset) -> None:
    students, enrolments, tests, results = sample_dataset
    enriched = build_enriched_results(
        results, students, enrolments, tests, DEFAULT_LEVEL_SHUTTLE_RULES
    )

    rel = relative_strength_metrics(enriched)
    assert rel.height == 1
    r0 = rel.row(0, named=True)
    assert r0["relative_value"] == pytest.approx(30.0)  # 1800 N / 60 kg
    assert r0["imtp_bodyweight_multiples"] == pytest.approx(1800.0 / (60.0 * 9.81), abs=0.005)
    assert r0["same_session_weight"] is True

    cmp = compare_athletes(enriched, enrolments, students, ["AAA0001", "AAA0005"], test_ids=[101, 203])
    assert cmp.height == 4

    ov = dashboard_overview(students, enrolments, results, season_year=2026)
    assert ov["total_students"] == 7
    assert ov["tested_students"] == 6
    assert ov["untested_students"] == 1

    # Empty DataFrame safety across every function
    empty_res = results.head(0)
    empty_enr = build_enriched_results(empty_res, students, enrolments, tests, DEFAULT_LEVEL_SHUTTLE_RULES)
    assert empty_enr.is_empty()
    assert athlete_trends(empty_enr, "AAA0001").is_empty()
    assert athlete_personal_bests(empty_enr, "AAA0001").is_empty()
    assert cohort_benchmarks(empty_enr, enrolments, students, DEFAULT_LEVEL_SHUTTLE_RULES).is_empty()
    assert athlete_percentiles(empty_enr, enrolments, students, "AAA0001").is_empty()
    assert athlete_category_profile(athlete_percentiles(empty_enr, enrolments, students, "AAA0001"), tests).is_empty()
    assert asymmetry_report(empty_enr, tests).is_empty()
    assert relative_strength_metrics(empty_enr).is_empty()
    assert squad_leaderboard(empty_enr, enrolments, students, 101, tests_df=tests).is_empty()
    assert compare_athletes(empty_enr, enrolments, students, ["AAA0001", "AAA0005"]).is_empty()


# ---- 9. WhcService RBAC & Snapshot Cache Invalidation ------------------------------
def test_service_analytics_rbac_and_cache_invalidation(tmp_path: Path) -> None:
    now = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)
    db = Database(tmp_path / "analytics.db")
    svc = WhcService(
        db,
        backup_dir=tmp_path / "backups",
        clock=lambda: now,
        hasher=FAST_HASHER,
        enable_auto_backups=False,
    )
    svc.bootstrap_lead_coach("lead", "Lead Coach", "StrongPass!234")
    _, lead_tok = svc.login("lead", "StrongPass!234")
    _, temp_pw = svc.create_user(lead_tok, "coach1", "Coach One", "Coach")
    _, coach_tok = svc.login("coach1", temp_pw)

    # Must-change-password gate blocks analytics
    with pytest.raises(AuthenticationError):
        svc.dashboard_overview(coach_tok)

    svc.change_my_password(coach_tok, temp_pw, "CoachPass!2345")
    _, coach_tok = svc.login("coach1", "CoachPass!2345")

    # Add student + enrolment + result
    svc.add_student(coach_tok, "AAA0001", "Jane", "Doe", "10/05/2011", "Female")
    svc.add_enrolment(coach_tok, "AAA0001", 2026, "Basketball", "Year 9")
    svc.add_result(coach_tok, "AAA0001", 101, "15/03/2026", 35.0)

    ov1 = svc.dashboard_overview(coach_tok, season_year=2026)
    assert ov1["total_results"] == 1

    # Second call hits snapshot cache
    assert svc._snapshot_cache is not None
    trends1 = svc.athlete_trends(coach_tok, "AAA0001", test_id=101)
    assert trends1.height == 1

    # Adding a new result MUST immediately invalidate the snapshot cache
    svc.add_result(coach_tok, "AAA0001", 101, "20/05/2026", 38.5)
    assert svc._snapshot_cache is None
    trends2 = svc.athlete_trends(coach_tok, "AAA0001", test_id=101)
    assert trends2.height == 2
    assert trends2["change_status"][1] == "Improved"

    # Unauthenticated token rejected
    with pytest.raises(AuthenticationError):
        svc.squad_leaderboard("bad-token", 101)
