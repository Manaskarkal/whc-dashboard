"""analytics.py - Phase 3: Headless, pure-Polars sports-science analytics engine.

Zero UI imports (no Streamlit, no Plotly) and zero SQLite imports — every function operates
purely on Polars DataFrames so it is fast, deterministic, and 100% unit-testable.

Locked design decisions:
    * Basis: defaults to `"latest"` per `(student_id, test_id)` within a cohort window
      (`"best"` and `"first"` are opt-in) to avoid best-of-window testing-frequency bias.
    * Default cohort: latest season + same Grade + same Sex (all sports), with an explicit
      human-readable `cohort_label` and `enrolled_n` / `coverage_pct` selection-bias metrics.
    * Small cohorts (`cohort_n < min_cohort_size`, default 5): still calculates percentiles
      and quartiles (except `z_score = None` when `cohort_n < 2` or `std == 0`) while setting
      `small_cohort = True` so coaches/UI can interpret small groups with context.
    * Athlete trends: raw change (`delta_from_prev`, direction-aligned `improvement_pct` with
      divide-by-zero protection, and `"Improved"` / `"Unchanged"` / `"Declined"`) so coaches
      decide what is practically meaningful.
    * Safeguarding (`ScoreDirection.NEUTRAL` — Height & Weight): excluded from leaderboards
      (raises ValidationError), athlete percentiles, and radar profiles. Available only as raw
      longitudinal trends and internally for relative-strength normalization.
    * Minor PII minimization: `dob` is dropped from analytics outputs once `age_at_test` is
      computed (nullable when `dob` or `test_date` is absent — never fabricated).
    * Level:Shuttle: converted to linear cumulative shuttles via `level_shuttle_rules` with
      inverse `"level:shuttle"` display formatting. Unconfigured shuttle tests are flagged
      `is_ordinal_only = True` (mean/std/z-score/% deltas nulled).
    * Left/Right Asymmetry: `(max - min) / max * 100` with per-unit thresholds
      (`5% / 10%` for Agility Seconds; `10% / 15%` for Strength Newtons) and non-diagnostic
      `asymmetry_flag` labels (`"Normal"`, `"Moderate"`, `"Elevated"`).
    * Two Radar Charts:
      1. Category Radar (`athlete_category_profile`): z-score `composite_index` (0-100) AND
         headline `representative_percentile` per category.
      2. Grouped Test-by-Test Radar (`athlete_percentiles`): every completed non-neutral test
         ordered by `(test_id // 100, test_id)` so similar tests sit together around the wheel.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import polars as pl

from whc.exceptions import ValidationError
from whc.fitness_tests import IDS_PER_CATEGORY, ScoreDirection, Unit
from whc.students import clean_student_id
from whc.test_results import IMTP_PEAK_ID, WEIGHT_ID

Basis = Literal["latest", "best", "first"]
SeasonFilter = int | Literal["all"] | None

VALID_BASES: frozenset[str] = frozenset({"latest", "best", "first"})
DEFAULT_MIN_COHORT_SIZE = 5

# Per-unit asymmetry thresholds: (moderate_pct, elevated_pct)
ASYMMETRY_THRESHOLDS_BY_UNIT: dict[str, tuple[float, float]] = {
    Unit.SECONDS.value: (5.0, 10.0),  # 5-0-5 Agility COD
    Unit.NEWTONS.value: (10.0, 15.0),  # Nordic, Adduction, Abduction
}
DEFAULT_ASYMMETRY_THRESHOLDS: tuple[float, float] = (10.0, 15.0)

# Default headline/representative test per category for Category Radar toggle
PRIMARY_CATEGORY_TESTS: dict[str, int] = {
    "Strength": 2,  # IMTP Relative (falls back to first completed Strength test)
    "Power": 101,  # CMJ Height
    "Speed": 203,  # 20m Sprint
    "Agility": 301,  # 5-0-5 Agility Left
    "Aerobic Endurance": 401,  # Yo-Yo IRT1
    "Anaerobic Endurance": 501,  # Yo-Yo IRT2
    "Muscular Endurance": 601,  # Push Up Test
    "Flexibility": 701,  # Sit-and-Reach
    "Reaction": 801,  # Reaction Time Ruler Drop
}

# Absolute strength tests eligible for body-weight normalization
RELATIVE_STRENGTH_TEST_IDS: dict[int, str] = {
    IMTP_PEAK_ID: "N/kg",
    9: "kg/kg BW",  # 1RM Back Squat
    10: "kg/kg BW",  # 1RM Bench Press
    11: "kg/kg BW",  # 1RM Trap Bar Deadlift
    12: "kg/kg BW",  # 3RM Back Squat
    13: "kg/kg BW",  # 3RM Bench Press
    14: "kg/kg BW",  # 3RM Trap Bar Deadlift
}

_ENRICHED_SCHEMA: dict[str, pl.DataType] = {
    "result_id": pl.Int64,
    "student_id": pl.String,
    "full_name": pl.String,
    "sex": pl.String,
    "is_active": pl.Boolean,
    "grade": pl.String,
    "sports": pl.String,
    "test_id": pl.Int64,
    "test_name": pl.String,
    "category": pl.String,
    "unit": pl.String,
    "score_direction": pl.String,
    "season_year": pl.Int64,
    "semester": pl.Int64,
    "test_date": pl.Date,
    "is_legacy": pl.Boolean,
    "period_label": pl.String,
    "value": pl.Float64,
    "level": pl.Int64,
    "shuttle": pl.Int64,
    "comparable_value": pl.Float64,
    "is_ordinal_only": pl.Boolean,
    "formatted_score": pl.String,
    "age_at_test": pl.Int64,
}


# ---- Pure math & formatting helpers ------------------------------------------------
def _check_basis(basis: str) -> Basis:
    if basis not in VALID_BASES:
        raise ValidationError(f"basis must be one of {sorted(VALID_BASES)}, got {basis!r}")
    return basis  # type: ignore[return-value]


def _normal_cdf_100(z: float | None) -> float | None:
    """Convert a z-score (clamped to [-3, +3]) to a 0-100 normal-equivalent composite index."""
    if z is None or not math.isfinite(z):
        return None
    clamped = max(-3.0, min(3.0, z))
    return round(0.5 * (1.0 + math.erf(clamped / math.sqrt(2.0))) * 100.0, 2)


def _format_number(val: float | None) -> str | None:
    if val is None or not math.isfinite(val):
        return None
    return f"{val:g}"


def _build_cohort_label(
    *,
    season_year: int | Literal["all"],
    semester: int | None,
    grade: str | None,
    sex: str | None,
    sport: str | None,
    basis: Basis,
    active_only: bool,
) -> str:
    parts: list[str] = []
    if season_year == "all":
        parts.append("All Seasons")
    elif semester is not None:
        parts.append(f"{season_year} Sem {semester}")
    else:
        parts.append(str(season_year))

    if grade and sex:
        parts.append(f"{grade} {sex}")
    elif grade:
        parts.append(grade)
    elif sex:
        parts.append(f"All Grades {sex}")
    else:
        parts.append("All Grades & Sexes")

    if sport:
        parts.append(f"{sport} (season enrolment)")
    if active_only:
        parts.append("Active only")
    return f"{' • '.join(parts)} ({basis.capitalize()})"


def build_shuttle_lookup(rules_by_test: dict[int, dict[int, int]]) -> pl.DataFrame:
    """Precompute `(test_id, level, shuttle) -> cumulative_shuttles` from protocol rules."""
    rows: list[tuple[int, int, int, float]] = []
    for test_id, level_map in sorted(rules_by_test.items()):
        cumulative = 0
        for lvl in sorted(level_map):
            max_sh = level_map[lvl]
            for sh in range(1, max_sh + 1):
                cumulative += 1
                rows.append((int(test_id), int(lvl), int(sh), float(cumulative)))
    return pl.DataFrame(
        rows,
        schema={
            "test_id": pl.Int64,
            "level": pl.Int64,
            "shuttle": pl.Int64,
            "cumulative_shuttles": pl.Float64,
        },
        orient="row",
    )


def format_shuttle_score(
    test_id: int,
    shuttles: float | None,
    rules_by_test: dict[int, dict[int, int]],
) -> str | None:
    """Inverse lookup: convert a cumulative shuttle count back to `'level:shuttle'`."""
    if shuttles is None or not math.isfinite(shuttles):
        return None
    level_map = rules_by_test.get(test_id)
    if not level_map:
        return None
    ordered: list[tuple[int, int]] = []
    for lvl in sorted(level_map):
        for sh in range(1, level_map[lvl] + 1):
            ordered.append((lvl, sh))
    if not ordered:
        return None
    idx = max(1, min(len(ordered), int(round(shuttles))))
    lvl, sh = ordered[idx - 1]
    return f"{lvl}:{sh}"


def _str_col(name: str) -> pl.Expr:
    return pl.col(name).cast(pl.String)


def _dedupe_enrolments(enrolments_df: pl.DataFrame) -> pl.DataFrame:
    """Produce 1 row per `(student_id, season_year)` with `grade` and sorted comma-joined `sports`."""
    if enrolments_df.is_empty():
        return pl.DataFrame(
            schema={
                "student_id": pl.String,
                "season_year": pl.Int64,
                "grade": pl.String,
                "sports": pl.String,
            }
        )
    norm = enrolments_df.select(
        pl.col("student_id").cast(pl.String),
        pl.col("season_year").cast(pl.Int64),
        _str_col("grade").alias("grade"),
        _str_col("sport").alias("sport"),
    )
    conflicts = (
        norm.group_by(["student_id", "season_year"])
        .agg(pl.col("grade").n_unique().alias("n_grades"))
        .filter(pl.col("n_grades") > 1)
    )
    if conflicts.height > 0:
        bad = conflicts.row(0)
        raise ValidationError(
            f"conflicting grades found for student {bad[0]} in season {bad[1]}"
        )
    return (
        norm.group_by(["student_id", "season_year"])
        .agg(
            pl.col("grade").first().alias("grade"),
            pl.col("sport").unique().sort().implode().list.join(", ").alias("sports"),
        )
    )


def _resolve_season(
    df: pl.DataFrame,
    season_year: SeasonFilter,
    *,
    student_id: str | None = None,
) -> int | Literal["all"]:
    if season_year == "all":
        return "all"
    if season_year is not None:
        if isinstance(season_year, bool) or not isinstance(season_year, int):
            raise ValidationError("season_year must be an integer, 'all', or None")
        return season_year
    # Default to latest season available (for the athlete if given, else overall)
    scope = df.filter(pl.col("student_id") == student_id) if student_id is not None else df
    if scope.is_empty():
        scope = df
    if scope.is_empty():
        return "all"
    latest = scope["season_year"].max()
    return int(latest) if latest is not None else "all"


def _count_enrolled(
    enrolments_df: pl.DataFrame,
    students_df: pl.DataFrame,
    *,
    season_year: int | Literal["all"],
    sport: str | None = None,
    grade: str | None = None,
    sex: str | None = None,
    active_only: bool = False,
) -> int:
    """Count distinct eligible enrolled students matching the exact cohort filter."""
    if enrolments_df.is_empty() or students_df.is_empty():
        return 0
    enrol = enrolments_df.select(
        pl.col("student_id").cast(pl.String),
        pl.col("season_year").cast(pl.Int64),
        _str_col("sport").alias("sport"),
        _str_col("grade").alias("grade"),
    )
    studs = students_df.select(
        pl.col("student_id").cast(pl.String),
        _str_col("sex").alias("sex"),
        pl.col("is_active").cast(pl.Boolean),
    )
    joined = enrol.join(studs, on="student_id", how="inner")
    if season_year != "all":
        joined = joined.filter(pl.col("season_year") == season_year)
    if sport is not None:
        target = sport.strip().casefold()
        joined = joined.filter(pl.col("sport").str.to_lowercase() == target)
    if grade is not None:
        joined = joined.filter(pl.col("grade") == grade)
    if sex is not None:
        joined = joined.filter(pl.col("sex") == sex)
    if active_only:
        joined = joined.filter(pl.col("is_active"))
    return int(joined["student_id"].n_unique())


def _coverage_pct(cohort_n: int, enrolled_n: int) -> float | None:
    if enrolled_n <= 0:
        return None
    return round(min(100.0, (cohort_n / enrolled_n) * 100.0), 1)


# ---- 1. Canonical Enriched Builder -------------------------------------------------
def build_enriched_results(
    results_df: pl.DataFrame,
    students_df: pl.DataFrame,
    enrolments_df: pl.DataFrame,
    tests_df: pl.DataFrame,
    shuttle_rules: dict[int, dict[int, int]],
    *,
    sport: str | None = None,
    active_only: bool = False,
) -> pl.DataFrame:
    """Build the canonical analytics DataFrame (strictly excludes untested students).

    - Multi-sport safe: LEFT-joins deduplicated `(student_id, season_year)` enrolments so an
      athlete enrolled in 2 sports is NEVER double-counted.
    - Sport filter: when `sport` is provided, restricts to `(student_id, season_year)` pairs
      enrolled in that sport (annual SSP enrolment grain) while preserving full `sports`.
    - Privacy: computes `age_at_test` when both `dob` and `test_date` exist, then drops `dob`.
    """
    if results_df.is_empty():
        return pl.DataFrame(schema=_ENRICHED_SCHEMA)

    res = results_df.select(
        pl.col("result_id").cast(pl.Int64),
        pl.col("student_id").cast(pl.String),
        pl.col("test_id").cast(pl.Int64),
        pl.col("test_date").cast(pl.Date),
        pl.col("season_year").cast(pl.Int64),
        pl.col("semester").cast(pl.Int64),
        pl.col("value").cast(pl.Float64),
        pl.col("level").cast(pl.Int64),
        pl.col("shuttle").cast(pl.Int64),
    )

    if sport is not None:
        target_sport = sport.strip().casefold()
        sport_pairs = (
            enrolments_df.select(
                pl.col("student_id").cast(pl.String),
                pl.col("season_year").cast(pl.Int64),
                _str_col("sport").alias("sport"),
            )
            .filter(pl.col("sport").str.to_lowercase() == target_sport)
            .select(["student_id", "season_year"])
            .unique()
        )
        res = res.join(sport_pairs, on=["student_id", "season_year"], how="semi")
        if res.is_empty():
            return pl.DataFrame(schema=_ENRICHED_SCHEMA)

    studs = students_df.select(
        pl.col("student_id").cast(pl.String),
        (pl.col("first_name").cast(pl.String) + pl.lit(" ") + pl.col("last_name").cast(pl.String)).alias("full_name"),
        pl.col("dob").cast(pl.Date),
        _str_col("sex").alias("sex"),
        pl.col("is_active").cast(pl.Boolean),
    )
    tests = tests_df.select(
        pl.col("test_id").cast(pl.Int64),
        pl.col("test_name").cast(pl.String),
        _str_col("category").alias("category"),
        _str_col("unit").alias("unit"),
        _str_col("score_direction").alias("score_direction"),
    )
    enrol_dedup = _dedupe_enrolments(enrolments_df)
    shuttle_lookup = build_shuttle_lookup(shuttle_rules)

    df = (
        res.join(studs, on="student_id", how="inner")
        .join(tests, on="test_id", how="inner")
        .join(enrol_dedup, on=["student_id", "season_year"], how="left")
        .join(shuttle_lookup, on=["test_id", "level", "shuttle"], how="left")
    )

    if active_only:
        df = df.filter(pl.col("is_active"))
    if df.is_empty():
        return pl.DataFrame(schema=_ENRICHED_SCHEMA)

    is_shuttle_test = pl.col("unit") == pl.lit(Unit.LEVEL_SHUTTLE.value)
    has_cum_shuttle = pl.col("cumulative_shuttles").is_not_null()

    age_expr = (
        pl.when(pl.col("dob").is_not_null() & pl.col("test_date").is_not_null())
        .then(
            pl.col("test_date").dt.year().cast(pl.Int64)
            - pl.col("dob").dt.year().cast(pl.Int64)
            - pl.when(
                (pl.col("test_date").dt.month() < pl.col("dob").dt.month())
                | (
                    (pl.col("test_date").dt.month() == pl.col("dob").dt.month())
                    & (pl.col("test_date").dt.day() < pl.col("dob").dt.day())
                )
            )
            .then(1)
            .otherwise(0)
        )
        .otherwise(None)
        .cast(pl.Int64)
        .alias("age_at_test")
    )

    comparable_expr = (
        pl.when(is_shuttle_test & has_cum_shuttle)
        .then(pl.col("cumulative_shuttles"))
        .when(is_shuttle_test)
        .then(pl.col("level").cast(pl.Float64) * 100.0 + pl.col("shuttle").cast(pl.Float64))
        .otherwise(pl.col("value"))
        .cast(pl.Float64)
        .alias("comparable_value")
    )

    ordinal_only_expr = (is_shuttle_test & ~has_cum_shuttle).alias("is_ordinal_only")

    formatted_expr = (
        pl.when(is_shuttle_test)
        .then(pl.col("level").cast(pl.String) + pl.lit(":") + pl.col("shuttle").cast(pl.String))
        .otherwise(
            pl.col("value").map_elements(_format_number, return_dtype=pl.String)
        )
        .alias("formatted_score")
    )

    period_expr = (
        pl.when(pl.col("test_date").is_not_null())
        .then(pl.col("test_date").dt.strftime("%d/%m/%Y"))
        .otherwise(
            pl.col("season_year").cast(pl.String)
            + pl.lit(" Sem ")
            + pl.col("semester").cast(pl.String)
        )
        .alias("period_label")
    )

    return (
        df.with_columns(
            pl.col("test_date").is_null().alias("is_legacy"),
            period_expr,
            comparable_expr,
            ordinal_only_expr,
            formatted_expr,
            age_expr,
        )
        .select(list(_ENRICHED_SCHEMA.keys()))
        .sort(
            ["season_year", "semester", "test_date", "result_id"],
            nulls_last=[False, False, False, False],
        )
    )


def select_by_basis(df: pl.DataFrame, basis: Basis = "latest") -> pl.DataFrame:
    """Collapse multiple results per `(student_id, test_id)` to 1 row using `basis`."""
    _check_basis(basis)
    if df.is_empty():
        return df

    ordered = df.sort(
        ["student_id", "test_id", "season_year", "semester", "test_date", "result_id"],
        nulls_last=[False, False, False, False, False, False],
    )
    if basis == "latest":
        return ordered.group_by(["student_id", "test_id"], maintain_order=True).last()
    if basis == "first":
        return ordered.group_by(["student_id", "test_id"], maintain_order=True).first()

    # basis == "best": direction-aware max/min; neutral uses latest chronologically
    eff_expr = (
        pl.when(pl.col("score_direction") == ScoreDirection.HIGHER.value)
        .then(pl.col("comparable_value"))
        .when(pl.col("score_direction") == ScoreDirection.LOWER.value)
        .then(-pl.col("comparable_value"))
        .otherwise(0.0)
        .alias("_eff")
    )
    return (
        ordered.with_columns(eff_expr)
        .sort(
            ["student_id", "test_id", "_eff", "season_year", "semester", "test_date", "result_id"],
            nulls_last=[False, False, False, False, False, False, False],
        )
        .group_by(["student_id", "test_id"], maintain_order=True)
        .last()
        .drop("_eff")
    )


# ---- 2. Athlete Trends & Personal Bests --------------------------------------------
_TRENDS_SCHEMA: dict[str, pl.DataType] = {
    "result_id": pl.Int64,
    "student_id": pl.String,
    "test_id": pl.Int64,
    "test_name": pl.String,
    "category": pl.String,
    "unit": pl.String,
    "score_direction": pl.String,
    "season_year": pl.Int64,
    "semester": pl.Int64,
    "test_date": pl.Date,
    "is_legacy": pl.Boolean,
    "period_label": pl.String,
    "age_at_test": pl.Int64,
    "comparable_value": pl.Float64,
    "formatted_score": pl.String,
    "is_ordinal_only": pl.Boolean,
    "attempt_number": pl.Int64,
    "prev_value": pl.Float64,
    "delta_from_prev": pl.Float64,
    "pct_from_prev": pl.Float64,
    "improvement_pct": pl.Float64,
    "baseline_value": pl.Float64,
    "delta_from_baseline": pl.Float64,
    "pct_from_baseline": pl.Float64,
    "change_status": pl.String,
    "is_pb": pl.Boolean,
    "chronology_uncertain": pl.Boolean,
}


def athlete_trends(
    enriched_df: pl.DataFrame,
    student_id: str,
    *,
    test_id: int | None = None,
    category: str | None = None,
) -> pl.DataFrame:
    """Chronological trend series for one athlete (raw change only — coaches judge significance)."""
    sid = clean_student_id(student_id)
    df = enriched_df.filter(pl.col("student_id") == sid)
    if test_id is not None:
        df = df.filter(pl.col("test_id") == test_id)
    if category is not None:
        df = df.filter(pl.col("category") == category)
    if df.is_empty():
        return pl.DataFrame(schema=_TRENDS_SCHEMA)

    df = df.sort(
        ["test_id", "season_year", "semester", "test_date", "result_id"],
        nulls_last=[False, False, False, False, False],
    )

    # Detect mixed legacy + dated rows in the same (test_id, season_year, semester)
    df = df.with_columns(
        (pl.col("is_legacy").n_unique().over(["test_id", "season_year", "semester"]) > 1).alias(
            "chronology_uncertain"
        )
    )

    # Compute per-test sequence metrics
    df = df.with_columns(
        pl.int_range(1, pl.len() + 1).over("test_id").cast(pl.Int64).alias("attempt_number"),
        pl.col("comparable_value").shift(1).over("test_id").alias("prev_value"),
        pl.col("comparable_value").first().over("test_id").alias("baseline_value"),
    )

    is_ordinal = pl.col("is_ordinal_only")
    is_neutral = pl.col("score_direction") == ScoreDirection.NEUTRAL.value
    is_higher = pl.col("score_direction") == ScoreDirection.HIGHER.value
    is_lower = pl.col("score_direction") == ScoreDirection.LOWER.value

    delta_prev = pl.when(is_ordinal).then(None).otherwise(
        pl.col("comparable_value") - pl.col("prev_value")
    )
    delta_base = pl.when(is_ordinal).then(None).otherwise(
        pl.col("comparable_value") - pl.col("baseline_value")
    )

    pct_prev = (
        pl.when(is_ordinal | pl.col("prev_value").is_null() | (pl.col("prev_value") == 0.0))
        .then(None)
        .otherwise((pl.col("comparable_value") - pl.col("prev_value")) / pl.col("prev_value").abs() * 100.0)
    )
    pct_base = (
        pl.when(is_ordinal | pl.col("baseline_value").is_null() | (pl.col("baseline_value") == 0.0))
        .then(None)
        .otherwise((pl.col("comparable_value") - pl.col("baseline_value")) / pl.col("baseline_value").abs() * 100.0)
    )

    improvement_pct = (
        pl.when(is_neutral | pct_prev.is_null())
        .then(None)
        .when(is_higher)
        .then(pct_prev)
        .when(is_lower)
        .then(-pct_prev)
        .otherwise(None)
    )

    # Raw directional change: positive = Improved, 0 = Unchanged, negative = Declined
    raw_dir_diff = (
        pl.when(pl.col("prev_value").is_null() | is_neutral)
        .then(None)
        .when(is_higher)
        .then(pl.col("comparable_value") - pl.col("prev_value"))
        .when(is_lower)
        .then(pl.col("prev_value") - pl.col("comparable_value"))
        .otherwise(None)
    )
    change_status = (
        pl.when(raw_dir_diff.is_null())
        .then(None)
        .when(raw_dir_diff > 0.0)
        .then(pl.lit("Improved"))
        .when(raw_dir_diff < 0.0)
        .then(pl.lit("Declined"))
        .otherwise(pl.lit("Unchanged"))
    )

    best_in_test = (
        pl.when(is_higher)
        .then(pl.col("comparable_value").max().over("test_id"))
        .when(is_lower)
        .then(pl.col("comparable_value").min().over("test_id"))
        .otherwise(None)
    )
    is_pb = (
        pl.when(is_neutral)
        .then(None)
        .otherwise(pl.col("comparable_value") == best_in_test)
    )

    return df.with_columns(
        delta_prev.cast(pl.Float64).alias("delta_from_prev"),
        pct_prev.cast(pl.Float64).alias("pct_from_prev"),
        improvement_pct.cast(pl.Float64).alias("improvement_pct"),
        delta_base.cast(pl.Float64).alias("delta_from_baseline"),
        pct_base.cast(pl.Float64).alias("pct_from_baseline"),
        change_status.cast(pl.String).alias("change_status"),
        is_pb.cast(pl.Boolean).alias("is_pb"),
    ).select(list(_TRENDS_SCHEMA.keys()))


_PB_SCHEMA: dict[str, pl.DataType] = {
    "test_id": pl.Int64,
    "test_name": pl.String,
    "category": pl.String,
    "unit": pl.String,
    "score_direction": pl.String,
    "attempt_count": pl.Int64,
    "pb_value": pl.Float64,
    "pb_formatted": pl.String,
    "pb_period": pl.String,
    "latest_value": pl.Float64,
    "latest_formatted": pl.String,
    "latest_period": pl.String,
    "delta_from_pb": pl.Float64,
    "is_latest_pb": pl.Boolean,
}


def athlete_personal_bests(
    enriched_df: pl.DataFrame,
    student_id: str,
    *,
    season_year: int | None = None,
) -> pl.DataFrame:
    """One summary row per tested `test_id` for an athlete (neutral tests have `pb_* = None`)."""
    sid = clean_student_id(student_id)
    df = enriched_df.filter(pl.col("student_id") == sid)
    if season_year is not None:
        df = df.filter(pl.col("season_year") == season_year)
    if df.is_empty():
        return pl.DataFrame(schema=_PB_SCHEMA)

    latest = select_by_basis(df, basis="latest")
    best = select_by_basis(df, basis="best")
    counts = df.group_by("test_id").agg(pl.len().cast(pl.Int64).alias("attempt_count"))

    merged = (
        latest.select(
            "test_id",
            "test_name",
            "category",
            "unit",
            "score_direction",
            "is_ordinal_only",
            pl.col("comparable_value").alias("latest_value"),
            pl.col("formatted_score").alias("latest_formatted"),
            pl.col("period_label").alias("latest_period"),
        )
        .join(
            best.select(
                "test_id",
                pl.col("comparable_value").alias("_best_val"),
                pl.col("formatted_score").alias("_best_fmt"),
                pl.col("period_label").alias("_best_period"),
            ),
            on="test_id",
            how="inner",
        )
        .join(counts, on="test_id", how="inner")
    )

    is_neutral = pl.col("score_direction") == ScoreDirection.NEUTRAL.value
    return (
        merged.with_columns(
            pl.when(is_neutral).then(None).otherwise(pl.col("_best_val")).alias("pb_value"),
            pl.when(is_neutral).then(None).otherwise(pl.col("_best_fmt")).alias("pb_formatted"),
            pl.when(is_neutral).then(None).otherwise(pl.col("_best_period")).alias("pb_period"),
            pl.when(is_neutral | pl.col("is_ordinal_only"))
            .then(None)
            .otherwise(pl.col("latest_value") - pl.col("_best_val"))
            .alias("delta_from_pb"),
            pl.when(is_neutral)
            .then(None)
            .otherwise(pl.col("latest_value") == pl.col("_best_val"))
            .alias("is_latest_pb"),
        )
        .select(list(_PB_SCHEMA.keys()))
        .sort("test_id")
    )


# ---- 3. Cohort Benchmarks ----------------------------------------------------------
_BENCHMARK_SCHEMA: dict[str, pl.DataType] = {
    "test_id": pl.Int64,
    "test_name": pl.String,
    "category": pl.String,
    "unit": pl.String,
    "score_direction": pl.String,
    "is_ordinal_only": pl.Boolean,
    "cohort_n": pl.Int64,
    "enrolled_n": pl.Int64,
    "coverage_pct": pl.Float64,
    "small_cohort": pl.Boolean,
    "cohort_label": pl.String,
    "mean": pl.Float64,
    "median": pl.Float64,
    "median_display": pl.String,
    "std": pl.Float64,
    "p10": pl.Float64,
    "p25": pl.Float64,
    "p75": pl.Float64,
    "p90": pl.Float64,
    "min_value": pl.Float64,
    "max_value": pl.Float64,
    "best_value": pl.Float64,
    "worst_value": pl.Float64,
}


def cohort_benchmarks(
    enriched_df: pl.DataFrame,
    enrolments_df: pl.DataFrame,
    students_df: pl.DataFrame,
    shuttle_rules: dict[int, dict[int, int]],
    *,
    season_year: SeasonFilter = None,
    semester: int | None = None,
    grade: str | None = None,
    sex: str | None = None,
    sport: str | None = None,
    test_id: int | None = None,
    category: str | None = None,
    basis: Basis = "latest",
    active_only: bool = False,
    min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
) -> pl.DataFrame:
    """Distribution benchmarks per test (1 score per student via `basis`, linear quantiles, ddof=1)."""
    _check_basis(basis)
    resolved_season = _resolve_season(enriched_df, season_year)
    label = _build_cohort_label(
        season_year=resolved_season,
        semester=semester,
        grade=grade,
        sex=sex,
        sport=sport,
        basis=basis,
        active_only=active_only,
    )

    df = enriched_df
    if resolved_season != "all":
        df = df.filter(pl.col("season_year") == resolved_season)
    if semester is not None:
        df = df.filter(pl.col("semester") == semester)
    if grade is not None:
        df = df.filter(pl.col("grade") == grade)
    if sex is not None:
        df = df.filter(pl.col("sex") == sex)
    if active_only:
        df = df.filter(pl.col("is_active"))
    if test_id is not None:
        df = df.filter(pl.col("test_id") == test_id)
    if category is not None:
        df = df.filter(pl.col("category") == category)
    if df.is_empty():
        return pl.DataFrame(schema=_BENCHMARK_SCHEMA)

    per_student = select_by_basis(df, basis=basis)
    enrolled_n = _count_enrolled(
        enrolments_df,
        students_df,
        season_year=resolved_season,
        sport=sport,
        grade=grade,
        sex=sex,
        active_only=active_only,
    )

    agg = (
        per_student.group_by(["test_id", "test_name", "category", "unit", "score_direction"])
        .agg(
            pl.col("is_ordinal_only").any().alias("is_ordinal_only"),
            pl.len().cast(pl.Int64).alias("cohort_n"),
            pl.col("comparable_value").mean().alias("_raw_mean"),
            pl.col("comparable_value").median().alias("median"),
            pl.col("comparable_value").std(ddof=1).alias("_raw_std"),
            pl.col("comparable_value").quantile(0.10, interpolation="linear").alias("p10"),
            pl.col("comparable_value").quantile(0.25, interpolation="linear").alias("p25"),
            pl.col("comparable_value").quantile(0.75, interpolation="linear").alias("p75"),
            pl.col("comparable_value").quantile(0.90, interpolation="linear").alias("p90"),
            pl.col("comparable_value").min().alias("min_value"),
            pl.col("comparable_value").max().alias("max_value"),
        )
        .sort("test_id")
    )

    rows_out: list[dict[str, Any]] = []
    for row in agg.iter_rows(named=True):
        tid = int(row["test_id"])
        direction = row["score_direction"]
        ordinal = bool(row["is_ordinal_only"])
        cn = int(row["cohort_n"])
        med = row["median"]

        if row["unit"] == Unit.LEVEL_SHUTTLE.value and not ordinal:
            med_disp = format_shuttle_score(tid, med, shuttle_rules)
        elif row["unit"] == Unit.LEVEL_SHUTTLE.value:
            med_disp = None
        else:
            med_disp = _format_number(med)

        if direction == ScoreDirection.HIGHER.value:
            best_v, worst_v = row["max_value"], row["min_value"]
        elif direction == ScoreDirection.LOWER.value:
            best_v, worst_v = row["min_value"], row["max_value"]
        else:
            best_v, worst_v = None, None

        rows_out.append(
            {
                "test_id": tid,
                "test_name": row["test_name"],
                "category": row["category"],
                "unit": row["unit"],
                "score_direction": direction,
                "is_ordinal_only": ordinal,
                "cohort_n": cn,
                "enrolled_n": enrolled_n,
                "coverage_pct": _coverage_pct(cn, enrolled_n),
                "small_cohort": cn < min_cohort_size,
                "cohort_label": label,
                "mean": None if ordinal else row["_raw_mean"],
                "median": med,
                "median_display": med_disp,
                "std": None if (ordinal or cn < 2) else row["_raw_std"],
                "p10": None if ordinal else row["p10"],
                "p25": None if ordinal else row["p25"],
                "p75": None if ordinal else row["p75"],
                "p90": None if ordinal else row["p90"],
                "min_value": row["min_value"],
                "max_value": row["max_value"],
                "best_value": best_v,
                "worst_value": worst_v,
            }
        )
    return pl.DataFrame(rows_out, schema=_BENCHMARK_SCHEMA)


# ---- 4. Athlete Percentiles & Category Radar Profile -------------------------------
_PERCENTILE_SCHEMA: dict[str, pl.DataType] = {
    "test_id": pl.Int64,
    "test_name": pl.String,
    "category": pl.String,
    "unit": pl.String,
    "score_direction": pl.String,
    "is_ordinal_only": pl.Boolean,
    "athlete_value": pl.Float64,
    "athlete_formatted": pl.String,
    "period_label": pl.String,
    "percentile": pl.Float64,
    "rank": pl.Int64,
    "cohort_n": pl.Int64,
    "enrolled_n": pl.Int64,
    "coverage_pct": pl.Float64,
    "cohort_mean": pl.Float64,
    "cohort_median": pl.Float64,
    "z_score": pl.Float64,
    "small_cohort": pl.Boolean,
    "cohort_label": pl.String,
}


def athlete_percentiles(
    enriched_df: pl.DataFrame,
    enrolments_df: pl.DataFrame,
    students_df: pl.DataFrame,
    student_id: str,
    *,
    season_year: SeasonFilter = None,
    semester: int | None = None,
    sport: str | None = None,
    match_grade: bool = True,
    match_sex: bool = True,
    basis: Basis = "latest",
    active_only: bool = False,
    min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
) -> pl.DataFrame:
    """Direction-aware Hazen mid-rank percentiles for one athlete.

    - Safeguarding: strictly excludes `ScoreDirection.NEUTRAL` (Height / Weight).
    - Default cohort: athlete's latest season + same Grade + same Sex (`basis="latest"`).
    - Still computes percentiles when `cohort_n < min_cohort_size` and sets `small_cohort=True`.
    - Ordered by `(test_id // 100, test_id)` so similar tests group cleanly on the
      Test-by-Test Radar Chart.
    """
    _check_basis(basis)
    sid = clean_student_id(student_id)
    non_neutral = enriched_df.filter(pl.col("score_direction") != ScoreDirection.NEUTRAL.value)
    if non_neutral.is_empty():
        return pl.DataFrame(schema=_PERCENTILE_SCHEMA)

    resolved_season = _resolve_season(non_neutral, season_year, student_id=sid)

    window = non_neutral
    if resolved_season != "all":
        window = window.filter(pl.col("season_year") == resolved_season)
    if semester is not None:
        window = window.filter(pl.col("semester") == semester)

    athlete_rows = window.filter(pl.col("student_id") == sid)
    if athlete_rows.is_empty():
        return pl.DataFrame(schema=_PERCENTILE_SCHEMA)

    # Resolve athlete's own grade & sex in the window for default cohort matching
    latest_athlete_row = athlete_rows.sort(
        ["season_year", "semester", "test_date", "result_id"],
        nulls_last=[False, False, False, False],
    ).row(-1, named=True)
    target_grade = latest_athlete_row["grade"] if match_grade else None
    target_sex = latest_athlete_row["sex"] if match_sex else None

    cohort_pool = window
    if target_grade is not None:
        cohort_pool = cohort_pool.filter(pl.col("grade") == target_grade)
    if target_sex is not None:
        cohort_pool = cohort_pool.filter(pl.col("sex") == target_sex)
    if active_only:
        cohort_pool = cohort_pool.filter(pl.col("is_active") | (pl.col("student_id") == sid))

    per_student = select_by_basis(cohort_pool, basis=basis)
    enrolled_n = _count_enrolled(
        enrolments_df,
        students_df,
        season_year=resolved_season,
        sport=sport,
        grade=target_grade,
        sex=target_sex,
        active_only=active_only,
    )
    label = _build_cohort_label(
        season_year=resolved_season,
        semester=semester,
        grade=target_grade,
        sex=target_sex,
        sport=sport,
        basis=basis,
        active_only=active_only,
    )

    eff_expr = (
        pl.when(pl.col("score_direction") == ScoreDirection.HIGHER.value)
        .then(pl.col("comparable_value"))
        .otherwise(-pl.col("comparable_value"))
        .alias("_eff")
    )

    ranked = (
        per_student.with_columns(eff_expr)
        .with_columns(
            pl.len().over("test_id").cast(pl.Int64).alias("cohort_n"),
            pl.col("_eff").rank(method="average").over("test_id").alias("_rank_asc_avg"),
            (-pl.col("_eff")).rank(method="min").over("test_id").cast(pl.Int64).alias("rank"),
            pl.col("comparable_value").mean().over("test_id").alias("_raw_mean"),
            pl.col("comparable_value").median().over("test_id").alias("cohort_median"),
            pl.col("_eff").mean().over("test_id").alias("_eff_mean"),
            pl.col("_eff").std(ddof=1).over("test_id").alias("_eff_std"),
            pl.col("is_ordinal_only").any().over("test_id").alias("_any_ordinal"),
        )
        .filter(pl.col("student_id") == sid)
    )

    if ranked.is_empty():
        return pl.DataFrame(schema=_PERCENTILE_SCHEMA)

    rows_out: list[dict[str, Any]] = []
    for row in ranked.iter_rows(named=True):
        cn = int(row["cohort_n"])
        ordinal = bool(row["_any_ordinal"])
        pct = round(((float(row["_rank_asc_avg"]) - 0.5) / cn) * 100.0, 2)
        eff_std = row["_eff_std"]
        if ordinal or cn < 2 or eff_std is None or eff_std == 0.0:
            z = None
        else:
            z = round((float(row["_eff"]) - float(row["_eff_mean"])) / float(eff_std), 3)

        rows_out.append(
            {
                "test_id": int(row["test_id"]),
                "test_name": row["test_name"],
                "category": row["category"],
                "unit": row["unit"],
                "score_direction": row["score_direction"],
                "is_ordinal_only": ordinal,
                "athlete_value": float(row["comparable_value"]),
                "athlete_formatted": row["formatted_score"],
                "period_label": row["period_label"],
                "percentile": pct,
                "rank": int(row["rank"]),
                "cohort_n": cn,
                "enrolled_n": enrolled_n,
                "coverage_pct": _coverage_pct(cn, enrolled_n),
                "cohort_mean": None if ordinal else float(row["_raw_mean"]),
                "cohort_median": float(row["cohort_median"]),
                "z_score": z,
                "small_cohort": cn < min_cohort_size,
                "cohort_label": label,
            }
        )

    return pl.DataFrame(rows_out, schema=_PERCENTILE_SCHEMA).sort(
        [pl.col("test_id") // IDS_PER_CATEGORY, pl.col("test_id")]
    )


_CATEGORY_PROFILE_SCHEMA: dict[str, pl.DataType] = {
    "category": pl.String,
    "mean_z_score": pl.Float64,
    "composite_index": pl.Float64,
    "representative_test_id": pl.Int64,
    "representative_test_name": pl.String,
    "representative_percentile": pl.Float64,
    "tests_completed": pl.Int64,
    "tests_in_category": pl.Int64,
    "min_cohort_n": pl.Int64,
    "small_cohort": pl.Boolean,
    "cohort_label": pl.String,
}


def athlete_category_profile(
    percentiles_df: pl.DataFrame,
    tests_df: pl.DataFrame,
    *,
    primary_tests: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Aggregate an athlete's non-neutral test percentiles by category for the Category Radar.

    Returns BOTH:
      - `mean_z_score` + `composite_index` (0-100 normal CDF of mean z-score — avoids
        rank compression toward 50).
      - `representative_test_name` + `representative_percentile` (single headline test per
        category, configurable via `primary_tests`).
    """
    if percentiles_df.is_empty():
        return pl.DataFrame(schema=_CATEGORY_PROFILE_SCHEMA)

    primaries = PRIMARY_CATEGORY_TESTS if primary_tests is None else primary_tests
    tests_norm = tests_df.select(
        pl.col("test_id").cast(pl.Int64),
        _str_col("category").alias("category"),
        _str_col("score_direction").alias("score_direction"),
    ).filter(pl.col("score_direction") != ScoreDirection.NEUTRAL.value)
    cat_totals = dict(
        tests_norm.group_by("category").agg(pl.len().cast(pl.Int64)).iter_rows()
    )
    cat_block = dict(
        tests_norm.group_by("category")
        .agg((pl.col("test_id").min() // IDS_PER_CATEGORY).alias("blk"))
        .iter_rows()
    )

    label = str(percentiles_df["cohort_label"][0])
    rows_out: list[tuple[int, dict[str, Any]]] = []
    for (cat,), grp in percentiles_df.group_by(["category"], maintain_order=True):
        cat_str = str(cat)
        z_vals = [float(z) for z in grp["z_score"].to_list() if z is not None]
        mean_z = round(sum(z_vals) / len(z_vals), 3) if z_vals else None
        comp_idx = _normal_cdf_100(mean_z)

        preferred_tid = primaries.get(cat_str)
        rep_row = None
        if preferred_tid is not None:
            matched = grp.filter(pl.col("test_id") == preferred_tid)
            if not matched.is_empty():
                rep_row = matched.row(0, named=True)
        if rep_row is None:
            rep_row = grp.sort("test_id").row(0, named=True)

        min_cn = int(grp["cohort_n"].min())  # type: ignore[arg-type]
        any_small = bool(grp["small_cohort"].any())
        rows_out.append(
            (
                int(cat_block.get(cat_str, 999)),
                {
                    "category": cat_str,
                    "mean_z_score": mean_z,
                    "composite_index": comp_idx,
                    "representative_test_id": int(rep_row["test_id"]),
                    "representative_test_name": rep_row["test_name"],
                    "representative_percentile": float(rep_row["percentile"]),
                    "tests_completed": int(grp.height),
                    "tests_in_category": int(cat_totals.get(cat_str, grp.height)),
                    "min_cohort_n": min_cn,
                    "small_cohort": any_small,
                    "cohort_label": label,
                },
            )
        )

    rows_out.sort(key=lambda item: (item[0], item[1]["category"]))
    return pl.DataFrame([r for _, r in rows_out], schema=_CATEGORY_PROFILE_SCHEMA)


# ---- 5. Left / Right Bilateral Asymmetry -------------------------------------------
_ASYMMETRY_SCHEMA: dict[str, pl.DataType] = {
    "student_id": pl.String,
    "full_name": pl.String,
    "grade": pl.String,
    "sports": pl.String,
    "pair_name": pl.String,
    "left_test_id": pl.Int64,
    "right_test_id": pl.Int64,
    "unit": pl.String,
    "score_direction": pl.String,
    "season_year": pl.Int64,
    "semester": pl.Int64,
    "test_date": pl.Date,
    "period_label": pl.String,
    "left_value": pl.Float64,
    "right_value": pl.Float64,
    "asymmetry_pct": pl.Float64,
    "better_side": pl.String,
    "moderate_threshold_pct": pl.Float64,
    "elevated_threshold_pct": pl.Float64,
    "asymmetry_flag": pl.String,
}


def discover_lr_pairs(tests_df: pl.DataFrame) -> list[tuple[str, int, int, str, str]]:
    """Auto-discover all `(pair_name, left_test_id, right_test_id, unit, score_direction)` pairs."""
    if tests_df.is_empty():
        return []
    lefts: dict[str, tuple[int, str, str]] = {}
    rights: dict[str, tuple[int, str, str]] = {}
    for row in tests_df.select(
        pl.col("test_id").cast(pl.Int64),
        pl.col("test_name").cast(pl.String),
        _str_col("unit").alias("unit"),
        _str_col("score_direction").alias("score_direction"),
    ).iter_rows(named=True):
        name = row["test_name"]
        if name.endswith(" Left"):
            base = name[: -len(" Left")].strip()
            lefts[base] = (int(row["test_id"]), row["unit"], row["score_direction"])
        elif name.endswith(" Right"):
            base = name[: -len(" Right")].strip()
            rights[base] = (int(row["test_id"]), row["unit"], row["score_direction"])

    pairs: list[tuple[str, int, int, str, str]] = []
    for base in sorted(lefts.keys() & rights.keys()):
        ltid, lunit, ldir = lefts[base]
        rtid, runit, rdir = rights[base]
        if lunit == runit and ldir == rdir:
            pairs.append((base, ltid, rtid, lunit, ldir))
    return pairs


def asymmetry_report(
    enriched_df: pl.DataFrame,
    tests_df: pl.DataFrame,
    *,
    student_id: str | None = None,
    season_year: SeasonFilter = "all",
    semester: int | None = None,
    grade: str | None = None,
    thresholds_by_test: dict[str, tuple[float, float]] | None = None,
    thresholds_by_unit: dict[str, tuple[float, float]] | None = None,
) -> pl.DataFrame:
    """Same-session Left/Right asymmetry screening using `(max - min) / max * 100`.

    Per-unit threshold defaults:
      - `Seconds` (5-0-5 Agility COD): 5% Moderate / 10% Elevated
      - `Newtons` & other units (Nordic, Adduction, Abduction): 10% Moderate / 15% Elevated
    """
    pairs = discover_lr_pairs(tests_df)
    if not pairs or enriched_df.is_empty():
        return pl.DataFrame(schema=_ASYMMETRY_SCHEMA)

    df = enriched_df
    if student_id is not None:
        df = df.filter(pl.col("student_id") == clean_student_id(student_id))
    resolved_season = _resolve_season(df, season_year)
    if resolved_season != "all":
        df = df.filter(pl.col("season_year") == resolved_season)
    if semester is not None:
        df = df.filter(pl.col("semester") == semester)
    if grade is not None:
        df = df.filter(pl.col("grade") == grade)
    if df.is_empty():
        return pl.DataFrame(schema=_ASYMMETRY_SCHEMA)

    unit_thresholds = thresholds_by_unit or ASYMMETRY_THRESHOLDS_BY_UNIT
    test_thresholds = thresholds_by_test or {}

    # Session key prevents cross-pairing dated and legacy rows and guards against duplicates
    session_cols = ["student_id", "season_year", "semester", "is_legacy", "test_date"]

    rows_out: list[dict[str, Any]] = []
    for pair_name, ltid, rtid, unit, direction in pairs:
        mod_thr, elev_thr = test_thresholds.get(
            pair_name, unit_thresholds.get(unit, DEFAULT_ASYMMETRY_THRESHOLDS)
        )
        left_df = (
            df.filter(pl.col("test_id") == ltid)
            .group_by(session_cols)
            .agg(
                pl.col("full_name").first(),
                pl.col("grade").first(),
                pl.col("sports").first(),
                pl.col("period_label").first(),
                pl.col("comparable_value").mean().alias("left_value"),
            )
        )
        right_df = (
            df.filter(pl.col("test_id") == rtid)
            .group_by(session_cols)
            .agg(pl.col("comparable_value").mean().alias("right_value"))
        )
        if left_df.is_empty() or right_df.is_empty():
            continue

        joined = left_df.join(right_df, on=session_cols, how="inner", nulls_equal=True)
        for row in joined.iter_rows(named=True):
            lv = float(row["left_value"])
            rv = float(row["right_value"])
            hi = max(lv, rv)
            lo = min(lv, rv)
            asym = 0.0 if hi <= 0.0 else round(((hi - lo) / hi) * 100.0, 2)

            if lv == rv:
                better = "Balanced"
            elif direction == ScoreDirection.LOWER.value:
                better = "Left" if lv < rv else "Right"
            else:
                better = "Left" if lv > rv else "Right"

            if asym >= elev_thr:
                flag = "Elevated"
            elif asym >= mod_thr:
                flag = "Moderate"
            else:
                flag = "Normal"

            rows_out.append(
                {
                    "student_id": row["student_id"],
                    "full_name": row["full_name"],
                    "grade": row["grade"],
                    "sports": row["sports"],
                    "pair_name": pair_name,
                    "left_test_id": ltid,
                    "right_test_id": rtid,
                    "unit": unit,
                    "score_direction": direction,
                    "season_year": int(row["season_year"]),
                    "semester": int(row["semester"]),
                    "test_date": row["test_date"],
                    "period_label": row["period_label"],
                    "left_value": lv,
                    "right_value": rv,
                    "asymmetry_pct": asym,
                    "better_side": better,
                    "moderate_threshold_pct": float(mod_thr),
                    "elevated_threshold_pct": float(elev_thr),
                    "asymmetry_flag": flag,
                }
            )

    if not rows_out:
        return pl.DataFrame(schema=_ASYMMETRY_SCHEMA)
    return pl.DataFrame(rows_out, schema=_ASYMMETRY_SCHEMA).sort(
        ["asymmetry_pct", "season_year", "semester", "student_id"],
        descending=[True, True, True, False],
    )


# ---- 6. Relative Strength Metrics --------------------------------------------------
_REL_STRENGTH_SCHEMA: dict[str, pl.DataType] = {
    "result_id": pl.Int64,
    "student_id": pl.String,
    "full_name": pl.String,
    "grade": pl.String,
    "sports": pl.String,
    "test_id": pl.Int64,
    "test_name": pl.String,
    "season_year": pl.Int64,
    "semester": pl.Int64,
    "period_label": pl.String,
    "absolute_value": pl.Float64,
    "absolute_unit": pl.String,
    "weight_kg": pl.Float64,
    "weight_period_label": pl.String,
    "same_session_weight": pl.Boolean,
    "relative_value": pl.Float64,
    "relative_unit": pl.String,
    "imtp_bodyweight_multiples": pl.Float64,
}


def relative_strength_metrics(
    enriched_df: pl.DataFrame,
    *,
    student_id: str | None = None,
    season_year: SeasonFilter = "all",
) -> pl.DataFrame:
    """Pair absolute strength lifts (IMTP Peak, 1RM/3RM) with same-session or same-season Weight."""
    if enriched_df.is_empty():
        return pl.DataFrame(schema=_REL_STRENGTH_SCHEMA)

    df = enriched_df
    if student_id is not None:
        df = df.filter(pl.col("student_id") == clean_student_id(student_id))
    resolved_season = _resolve_season(df, season_year)
    if resolved_season != "all":
        df = df.filter(pl.col("season_year") == resolved_season)

    strength_df = df.filter(pl.col("test_id").is_in(list(RELATIVE_STRENGTH_TEST_IDS.keys())))
    if strength_df.is_empty():
        return pl.DataFrame(schema=_REL_STRENGTH_SCHEMA)

    weights_df = df.filter(pl.col("test_id") == WEIGHT_ID)
    # Index weights by (student_id, season_year)
    weights_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for wrow in weights_df.iter_rows(named=True):
        key = (wrow["student_id"], int(wrow["season_year"]))
        weights_by_key.setdefault(key, []).append(wrow)

    rows_out: list[dict[str, Any]] = []
    for row in strength_df.iter_rows(named=True):
        sid = row["student_id"]
        syear = int(row["season_year"])
        tid = int(row["test_id"])
        abs_val = float(row["comparable_value"])
        rel_unit = RELATIVE_STRENGTH_TEST_IDS[tid]

        candidates = weights_by_key.get((sid, syear), [])
        matched_w: dict[str, Any] | None = None
        same_session = False

        # 1) Exact same session first
        for w in candidates:
            if row["test_date"] is not None and w["test_date"] == row["test_date"]:
                matched_w = w
                same_session = True
                break
            if (
                row["test_date"] is None
                and w["test_date"] is None
                and w["semester"] == row["semester"]
            ):
                matched_w = w
                same_session = True
                break

        # 2) Fallback: nearest weight in same season_year
        if matched_w is None and candidates:
            if row["test_date"] is not None:
                dated_cands = [w for w in candidates if w["test_date"] is not None]
                if dated_cands:
                    matched_w = min(
                        dated_cands, key=lambda w: abs((w["test_date"] - row["test_date"]).days)
                    )
                else:
                    matched_w = min(
                        candidates, key=lambda w: abs(int(w["semester"]) - int(row["semester"]))
                    )
            else:
                matched_w = min(
                    candidates, key=lambda w: abs(int(w["semester"]) - int(row["semester"]))
                )

        if matched_w is not None and matched_w["comparable_value"] > 0.0:
            w_kg = float(matched_w["comparable_value"])
            w_period = str(matched_w["period_label"])
            rel_val = round(abs_val / w_kg, 3)
            bw_mult = round(abs_val / (w_kg * 9.81), 3) if tid == IMTP_PEAK_ID else None
        else:
            w_kg = None
            w_period = None
            rel_val = None
            bw_mult = None

        rows_out.append(
            {
                "result_id": int(row["result_id"]),
                "student_id": sid,
                "full_name": row["full_name"],
                "grade": row["grade"],
                "sports": row["sports"],
                "test_id": tid,
                "test_name": row["test_name"],
                "season_year": syear,
                "semester": int(row["semester"]),
                "period_label": row["period_label"],
                "absolute_value": abs_val,
                "absolute_unit": row["unit"],
                "weight_kg": w_kg,
                "weight_period_label": w_period,
                "same_session_weight": same_session if matched_w is not None else False,
                "relative_value": rel_val,
                "relative_unit": rel_unit,
                "imtp_bodyweight_multiples": bw_mult,
            }
        )

    return pl.DataFrame(rows_out, schema=_REL_STRENGTH_SCHEMA).sort(
        ["season_year", "semester", "student_id", "test_id"]
    )


# ---- 7. Squad Leaderboard, Head-to-Head Comparison & Dashboard Overview ------------
_LEADERBOARD_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "student_id": pl.String,
    "full_name": pl.String,
    "grade": pl.String,
    "sports": pl.String,
    "sex": pl.String,
    "comparable_value": pl.Float64,
    "formatted_score": pl.String,
    "period_label": pl.String,
    "percentile": pl.Float64,
    "cohort_n": pl.Int64,
    "enrolled_n": pl.Int64,
    "coverage_pct": pl.Float64,
    "small_cohort": pl.Boolean,
    "cohort_label": pl.String,
}


def squad_leaderboard(
    enriched_df: pl.DataFrame,
    enrolments_df: pl.DataFrame,
    students_df: pl.DataFrame,
    test_id: int,
    *,
    tests_df: pl.DataFrame | None = None,
    season_year: SeasonFilter = None,
    semester: int | None = None,
    grade: str | None = None,
    sex: str | None = None,
    sport: str | None = None,
    basis: Basis = "latest",
    active_only: bool = False,
    min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
    limit: int | None = None,
) -> pl.DataFrame:
    """Direction-aware squad leaderboard for one test (strictly forbidden for neutral tests)."""
    _check_basis(basis)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValidationError("limit must be a positive whole number")

    # Safeguarding check against tests_df (even when there are 0 results)
    if tests_df is not None and not tests_df.is_empty():
        tmatch = tests_df.filter(pl.col("test_id") == test_id)
        if not tmatch.is_empty() and str(tmatch["score_direction"][0]) == ScoreDirection.NEUTRAL.value:
            raise ValidationError(
                "Leaderboards are not available for neutral anthropometric measurements (Height/Weight)"
            )

    df = enriched_df.filter(pl.col("test_id") == test_id)
    if not df.is_empty() and str(df["score_direction"][0]) == ScoreDirection.NEUTRAL.value:
        raise ValidationError(
            "Leaderboards are not available for neutral anthropometric measurements (Height/Weight)"
        )

    resolved_season = _resolve_season(enriched_df, season_year)
    if resolved_season != "all":
        df = df.filter(pl.col("season_year") == resolved_season)
    if semester is not None:
        df = df.filter(pl.col("semester") == semester)
    if grade is not None:
        df = df.filter(pl.col("grade") == grade)
    if sex is not None:
        df = df.filter(pl.col("sex") == sex)
    if active_only:
        df = df.filter(pl.col("is_active"))
    if df.is_empty():
        return pl.DataFrame(schema=_LEADERBOARD_SCHEMA)

    per_student = select_by_basis(df, basis=basis)
    cn = int(per_student.height)
    enrolled_n = _count_enrolled(
        enrolments_df,
        students_df,
        season_year=resolved_season,
        sport=sport,
        grade=grade,
        sex=sex,
        active_only=active_only,
    )
    cov = _coverage_pct(cn, enrolled_n)
    label = _build_cohort_label(
        season_year=resolved_season,
        semester=semester,
        grade=grade,
        sex=sex,
        sport=sport,
        basis=basis,
        active_only=active_only,
    )

    direction = str(per_student["score_direction"][0])
    eff_expr = (
        pl.col("comparable_value")
        if direction == ScoreDirection.HIGHER.value
        else -pl.col("comparable_value")
    )

    ranked = (
        per_student.with_columns(eff_expr.alias("_eff"))
        .with_columns(
            (-pl.col("_eff")).rank(method="min").cast(pl.Int64).alias("rank"),
            (((pl.col("_eff").rank(method="average") - 0.5) / float(cn)) * 100.0)
            .round(2)
            .alias("percentile"),
            pl.lit(cn, dtype=pl.Int64).alias("cohort_n"),
            pl.lit(enrolled_n, dtype=pl.Int64).alias("enrolled_n"),
            pl.lit(cov, dtype=pl.Float64).alias("coverage_pct"),
            pl.lit(cn < min_cohort_size, dtype=pl.Boolean).alias("small_cohort"),
            pl.lit(label, dtype=pl.String).alias("cohort_label"),
        )
        .sort(["rank", "full_name", "student_id"])
        .select(list(_LEADERBOARD_SCHEMA.keys()))
    )
    return ranked.head(limit) if limit is not None else ranked


_COMPARE_SCHEMA: dict[str, pl.DataType] = {
    "student_id": pl.String,
    "full_name": pl.String,
    "grade": pl.String,
    "sports": pl.String,
    "sex": pl.String,
    "test_id": pl.Int64,
    "test_name": pl.String,
    "category": pl.String,
    "unit": pl.String,
    "score_direction": pl.String,
    "comparable_value": pl.Float64,
    "formatted_score": pl.String,
    "period_label": pl.String,
    "percentile": pl.Float64,
    "cohort_n": pl.Int64,
    "small_cohort": pl.Boolean,
    "cohort_label": pl.String,
}


def compare_athletes(
    enriched_df: pl.DataFrame,
    enrolments_df: pl.DataFrame,
    students_df: pl.DataFrame,
    student_ids: list[str],
    *,
    season_year: SeasonFilter = None,
    semester: int | None = None,
    sport: str | None = None,
    test_ids: list[int] | None = None,
    category: str | None = None,
    basis: Basis = "latest",
    match_grade: bool = False,
    match_sex: bool = False,
    active_only: bool = False,
    min_cohort_size: int = DEFAULT_MIN_COHORT_SIZE,
) -> pl.DataFrame:
    """Head-to-head comparison across 2+ athletes (includes each athlete's cohort percentile)."""
    _check_basis(basis)
    clean_ids = [clean_student_id(s) for s in student_ids]
    if not clean_ids or enriched_df.is_empty():
        return pl.DataFrame(schema=_COMPARE_SCHEMA)

    resolved_season = _resolve_season(enriched_df, season_year)
    window = enriched_df
    if resolved_season != "all":
        window = window.filter(pl.col("season_year") == resolved_season)
    if semester is not None:
        window = window.filter(pl.col("semester") == semester)
    if test_ids is not None:
        window = window.filter(pl.col("test_id").is_in(test_ids))
    if category is not None:
        window = window.filter(pl.col("category") == category)

    selected = select_by_basis(
        window.filter(pl.col("student_id").is_in(clean_ids)), basis=basis
    )
    if selected.is_empty():
        return pl.DataFrame(schema=_COMPARE_SCHEMA)

    # Compute percentiles per athlete (neutral tests naturally get percentile=None)
    pct_frames: list[pl.DataFrame] = []
    for sid in dict.fromkeys(clean_ids):
        p = athlete_percentiles(
            enriched_df,
            enrolments_df,
            students_df,
            sid,
            season_year=resolved_season,
            semester=semester,
            sport=sport,
            match_grade=match_grade,
            match_sex=match_sex,
            basis=basis,
            active_only=active_only,
            min_cohort_size=min_cohort_size,
        )
        if not p.is_empty():
            pct_frames.append(
                p.select(
                    pl.lit(sid).alias("student_id"),
                    "test_id",
                    "percentile",
                    "cohort_n",
                    "small_cohort",
                    "cohort_label",
                )
            )

    if pct_frames:
        pct_all = pl.concat(pct_frames)
        merged = selected.join(pct_all, on=["student_id", "test_id"], how="left")
    else:
        merged = selected.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("percentile"),
            pl.lit(None, dtype=pl.Int64).alias("cohort_n"),
            pl.lit(None, dtype=pl.Boolean).alias("small_cohort"),
            pl.lit(None, dtype=pl.String).alias("cohort_label"),
        )

    return merged.select(list(_COMPARE_SCHEMA.keys())).sort(["test_id", "student_id"])


def dashboard_overview(
    students_df: pl.DataFrame,
    enrolments_df: pl.DataFrame,
    results_df: pl.DataFrame,
    *,
    season_year: int | None = None,
    sport: str | None = None,
    grade: str | None = None,
    active_only: bool = False,
) -> dict[str, Any]:
    """High-level KPI metrics for the dashboard home page (distinguishes tested vs untested)."""
    studs = students_df.select(
        pl.col("student_id").cast(pl.String),
        pl.col("is_active").cast(pl.Boolean),
    )
    if active_only:
        studs = studs.filter(pl.col("is_active"))

    enrol = enrolments_df.select(
        pl.col("student_id").cast(pl.String),
        pl.col("season_year").cast(pl.Int64),
        _str_col("sport").alias("sport"),
        _str_col("grade").alias("grade"),
    ).join(studs.select("student_id"), on="student_id", how="semi")

    seasons_available = sorted(int(y) for y in enrol["season_year"].unique().to_list()) if not enrol.is_empty() else []
    sports_available = sorted(str(s) for s in enrol["sport"].unique().to_list()) if not enrol.is_empty() else []
    grades_available = sorted(str(g) for g in enrol["grade"].unique().to_list()) if not enrol.is_empty() else []

    filtered_enrol = enrol
    if season_year is not None:
        filtered_enrol = filtered_enrol.filter(pl.col("season_year") == season_year)
    if sport is not None:
        filtered_enrol = filtered_enrol.filter(pl.col("sport").str.to_lowercase() == sport.strip().casefold())
    if grade is not None:
        filtered_enrol = filtered_enrol.filter(pl.col("grade") == grade)

    if season_year is not None or sport is not None or grade is not None:
        eligible_ids = filtered_enrol.select("student_id").unique()
    else:
        eligible_ids = studs.select("student_id").unique()

    res = results_df.select(
        pl.col("student_id").cast(pl.String),
        pl.col("test_id").cast(pl.Int64),
        pl.col("season_year").cast(pl.Int64),
    ).join(eligible_ids, on="student_id", how="semi")
    if season_year is not None:
        res = res.filter(pl.col("season_year") == season_year)

    total_students = int(eligible_ids.height)
    tested_students = int(res["student_id"].n_unique()) if not res.is_empty() else 0
    untested_students = max(0, total_students - tested_students)

    return {
        "total_students": total_students,
        "tested_students": tested_students,
        "untested_students": untested_students,
        "coverage_pct": _coverage_pct(tested_students, total_students),
        "total_results": int(res.height),
        "distinct_tests": int(res["test_id"].n_unique()) if not res.is_empty() else 0,
        "seasons_available": seasons_available,
        "sports_available": sports_available,
        "grades_available": grades_available,
    }
