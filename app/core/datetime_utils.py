from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


DATETIME_NAME_HINTS = (
    "date",
    "time",
    "datetime",
    "timestamp",
    "year",
    "month",
    "quarter",
    "week",
    "日期",
    "时间",
    "年份",
    "年度",
    "年月",
    "月份",
    "季度",
    "周",
)


def has_datetime_hint(column_name: str) -> bool:
    normalized = str(column_name).strip().lower()
    return (
        any(hint in normalized for hint in DATETIME_NAME_HINTS)
        or normalized in {"年", "月", "日"}
        or normalized.endswith("_year")
        or normalized.endswith("_month")
        or normalized.endswith("_date")
        or normalized.endswith("_time")
    )


def parse_datetime_series(series: pd.Series, column_name: str | None = None) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    text = _normalize_text_series(series)
    valid_text = text.dropna()
    if valid_text.empty:
        return pd.to_datetime(text, errors="coerce")

    epoch_parsed = _parse_epoch_series(text, column_name)
    if epoch_parsed.notna().mean() >= 0.9:
        return epoch_parsed

    year_parsed = _parse_year_series(text, column_name)
    if year_parsed.notna().mean() >= 0.9:
        return year_parsed

    quarter_parsed = _parse_quarter_series(text, column_name)
    if quarter_parsed.notna().mean() >= 0.9:
        return quarter_parsed

    year_month_parsed = _parse_year_month_series(text, column_name)
    if year_month_parsed.notna().mean() >= 0.9:
        return year_month_parsed

    compact_date_parsed = _parse_compact_date_series(text, column_name)
    if compact_date_parsed.notna().mean() >= 0.9:
        return compact_date_parsed

    fallback = pd.to_datetime(text, errors="coerce", format="mixed")
    return fallback


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "NULL": pd.NA,
                "null": pd.NA,
                "NaT": pd.NA,
            }
        )
    )


def _parse_year_series(text: pd.Series, column_name: str | None) -> pd.Series:
    numeric = pd.to_numeric(text, errors="coerce")
    four_digit_ratio = text.dropna().str.fullmatch(r"\d{4}").mean() if not text.dropna().empty else 0.0
    year_range_ratio = (
        numeric.dropna().between(1800, 2200).mean() if not numeric.dropna().empty else 0.0
    )
    if _has_year_hint(column_name) or (four_digit_ratio >= 0.9 and year_range_ratio >= 0.9):
        return pd.to_datetime(text, format="%Y", errors="coerce")
    return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")


def _parse_year_month_series(text: pd.Series, column_name: str | None) -> pd.Series:
    compact = text.str.replace(r"[^\d]", "", regex=True)
    valid_compact = compact.dropna()
    if valid_compact.empty:
        return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")

    if _has_month_hint(column_name) or valid_compact.str.len().eq(6).mean() >= 0.9:
        return pd.to_datetime(compact, format="%Y%m", errors="coerce")
    return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")


def _parse_compact_date_series(text: pd.Series, column_name: str | None) -> pd.Series:
    compact = text.str.replace(r"[^\d]", "", regex=True)
    valid_compact = compact.dropna()
    if valid_compact.empty:
        return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")

    if _has_day_hint(column_name) or valid_compact.str.len().eq(8).mean() >= 0.9:
        return pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")


def _parse_quarter_series(text: pd.Series, column_name: str | None) -> pd.Series:
    normalized = (
        text.str.upper()
        .str.replace("年", "", regex=False)
        .str.replace("第", "", regex=False)
        .str.replace("季度", "Q", regex=False)
        .str.replace("季", "Q", regex=False)
        .str.replace(" ", "", regex=False)
    )
    normalized = normalized.str.replace(r"(\d{4})[-/]?Q([1-4])", r"\1Q\2", regex=True)
    mask = normalized.str.fullmatch(r"\d{4}Q[1-4]")

    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")
    if (_has_quarter_hint(column_name) or mask.mean() >= 0.9) and mask.any():
        parsed.loc[mask] = pd.PeriodIndex(normalized[mask], freq="Q").to_timestamp(how="start")
    return parsed


def _parse_epoch_series(text: pd.Series, column_name: str | None) -> pd.Series:
    if not _has_epoch_hint(column_name):
        return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")

    valid_text = text.dropna()
    if valid_text.empty:
        return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")

    if valid_text.str.fullmatch(r"\d{10}").mean() >= 0.9:
        numeric = pd.to_numeric(text, errors="coerce")
        return pd.to_datetime(numeric, unit="s", errors="coerce")

    if valid_text.str.fullmatch(r"\d{13}").mean() >= 0.9:
        numeric = pd.to_numeric(text, errors="coerce")
        return pd.to_datetime(numeric, unit="ms", errors="coerce")

    return pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")


def _has_year_hint(column_name: str | None) -> bool:
    normalized = str(column_name or "").strip().lower()
    return (
        "year" in normalized
        or "年份" in normalized
        or "年度" in normalized
        or normalized == "年"
        or normalized.endswith("_year")
    )


def _has_month_hint(column_name: str | None) -> bool:
    normalized = str(column_name or "").strip().lower()
    return (
        "month" in normalized
        or "月份" in normalized
        or "年月" in normalized
        or normalized == "月"
        or normalized.endswith("_month")
    )


def _has_day_hint(column_name: str | None) -> bool:
    normalized = str(column_name or "").strip().lower()
    return (
        "date" in normalized
        or "day" in normalized
        or "日期" in normalized
        or normalized == "日"
        or normalized.endswith("_date")
    )


def _has_quarter_hint(column_name: str | None) -> bool:
    normalized = str(column_name or "").strip().lower()
    return "quarter" in normalized or "季度" in normalized or normalized.endswith("_quarter")


def _has_epoch_hint(column_name: str | None) -> bool:
    normalized = str(column_name or "").strip().lower()
    return "timestamp" in normalized or "unix" in normalized or "时间戳" in normalized
