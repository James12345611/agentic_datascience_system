from __future__ import annotations

from typing import Dict

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from app.core.datetime_utils import DATETIME_NAME_HINTS, has_datetime_hint, parse_datetime_series


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    inferred: Dict[str, str] = {}
    row_count = max(len(df), 1)

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()

        if non_null.empty:
            inferred[column] = "categorical"
            continue

        if is_datetime64_any_dtype(series):
            inferred[column] = "datetime"
            continue

        unique_ratio = non_null.nunique(dropna=True) / row_count

        if _looks_like_boolean(non_null):
            inferred[column] = "boolean"
        elif _looks_like_numeric(non_null):
            if _looks_like_compact_datetime(non_null, column):
                inferred[column] = "datetime"
            elif unique_ratio > 0.95 and non_null.nunique(dropna=True) > 20:
                inferred[column] = "id_like"
            else:
                inferred[column] = "numerical"
        elif _looks_like_datetime(non_null, column):
            inferred[column] = "datetime"
        else:
            if unique_ratio > 0.95 and non_null.nunique(dropna=True) > 20:
                inferred[column] = "id_like"
            elif non_null.nunique(dropna=True) <= 20:
                inferred[column] = "categorical"
            else:
                inferred[column] = "text"

    return inferred


def _looks_like_numeric(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().mean() >= 0.9


def _looks_like_datetime(series: pd.Series, column_name: str) -> bool:
    normalized_name = column_name.lower()
    if not has_datetime_hint(normalized_name):
        string_sample = series.astype(str).head(20)
        has_datetime_pattern = (
            string_sample.str.contains(r"[-/:年月日时分秒Tt]", regex=True).mean() >= 0.6
        )
        if not has_datetime_pattern:
            return False

    converted = parse_datetime_series(series, column_name)
    return converted.notna().mean() >= 0.9


def _looks_like_boolean(series: pd.Series) -> bool:
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"true": "1", "false": "0", "yes": "1", "no": "0"})
    )
    return normalized.isin({"0", "1"}).mean() >= 0.9 and normalized.nunique() <= 2


def _looks_like_compact_datetime(series: pd.Series, column_name: str) -> bool:
    if not has_datetime_hint(column_name):
        return False

    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return False

    string_values = numeric_series.round().astype("Int64").astype(str)
    valid_lengths = string_values.str.len().isin([4, 6, 8, 10, 13])
    return valid_lengths.mean() >= 0.9
