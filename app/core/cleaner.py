from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from app.core.datetime_utils import parse_datetime_series
from app.schemas.config import ColumnConfig, PreprocessConfig


def clean_dataframe(
    df: pd.DataFrame, inferred_types: Dict[str, str], config: PreprocessConfig
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    cleaned = df.copy()
    run_log: Dict[str, str] = {}

    disabled_columns = [
        column for column, column_config in config.column_configs.items() if not column_config.enabled
    ]
    effective_drop_columns = sorted(set(config.drop_columns + disabled_columns))

    if effective_drop_columns:
        existing = [col for col in effective_drop_columns if col in cleaned.columns]
        cleaned = cleaned.drop(columns=existing)
        run_log["drop_columns"] = f"Dropped columns: {existing}"

    if config.remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        run_log["remove_duplicates"] = f"Removed {before - len(cleaned)} duplicated rows"

    if config.drop_all_null_columns:
        null_columns = cleaned.columns[cleaned.isna().all()].tolist()
        if null_columns:
            cleaned = cleaned.drop(columns=null_columns)
            run_log["drop_all_null_columns"] = f"Dropped all-null columns: {null_columns}"

    if config.drop_single_value_columns:
        single_columns = [col for col in cleaned.columns if cleaned[col].nunique(dropna=True) <= 1]
        if single_columns:
            cleaned = cleaned.drop(columns=single_columns)
            run_log["drop_single_value_columns"] = (
                f"Dropped single-value columns: {single_columns}"
            )

    resolved_types = {
        column: _resolve_type(column, inferred_types, config)
        for column in cleaned.columns
    }

    cleaned = _apply_type_conversions(cleaned, resolved_types)
    cleaned = _apply_missing_value_strategies(cleaned, resolved_types, config, run_log)
    cleaned, outlier_log = _apply_outlier_strategy(cleaned, resolved_types, config)
    run_log.update(outlier_log)
    cleaned = _normalize_numeric_semantics(cleaned, resolved_types, config, run_log)

    return cleaned, run_log, resolved_types


def _resolve_type(column: str, inferred_types: Dict[str, str], config: PreprocessConfig) -> str:
    column_config = config.column_configs.get(column)
    if column_config and column_config.inferred_type_override:
        return column_config.inferred_type_override
    if column in config.type_overrides:
        return config.type_overrides[column]
    return inferred_types.get(column, "categorical")


def _apply_type_conversions(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    converted = df.copy()
    for column, dtype_name in type_map.items():
        if column not in converted.columns:
            continue
        if dtype_name == "numerical":
            converted[column] = pd.to_numeric(converted[column], errors="coerce")
        elif dtype_name == "datetime":
            converted[column] = parse_datetime_series(converted[column], column)
        elif dtype_name == "boolean":
            normalized = (
                converted[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"true": "1", "false": "0", "yes": "1", "no": "0"})
            )
            converted[column] = normalized.map({"1": True, "0": False})
        else:
            converted[column] = converted[column].astype("object")
    return converted


def _apply_missing_value_strategies(
    df: pd.DataFrame,
    type_map: Dict[str, str],
    config: PreprocessConfig,
    run_log: Dict[str, str],
) -> pd.DataFrame:
    filled = df.copy()

    for column, dtype_name in type_map.items():
        if column not in filled.columns or not filled[column].isna().any():
            continue

        column_config = config.column_configs.get(column)
        if dtype_name == "numerical":
            strategy = _resolve_numeric_missing_strategy(column_config, config)
            constant_value = _resolve_missing_constant_value(column_config, config)
        else:
            strategy = _resolve_categorical_missing_strategy(column_config, config)
            constant_value = _resolve_missing_constant_value(column_config, config)

        if strategy == "drop_row":
            before = len(filled)
            filled = filled.loc[filled[column].notna()].copy()
            run_log[f"missing_{column}"] = f"Dropped {before - len(filled)} rows by missing values"
            continue

        if dtype_name == "numerical":
            if strategy == "mean":
                filled[column] = filled[column].fillna(filled[column].mean())
            elif strategy == "median":
                filled[column] = filled[column].fillna(filled[column].median())
            elif strategy == "constant":
                filled[column] = filled[column].fillna(constant_value)
        else:
            if strategy == "mode":
                mode = filled[column].mode(dropna=True)
                if not mode.empty:
                    filled[column] = filled[column].fillna(mode.iloc[0])
            elif strategy == "constant":
                filled[column] = filled[column].fillna(constant_value)

        run_log[f"missing_{column}"] = f"Applied missing strategy: {strategy}"

    return filled


def _apply_outlier_strategy(
    df: pd.DataFrame, type_map: Dict[str, str], config: PreprocessConfig
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    adjusted = df.copy()
    run_log: Dict[str, str] = {}

    numerical_columns = [
        column
        for column, dtype_name in type_map.items()
        if dtype_name == "numerical" and column in adjusted.columns
    ]

    for column in numerical_columns:
        column_config = config.column_configs.get(column)
        strategy = _resolve_outlier_strategy(column_config, config)
        if strategy == "none":
            continue

        series = pd.to_numeric(adjusted[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue

        if strategy in {"clip_iqr", "drop_iqr"}:
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr <= 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        else:
            lower_quantile = _resolve_lower_quantile(column_config, config)
            upper_quantile = _resolve_upper_quantile(column_config, config)
            lower = valid.quantile(lower_quantile)
            upper = valid.quantile(upper_quantile)

        outlier_mask = (series < lower) | (series > upper)
        outlier_count = int(outlier_mask.sum())
        if outlier_count == 0:
            continue

        if strategy in {"clip_iqr", "clip_quantile"}:
            adjusted[column] = series.clip(lower=lower, upper=upper)
            run_log[f"outlier_{column}"] = (
                f"Clipped {outlier_count} values to [{round(float(lower), 4)}, "
                f"{round(float(upper), 4)}]"
            )
        elif strategy == "drop_iqr":
            before = len(adjusted)
            adjusted = adjusted.loc[~outlier_mask].copy()
            run_log[f"outlier_{column}"] = f"Dropped {before - len(adjusted)} rows by IQR bounds"

    return adjusted, run_log


def _normalize_numeric_semantics(
    df: pd.DataFrame,
    type_map: Dict[str, str],
    config: PreprocessConfig,
    run_log: Dict[str, str],
) -> pd.DataFrame:
    normalized = df.copy()
    for column, dtype_name in type_map.items():
        if dtype_name != "numerical" or column not in normalized.columns:
            continue

        column_config = config.column_configs.get(column)
        if not column_config:
            continue

        metadata = column_config.metadata
        if metadata.semantic_type == "proportion" and metadata.raw_scale == "0-100":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce") / 100.0
            run_log[f"semantic_{column}"] = "Converted proportion from 0-100 scale to 0-1 scale"

    return normalized


def _resolve_numeric_missing_strategy(
    column_config: ColumnConfig | None, config: PreprocessConfig
) -> str:
    if column_config and column_config.missing_strategy:
        return column_config.missing_strategy
    return config.missing_numeric


def _resolve_categorical_missing_strategy(
    column_config: ColumnConfig | None, config: PreprocessConfig
) -> str:
    if column_config and column_config.missing_strategy:
        return column_config.missing_strategy
    return config.missing_categorical


def _resolve_missing_constant_value(
    column_config: ColumnConfig | None, config: PreprocessConfig
) -> str | None:
    if column_config and column_config.missing_constant_value is not None:
        return column_config.missing_constant_value
    return config.missing_constant_value


def _resolve_outlier_strategy(column_config: ColumnConfig | None, config: PreprocessConfig) -> str:
    if column_config and column_config.outlier_strategy:
        return column_config.outlier_strategy
    return config.outlier_strategy


def _resolve_lower_quantile(column_config: ColumnConfig | None, config: PreprocessConfig) -> float:
    if column_config and column_config.outlier_lower_quantile is not None:
        return column_config.outlier_lower_quantile
    return config.outlier_lower_quantile


def _resolve_upper_quantile(column_config: ColumnConfig | None, config: PreprocessConfig) -> float:
    if column_config and column_config.outlier_upper_quantile is not None:
        return column_config.outlier_upper_quantile
    return config.outlier_upper_quantile
