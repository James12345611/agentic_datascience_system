from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from app.core.datetime_utils import parse_datetime_series
from app.schemas.config import ColumnConfig, PreprocessConfig


def transform_dataframe(
    df: pd.DataFrame, resolved_types: Dict[str, str], config: PreprocessConfig
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    transformed = df.copy()
    run_log: Dict[str, str] = {}

    datetime_columns = [
        col for col, kind in resolved_types.items() if col in transformed.columns and kind == "datetime"
    ]
    for column in datetime_columns:
        series = parse_datetime_series(transformed[column], column)
        transformed[f"{column}_year"] = series.dt.year
        transformed[f"{column}_month"] = series.dt.month
        transformed[f"{column}_day"] = series.dt.day
        run_log[f"datetime_expand_{column}"] = "Expanded datetime into year/month/day"

    transformed, encoding_log = _apply_encoding(transformed, resolved_types, config)
    transformed, scaling_log = _apply_scaling(transformed, resolved_types, config)
    run_log.update(encoding_log)
    run_log.update(scaling_log)

    return transformed, run_log


def _apply_encoding(
    df: pd.DataFrame, resolved_types: Dict[str, str], config: PreprocessConfig
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    transformed = df.copy()
    run_log: Dict[str, str] = {}

    onehot_columns = []
    label_columns = []

    for column, kind in resolved_types.items():
        if column not in transformed.columns or kind not in {"categorical", "boolean"}:
            continue
        strategy = _resolve_encoding_strategy(config.column_configs.get(column), config)
        if strategy == "onehot":
            onehot_columns.append(column)
        elif strategy == "label":
            label_columns.append(column)

    if onehot_columns:
        transformed = pd.get_dummies(transformed, columns=onehot_columns, dummy_na=False)
        run_log["encoding_onehot"] = f"Applied one-hot encoding to {onehot_columns}"

    for column in label_columns:
        if column not in transformed.columns:
            continue
        encoder = LabelEncoder()
        transformed[column] = encoder.fit_transform(transformed[column].astype(str))

    if label_columns:
        run_log["encoding_label"] = f"Applied label encoding to {label_columns}"

    return transformed, run_log


def _apply_scaling(
    df: pd.DataFrame, resolved_types: Dict[str, str], config: PreprocessConfig
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    transformed = df.copy()
    run_log: Dict[str, str] = {}

    standard_columns = []
    minmax_columns = []

    for column, kind in resolved_types.items():
        if column not in transformed.columns or kind != "numerical":
            continue
        strategy = _resolve_scaling_strategy(config.column_configs.get(column), config)
        if strategy == "standard":
            standard_columns.append(column)
        elif strategy == "minmax":
            minmax_columns.append(column)

    if standard_columns:
        scaler = StandardScaler()
        transformed[standard_columns] = scaler.fit_transform(transformed[standard_columns])
        run_log["scaling_standard"] = f"Applied standard scaling to {standard_columns}"

    if minmax_columns:
        scaler = MinMaxScaler()
        transformed[minmax_columns] = scaler.fit_transform(transformed[minmax_columns])
        run_log["scaling_minmax"] = f"Applied min-max scaling to {minmax_columns}"

    return transformed, run_log


def _resolve_encoding_strategy(column_config: ColumnConfig | None, config: PreprocessConfig) -> str:
    if column_config and column_config.encoding:
        return column_config.encoding
    return config.encoding


def _resolve_scaling_strategy(column_config: ColumnConfig | None, config: PreprocessConfig) -> str:
    if column_config and column_config.scaling:
        return column_config.scaling
    return config.scaling
