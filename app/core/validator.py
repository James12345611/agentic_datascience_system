from __future__ import annotations

from typing import Dict, List

import pandas as pd

from app.core.datetime_utils import parse_datetime_series
from app.schemas.config import ColumnConfig


def validate_columns(
    df: pd.DataFrame,
    resolved_types: Dict[str, str],
    column_configs: Dict[str, ColumnConfig],
) -> Dict[str, List[str]]:
    findings: Dict[str, List[str]] = {}

    for column, column_type in resolved_types.items():
        if column not in df.columns:
            continue

        config = column_configs.get(column)
        if config is None:
            continue

        messages: List[str] = []
        metadata = config.metadata
        series = df[column]

        if column_type == "numerical":
            numeric_series = pd.to_numeric(series, errors="coerce")
            valid = numeric_series.dropna()

            if metadata.integer_only:
                decimal_count = int((valid % 1 != 0).sum())
                if decimal_count > 0:
                    messages.append(f"该字段要求整数，但检测到 {decimal_count} 个非整数值。")

            if not metadata.allow_negative:
                negative_count = int((valid < 0).sum())
                if negative_count > 0:
                    messages.append(f"该字段不允许负值，但检测到 {negative_count} 个负数。")

            if metadata.legal_min is not None:
                below_count = int((valid < metadata.legal_min).sum())
                if below_count > 0:
                    messages.append(f"检测到 {below_count} 个值低于合法下界 {metadata.legal_min}。")

            if metadata.legal_max is not None:
                above_count = int((valid > metadata.legal_max).sum())
                if above_count > 0:
                    messages.append(f"检测到 {above_count} 个值高于合法上界 {metadata.legal_max}。")

            if metadata.soft_min is not None:
                below_soft = int((valid < metadata.soft_min).sum())
                if below_soft > 0:
                    messages.append(f"检测到 {below_soft} 个值低于经验下界 {metadata.soft_min}。")

            if metadata.soft_max is not None:
                above_soft = int((valid > metadata.soft_max).sum())
                if above_soft > 0:
                    messages.append(f"检测到 {above_soft} 个值高于经验上界 {metadata.soft_max}。")

            if metadata.semantic_type == "proportion":
                messages.extend(_validate_proportion(valid, metadata.raw_scale))
        elif column_type == "datetime":
            datetime_series = parse_datetime_series(series, column)
            valid = datetime_series.dropna()

            legal_start = _parse_datetime_bound(metadata.legal_start_time)
            legal_end = _parse_datetime_bound(metadata.legal_end_time)
            soft_start = _parse_datetime_bound(metadata.soft_start_time)
            soft_end = _parse_datetime_bound(metadata.soft_end_time)

            if legal_start is not None:
                below_count = int((valid < legal_start).sum())
                if below_count > 0:
                    messages.append(
                        f"检测到 {below_count} 个时间值早于合法起始时间 {legal_start.strftime('%Y-%m-%d %H:%M:%S')}。"
                    )

            if legal_end is not None:
                above_count = int((valid > legal_end).sum())
                if above_count > 0:
                    messages.append(
                        f"检测到 {above_count} 个时间值晚于合法结束时间 {legal_end.strftime('%Y-%m-%d %H:%M:%S')}。"
                    )

            if soft_start is not None:
                below_soft = int((valid < soft_start).sum())
                if below_soft > 0:
                    messages.append(
                        f"检测到 {below_soft} 个时间值早于经验起始时间 {soft_start.strftime('%Y-%m-%d %H:%M:%S')}。"
                    )

            if soft_end is not None:
                above_soft = int((valid > soft_end).sum())
                if above_soft > 0:
                    messages.append(
                        f"检测到 {above_soft} 个时间值晚于经验结束时间 {soft_end.strftime('%Y-%m-%d %H:%M:%S')}。"
                    )

        if messages:
            findings[column] = messages

    return findings


def _validate_proportion(series: pd.Series, raw_scale: str | None) -> List[str]:
    messages: List[str] = []
    if series.empty:
        return messages

    if raw_scale == "0-1":
        if int((series > 1).sum()) > 0:
            messages.append("该字段标记为 0-1 比例，但存在大于 1 的取值。")
        if int((series < 0).sum()) > 0:
            messages.append("该字段标记为 0-1 比例，但存在小于 0 的取值。")
    elif raw_scale == "0-100":
        if int((series > 100).sum()) > 0:
            messages.append("该字段标记为 0-100 百分比，但存在大于 100 的取值。")
        if int((series < 0).sum()) > 0:
            messages.append("该字段标记为 0-100 百分比，但存在小于 0 的取值。")
        if int(((series > 0) & (series < 1)).sum()) > 0:
            messages.append("该字段标记为百分比尺度，但检测到 0 到 1 之间的小数，可能混用了 0-1 和 0-100。")

    return messages


def _parse_datetime_bound(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(str(value), errors="coerce", format="mixed")
    if pd.isna(parsed):
        return None
    return parsed
