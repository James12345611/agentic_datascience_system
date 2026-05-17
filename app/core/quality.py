from __future__ import annotations

from typing import Dict, List

import pandas as pd


def build_quality_report(df: pd.DataFrame, inferred_types: Dict[str, str]) -> Dict:
    column_profiles: List[Dict] = []
    warnings: List[str] = []

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"Detected {duplicate_rows} duplicated rows.")

    for column in df.columns:
        series = df[column]
        dtype_name = inferred_types.get(column, "unknown")
        missing_ratio = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = float(unique_count / max(len(df), 1))
        example_values = [str(v) for v in series.dropna().head(5).tolist()]
        column_warnings: List[str] = []

        if missing_ratio > 0.3:
            column_warnings.append("high_missing")
        if unique_count <= 1 and len(df) > 0:
            column_warnings.append("single_value")
        if dtype_name == "id_like":
            column_warnings.append("possible_id_column")

        if dtype_name == "numerical":
            numeric_series = pd.to_numeric(series, errors="coerce")
            q1 = numeric_series.quantile(0.25)
            q3 = numeric_series.quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                outlier_mask = (numeric_series < (q1 - 1.5 * iqr)) | (
                    numeric_series > (q3 + 1.5 * iqr)
                )
                outlier_count = int(outlier_mask.sum())
            else:
                outlier_count = 0
        else:
            outlier_count = None

        column_profiles.append(
            {
                "column_name": column,
                "inferred_type": dtype_name,
                "missing_ratio": round(missing_ratio, 4),
                "unique_count": unique_count,
                "unique_ratio": round(unique_ratio, 4),
                "example_values": example_values,
                "outlier_count": outlier_count,
                "warnings": column_warnings,
            }
        )

    report = {
        "dataset_summary": {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "duplicate_rows": duplicate_rows,
            "all_null_rows": int(df.isna().all(axis=1).sum()),
        },
        "column_profiles": column_profiles,
        "table_warnings": warnings,
    }
    return report
