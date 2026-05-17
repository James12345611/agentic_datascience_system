from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(file_path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)

    raise ValueError(f"Unsupported file type: {suffix}")
