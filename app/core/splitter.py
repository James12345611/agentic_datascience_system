from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from app.schemas.config import SplitConfig


def split_dataframe(df: pd.DataFrame, split_config: SplitConfig | None) -> Dict[str, pd.DataFrame]:
    if split_config is None:
        return {"full": df}

    train_df, test_df = train_test_split(
        df,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
    )
    return {"train": train_df, "test": test_df}
