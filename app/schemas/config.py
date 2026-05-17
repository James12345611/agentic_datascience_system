from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


MissingStrategy = Literal["none", "mean", "median", "mode", "constant", "drop_row"]
ScalingStrategy = Literal["none", "standard", "minmax"]
EncodingStrategy = Literal["none", "onehot", "label"]
OutlierStrategy = Literal["none", "clip_iqr", "clip_quantile", "drop_iqr"]
ProblemType = Literal["regression", "binary_classification", "multiclass_classification", "count_regression"]
ModelFamily = Literal["statistical", "machine_learning"]
SemanticType = Literal[
    "general",
    "proportion",
    "amount",
    "count",
    "score",
    "index",
    "duration",
    "nominal_category",
    "ordinal_category",
    "geographic_region",
    "industry_category",
    "administrative_region",
    "text_label",
    "custom",
]


class SplitConfig(BaseModel):
    method: Literal["random"] = "random"
    test_size: float = 0.2
    random_state: int = 42


class ColumnMetadata(BaseModel):
    semantic_type: SemanticType = "general"
    custom_semantic_type: Optional[str] = None
    unit: Optional[str] = None
    canonical_unit: Optional[str] = None
    raw_scale: Optional[str] = None
    datetime_granularity: Optional[str] = None
    datetime_format: Optional[str] = None
    legal_start_time: Optional[str] = None
    legal_end_time: Optional[str] = None
    soft_start_time: Optional[str] = None
    soft_end_time: Optional[str] = None
    legal_min: Optional[float] = None
    legal_max: Optional[float] = None
    soft_min: Optional[float] = None
    soft_max: Optional[float] = None
    allow_negative: bool = True
    integer_only: bool = False
    description: Optional[str] = None


class ColumnConfig(BaseModel):
    enabled: bool = True
    inferred_type_override: Optional[str] = None
    missing_strategy: Optional[MissingStrategy] = None
    missing_constant_value: Optional[str] = None
    outlier_strategy: Optional[OutlierStrategy] = None
    outlier_lower_quantile: Optional[float] = None
    outlier_upper_quantile: Optional[float] = None
    encoding: Optional[EncodingStrategy] = None
    scaling: Optional[ScalingStrategy] = None
    metadata: ColumnMetadata = Field(default_factory=ColumnMetadata)


class PreprocessConfig(BaseModel):
    drop_columns: List[str] = Field(default_factory=list)
    type_overrides: Dict[str, str] = Field(default_factory=dict)
    column_configs: Dict[str, ColumnConfig] = Field(default_factory=dict)
    missing_numeric: MissingStrategy = "median"
    missing_categorical: MissingStrategy = "mode"
    missing_constant_value: Optional[str] = None
    outlier_strategy: OutlierStrategy = "none"
    outlier_lower_quantile: float = 0.01
    outlier_upper_quantile: float = 0.99
    scaling: ScalingStrategy = "none"
    encoding: EncodingStrategy = "onehot"
    remove_duplicates: bool = True
    drop_all_null_columns: bool = True
    drop_single_value_columns: bool = True
    save_artifacts: bool = True
    split: Optional[SplitConfig] = None


class ModelingConfig(BaseModel):
    model_family: ModelFamily = "statistical"
    algorithm: str = "ols"
    problem_type: ProblemType = "regression"
    target_column: str
    feature_columns: List[str] = Field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42
    save_artifacts: bool = True
    drop_target_missing: bool = True
    drop_unsupported_features: bool = True
    max_categorical_levels: int = 50
    n_estimators: int = 200
    max_depth: Optional[int] = 5
    learning_rate: float = 0.05
