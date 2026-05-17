from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.schemas.config import ModelingConfig


@dataclass
class ModelingArtifacts:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    prediction_frame: pd.DataFrame


def prepare_modeling_dataset(df: pd.DataFrame, config: ModelingConfig) -> pd.DataFrame:
    if config.target_column not in df.columns:
        raise ValueError(f"目标字段不存在: {config.target_column}")

    feature_columns = config.feature_columns or [column for column in df.columns if column != config.target_column]
    if not feature_columns:
        raise ValueError("至少需要选择一个特征字段。")

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"以下特征字段不存在: {missing_features}")

    modeling_columns = feature_columns + [config.target_column]
    model_df = df[modeling_columns].copy()
    if config.drop_target_missing:
        model_df = model_df[model_df[config.target_column].notna()].copy()

    if model_df.empty:
        raise ValueError("清理目标字段缺失后没有可用于建模的数据。")

    return model_df


def run_statistical_model(df: pd.DataFrame, config: ModelingConfig) -> Dict[str, Any]:
    model_df = prepare_modeling_dataset(df, config)
    feature_columns = [column for column in model_df.columns if column != config.target_column]
    design_df = _encode_for_stats_model(model_df[feature_columns], max_categorical_levels=config.max_categorical_levels)
    design_df = design_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    target_series = model_df.loc[design_df.index, config.target_column]

    if config.problem_type == "regression" and config.algorithm == "ols":
        x = sm.add_constant(design_df, has_constant="add")
        model = sm.OLS(target_series.astype(float), x).fit()
        prediction = model.predict(x)
        metrics = {
            "r2": float(r2_score(target_series, prediction)),
            "mae": float(mean_absolute_error(target_series, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(target_series, prediction))),
        }
        coefficient_table = _build_coefficient_table(model)
        summary_text = model.summary().as_text()
    elif config.problem_type == "binary_classification" and config.algorithm == "logit":
        binary_target = _normalize_binary_target(target_series)
        x = sm.add_constant(design_df, has_constant="add")
        model = sm.Logit(binary_target, x).fit(disp=False)
        probability = model.predict(x)
        prediction = (probability >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(binary_target, prediction)),
            "f1": float(f1_score(binary_target, prediction)),
            "auc": float(roc_auc_score(binary_target, probability)),
        }
        coefficient_table = _build_coefficient_table(model)
        summary_text = model.summary().as_text()
        target_series = binary_target
    else:
        raise ValueError("当前仅支持 OLS 回归和二分类 Logistic 回归。")

    prediction_frame = pd.DataFrame(
        {
            "actual": target_series,
            "prediction": prediction,
        }
    )
    if config.problem_type == "binary_classification" and config.algorithm == "logit":
        prediction_frame["probability"] = probability

    return {
        "model_type": config.algorithm,
        "model_family": config.model_family,
        "problem_type": config.problem_type,
        "feature_columns": feature_columns,
        "sample_size": int(len(prediction_frame)),
        "metrics": metrics,
        "coefficient_table": coefficient_table,
        "summary_text": summary_text,
        "artifacts": ModelingArtifacts(
            train_frame=model_df.loc[prediction_frame.index, :],
            test_frame=pd.DataFrame(),
            prediction_frame=prediction_frame,
        ),
    }


def run_machine_learning_model(df: pd.DataFrame, config: ModelingConfig) -> Dict[str, Any]:
    model_df = prepare_modeling_dataset(df, config)
    feature_columns = [column for column in model_df.columns if column != config.target_column]
    x = model_df[feature_columns]
    y = model_df[config.target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    numeric_columns = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in x.columns if column not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    estimator = _build_ml_estimator(config)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )
    pipeline.fit(x_train, y_train)

    prediction = pipeline.predict(x_test)
    metrics = _compute_ml_metrics(config.problem_type, y_test, prediction, pipeline, x_test)
    importance_table = _build_feature_importance_table(pipeline, numeric_columns, categorical_columns)

    prediction_frame = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "prediction": pd.Series(prediction).reset_index(drop=True),
        }
    )
    if config.problem_type in {"binary_classification", "multiclass_classification"} and hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(x_test)
        if probability.ndim == 2 and probability.shape[1] == 2:
            prediction_frame["probability"] = probability[:, 1]

    return {
        "model_type": config.algorithm,
        "model_family": config.model_family,
        "problem_type": config.problem_type,
        "feature_columns": feature_columns,
        "sample_size": int(len(model_df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "metrics": metrics,
        "feature_importance_table": importance_table,
        "summary_text": "机器学习模型训练完成。",
        "artifacts": ModelingArtifacts(
            train_frame=pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1),
            test_frame=pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
            prediction_frame=prediction_frame,
        ),
    }


def _encode_for_stats_model(df: pd.DataFrame, max_categorical_levels: int) -> pd.DataFrame:
    usable_df = df.copy()
    for column in list(usable_df.columns):
        if usable_df[column].dtype == "object" or str(usable_df[column].dtype).startswith("category"):
            level_count = usable_df[column].nunique(dropna=True)
            if level_count > max_categorical_levels:
                usable_df = usable_df.drop(columns=[column])
    encoded_df = pd.get_dummies(usable_df, drop_first=True, dtype=float)
    if encoded_df.empty:
        raise ValueError("可用于统计建模的特征为空，请减少高基数字段或选择更多数值字段。")
    return encoded_df


def _normalize_binary_target(target: pd.Series) -> pd.Series:
    unique_values = list(pd.Series(target).dropna().unique())
    if len(unique_values) != 2:
        raise ValueError("二分类建模要求目标变量恰好包含 2 个有效类别。")
    mapping = {unique_values[0]: 0, unique_values[1]: 1}
    return target.map(mapping).astype(int)


def _build_coefficient_table(model: Any) -> list[Dict[str, Any]]:
    pvalues = getattr(model, "pvalues", None)
    conf_int = model.conf_int()
    rows = []
    for term, coef in model.params.items():
        rows.append(
            {
                "term": term,
                "coefficient": float(coef),
                "p_value": float(pvalues[term]) if pvalues is not None else None,
                "ci_lower": float(conf_int.loc[term, 0]),
                "ci_upper": float(conf_int.loc[term, 1]),
            }
        )
    return rows


def _build_ml_estimator(config: ModelingConfig) -> Any:
    if config.problem_type == "regression":
        if config.algorithm == "random_forest":
            return RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                random_state=config.random_state,
            )
        if config.algorithm == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth or 3,
                random_state=config.random_state,
            )
    if config.problem_type in {"binary_classification", "multiclass_classification"}:
        if config.algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                random_state=config.random_state,
            )
        if config.algorithm == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                random_state=config.random_state,
            )
    raise ValueError("当前仅支持随机森林与梯度提升作为机器学习建模算法。")


def _compute_ml_metrics(
    problem_type: str,
    y_true: pd.Series,
    prediction: np.ndarray,
    pipeline: Pipeline,
    x_test: pd.DataFrame,
) -> Dict[str, float]:
    if problem_type == "regression":
        return {
            "r2": float(r2_score(y_true, prediction)),
            "mae": float(mean_absolute_error(y_true, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro")),
    }
    if problem_type == "binary_classification" and hasattr(pipeline, "predict_proba"):
        probability = pipeline.predict_proba(x_test)
        if probability.ndim == 2 and probability.shape[1] == 2:
            metrics["auc"] = float(roc_auc_score(y_true, probability[:, 1]))
    return metrics


def _build_feature_importance_table(
    pipeline: Pipeline,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> list[Dict[str, Any]]:
    estimator = pipeline.named_steps["estimator"]
    if not hasattr(estimator, "feature_importances_"):
        return []

    feature_names: list[str] = []
    feature_names.extend(numeric_columns)

    if categorical_columns:
        encoder = pipeline.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
        encoded_names = encoder.get_feature_names_out(categorical_columns).tolist()
        feature_names.extend(encoded_names)

    importances = estimator.feature_importances_
    table = [
        {"feature": feature_name, "importance": float(importance)}
        for feature_name, importance in zip(feature_names, importances)
    ]
    table.sort(key=lambda item: item["importance"], reverse=True)
    return table[:30]
