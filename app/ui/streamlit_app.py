from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.eda_stats import (
    compute_vif_table,
    generate_eda_insights,
    run_statistical_diagnostic,
    run_statistical_test,
)
from app.core.datetime_utils import parse_datetime_series
from app.core.reader import read_table
from app.schemas.config import ColumnConfig, ColumnMetadata, ModelingConfig, PreprocessConfig, SplitConfig
from app.services.modeling_service import run_modeling_with_source
from app.services.preprocess_service import (
    analyze_dataframe,
    persist_preprocessed_dataframe,
    run_preprocess_with_source,
    persist_raw_dataframe,
)
from app.storage.sqlite_store import (
    find_preprocessed_entry_by_task_id,
    get_preprocessed_entry,
    get_raw_dataset_entry,
    get_task_run,
    list_preprocessed_datasets,
    list_raw_datasets,
    list_task_runs,
    load_preprocessed_dataset,
    load_raw_dataset,
)


st.set_page_config(page_title="数据预处理原型 V1.8", layout="wide")


TYPE_OPTIONS = ["自动识别", "数值型", "分类型", "时间型", "布尔型", "文本型", "标识列"]
TYPE_MAPPING = {
    "自动识别": None,
    "数值型": "numerical",
    "分类型": "categorical",
    "时间型": "datetime",
    "布尔型": "boolean",
    "文本型": "text",
    "标识列": "id_like",
}
TYPE_LABELS = {
    "numerical": "数值型",
    "categorical": "分类型",
    "datetime": "时间型",
    "boolean": "布尔型",
    "text": "文本型",
    "id_like": "标识列",
}
NUMERIC_SEMANTICS = {
    "general": "普通连续值",
    "proportion": "比例/百分比",
    "amount": "金额",
    "count": "计数",
    "score": "评分",
    "index": "指数",
    "duration": "时长",
    "custom": "自定义",
}
CATEGORICAL_SEMANTICS = {
    "nominal_category": "名义分类",
    "ordinal_category": "有序分类",
    "geographic_region": "地理区域",
    "industry_category": "行业类别",
    "administrative_region": "行政区划",
    "text_label": "文本标签",
    "custom": "自定义",
}
DATETIME_SEMANTICS = {
    "general": "通用时间字段",
    "duration": "时间长度/时序索引",
    "custom": "自定义",
}
GENERIC_SEMANTICS = {
    "general": "通用字段",
    "custom": "自定义",
}
MISSING_OPTIONS = {
    "none": "不处理",
    "mean": "均值填补",
    "median": "中位数填补",
    "mode": "众数填补",
    "constant": "固定值填补",
    "drop_row": "删除缺失行",
}
OUTLIER_OPTIONS = {
    "none": "不处理",
    "clip_iqr": "按 IQR 截断",
    "clip_quantile": "按分位数截断",
    "drop_iqr": "按 IQR 删除异常样本",
}
ENCODING_OPTIONS = {
    "none": "不编码",
    "onehot": "独热编码",
    "label": "标签编码",
}
SCALING_OPTIONS = {
    "none": "不缩放",
    "standard": "标准化",
    "minmax": "归一化",
}
RAW_SCALE_OPTIONS = ["未指定", "0-1", "0-100"]
DATETIME_GRANULARITY_OPTIONS = ["未指定", "年", "季度", "月", "周", "日", "小时", "分钟", "秒"]
TIME_GRAIN_OPTIONS = ["自动", "年", "季度", "月", "周", "日"]
MODELING_FAMILY_OPTIONS = {
    "经典统计建模": "statistical",
    "机器学习建模": "machine_learning",
}
STATISTICAL_ALGORITHM_OPTIONS = {
    "线性回归 OLS": "ols",
    "二分类 Logistic 回归": "logit",
}
ML_ALGORITHM_OPTIONS = {
    "随机森林": "random_forest",
    "梯度提升树": "gradient_boosting",
}
QUALITY_WARNING_LABELS = {
    "high_missing": "缺失比例较高",
    "single_value": "单一取值列",
    "possible_id_column": "疑似标识列",
}
CAT_LIKE_TYPES = {"categorical", "boolean"}


def init_session_state() -> None:
    st.session_state.setdefault("current_df", None)
    st.session_state.setdefault("current_dataset_name", None)
    st.session_state.setdefault("current_source_file_name", None)
    st.session_state.setdefault("current_dataset_id", None)
    st.session_state.setdefault("current_source_mode", None)
    st.session_state.setdefault("last_uploaded_signature", None)
    st.session_state.setdefault("last_raw_storage_result", None)


def build_type_table(analysis: dict) -> pd.DataFrame:
    suggestions = analysis.get("type_review_suggestions", {})
    rows = []
    for column, inferred_type in analysis["inferred_types"].items():
        rows.append(
            {
                "字段名": column,
                "自动识别结果": TYPE_LABELS.get(inferred_type, inferred_type),
                "建议人工确认": "是" if column in suggestions else "否",
                "提示": "；".join(suggestions.get(column, [])),
            }
        )
    return pd.DataFrame(rows)


def build_dataset_summary(summary: dict) -> dict:
    mapping = {
        "row_count": "样本数",
        "column_count": "字段数",
        "duplicate_rows": "重复行数",
        "all_null_rows": "全空行数",
    }
    return {mapping.get(key, key): value for key, value in summary.items()}


def build_quality_table(profile_df: pd.DataFrame) -> pd.DataFrame:
    if profile_df.empty:
        return profile_df
    report = profile_df.copy()
    report["inferred_type"] = report["inferred_type"].map(lambda x: TYPE_LABELS.get(x, x))
    report["example_values"] = report["example_values"].map(
        lambda items: "；".join(map(str, items)) if items else ""
    )
    report["warnings"] = report["warnings"].map(
        lambda items: "；".join(QUALITY_WARNING_LABELS.get(item, item) for item in items) if items else ""
    )
    return report.rename(
        columns={
            "column_name": "字段名",
            "inferred_type": "识别类型",
            "missing_ratio": "缺失比例",
            "unique_count": "唯一值个数",
            "unique_ratio": "唯一值比例",
            "example_values": "样例值",
            "outlier_count": "异常值数量",
            "warnings": "风险标记",
        }
    )


def get_semantic_options(base_type: str | None) -> dict[str, str]:
    if base_type == "numerical":
        return NUMERIC_SEMANTICS
    if base_type == "categorical":
        return CATEGORICAL_SEMANTICS
    if base_type == "datetime":
        return DATETIME_SEMANTICS
    return GENERIC_SEMANTICS


def is_numeric_like_base_type(base_type: str | None) -> bool:
    return base_type == "numerical"


def is_categorical_like_base_type(base_type: str | None) -> bool:
    return base_type in {"categorical", "boolean"}


def is_datetime_like_base_type(base_type: str | None) -> bool:
    return base_type == "datetime"


def parse_float_or_none(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def set_current_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    source_file_name: str | None,
    dataset_id: str,
    source_mode: str,
) -> None:
    st.session_state["current_df"] = df
    st.session_state["current_dataset_name"] = dataset_name
    st.session_state["current_source_file_name"] = source_file_name
    st.session_state["current_dataset_id"] = dataset_id
    st.session_state["current_source_mode"] = source_mode


def _is_year_like_series(series: pd.Series, column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return (
        ("year" in normalized or "年份" in normalized or "年度" in normalized or normalized == "年")
        and (series.dt.month.eq(1) & series.dt.day.eq(1)).mean() >= 0.9
    ) or (
        series.dt.month.eq(1).all()
        and series.dt.day.eq(1).all()
        and series.nunique() == series.dt.year.nunique()
    )


def _is_quarter_like_series(series: pd.Series, column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return (
        "quarter" in normalized or "季度" in normalized
    ) and series.dt.month.isin([1, 4, 7, 10]).mean() >= 0.9 and series.dt.day.eq(1).mean() >= 0.9


def _is_month_like_series(series: pd.Series, column_name: str) -> bool:
    normalized = column_name.strip().lower()
    return (
        ("month" in normalized or "月份" in normalized or "年月" in normalized or normalized == "月")
        and series.dt.day.eq(1).mean() >= 0.9
    ) or (
        series.dt.day.eq(1).all() and series.nunique() == series.dt.to_period("M").nunique()
    )


def sample_for_eda(df: pd.DataFrame, max_rows: int = 5000) -> tuple[pd.DataFrame, bool]:
    if len(df) <= max_rows:
        return df, False
    return df.sample(n=max_rows, random_state=42), True


def resolve_time_grain(series: pd.Series, requested_grain: str, column_name: str) -> str:
    valid = series.dropna()
    if valid.empty:
        return "日"

    if requested_grain != "自动":
        return requested_grain

    if _is_year_like_series(valid, column_name):
        return "年"
    if _is_quarter_like_series(valid, column_name):
        return "季度"
    if _is_month_like_series(valid, column_name):
        return "月"

    span_days = max((valid.max() - valid.min()).days, 0)
    if span_days >= 365 * 3:
        return "年"
    if span_days >= 180:
        return "月"
    if span_days >= 60:
        return "周"
    return "日"


def build_time_bucket(series: pd.Series, grain: str) -> pd.Series:
    if grain == "年":
        return series.dt.to_period("Y").dt.start_time
    if grain == "季度":
        return series.dt.to_period("Q").dt.start_time
    if grain == "月":
        return series.dt.to_period("M").dt.start_time
    if grain == "周":
        return series.dt.to_period("W").dt.start_time
    return series.dt.floor("D")


def build_time_coverage_frame(
    series: pd.Series, column_name: str, requested_grain: str
) -> tuple[pd.DataFrame, str]:
    valid = series.dropna()
    actual_grain = resolve_time_grain(valid, requested_grain, column_name)
    bucket = build_time_bucket(valid, actual_grain)
    summary = bucket.value_counts().sort_index().rename_axis("时间").reset_index(name="记录数")
    return summary, actual_grain


def build_time_numeric_frame(
    dt_series: pd.Series,
    value_series: pd.Series,
    column_name: str,
    requested_grain: str,
    agg_method: str,
) -> tuple[pd.DataFrame, str]:
    frame = pd.DataFrame({"时间": dt_series, "数值": pd.to_numeric(value_series, errors="coerce")}).dropna()
    actual_grain = resolve_time_grain(frame["时间"], requested_grain, column_name)
    frame["时间"] = build_time_bucket(frame["时间"], actual_grain)
    aggregated = frame.groupby("时间", as_index=False)["数值"].agg(agg_method)
    aggregated = aggregated.rename(columns={"数值": agg_method})
    return aggregated, actual_grain


def build_time_category_frame(
    dt_series: pd.Series,
    category_series: pd.Series,
    column_name: str,
    requested_grain: str,
) -> tuple[pd.DataFrame, str]:
    frame = pd.DataFrame({"时间": dt_series, "类别": category_series.astype(str)}).dropna()
    actual_grain = resolve_time_grain(frame["时间"], requested_grain, column_name)
    frame["时间"] = build_time_bucket(frame["时间"], actual_grain)

    top_categories = frame["类别"].value_counts().head(6).index
    frame["类别"] = frame["类别"].where(frame["类别"].isin(top_categories), "其他")
    aggregated = frame.groupby(["时间", "类别"], as_index=False).size().rename(columns={"size": "记录数"})
    return aggregated, actual_grain


def format_time_value(timestamp: pd.Timestamp, grain: str) -> str:
    if pd.isna(timestamp):
        return "-"
    if grain == "年":
        return timestamp.strftime("%Y")
    if grain == "季度":
        quarter = ((timestamp.month - 1) // 3) + 1
        return f"{timestamp.year}Q{quarter}"
    if grain == "月":
        return timestamp.strftime("%Y-%m")
    if timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0:
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def numeric_numeric_association(series_a: pd.Series, series_b: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {
            "a": pd.to_numeric(series_a, errors="coerce"),
            "b": pd.to_numeric(series_b, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 3:
        return None
    corr = frame["a"].corr(frame["b"])
    if pd.isna(corr):
        return None
    return float(abs(corr))


def cramers_v(series_a: pd.Series, series_b: pd.Series) -> float | None:
    frame = pd.DataFrame({"a": series_a.astype(str), "b": series_b.astype(str)}).dropna()
    if frame.empty:
        return None

    confusion = pd.crosstab(frame["a"], frame["b"])
    if confusion.empty or confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return None

    observed = confusion.to_numpy(dtype=float)
    n = observed.sum()
    if n <= 1:
        return None

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    valid_expected = expected > 0
    chi2 = (((observed - expected) ** 2) / expected)[valid_expected].sum()

    r, k = observed.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denominator = min(k_corr - 1, r_corr - 1)
    if denominator <= 0:
        return None
    return float((phi2_corr / denominator) ** 0.5)


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    frame = pd.DataFrame({"category": categories.astype(str), "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if frame.empty or frame["category"].nunique() < 2:
        return None

    grand_mean = frame["value"].mean()
    grouped = frame.groupby("category")["value"]
    counts = grouped.size()
    means = grouped.mean()
    ss_between = float((counts * ((means - grand_mean) ** 2)).sum())
    ss_total = float(((frame["value"] - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return None
    return float((ss_between / ss_total) ** 0.5)


def datetime_numeric_association(dt_series: pd.Series, numeric_series: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {
            "time": parse_datetime_series(dt_series, "datetime_proxy"),
            "value": pd.to_numeric(numeric_series, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 3:
        return None
    time_numeric = frame["time"].astype("int64")
    corr = pd.Series(time_numeric).corr(frame["value"])
    if pd.isna(corr):
        return None
    return float(abs(corr))


def datetime_category_association(dt_series: pd.Series, category_series: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {
            "time": parse_datetime_series(dt_series, "datetime_proxy"),
            "category": category_series.astype(str),
        }
    ).dropna()
    if len(frame) < 3:
        return None
    time_numeric = frame["time"].astype("int64")
    return correlation_ratio(frame["category"], time_numeric)


def datetime_datetime_association(series_a: pd.Series, series_b: pd.Series) -> float | None:
    frame = pd.DataFrame(
        {
            "time_a": parse_datetime_series(series_a, "datetime_a"),
            "time_b": parse_datetime_series(series_b, "datetime_b"),
        }
    ).dropna()
    if len(frame) < 3:
        return None
    corr = frame["time_a"].astype("int64").corr(frame["time_b"].astype("int64"))
    if pd.isna(corr):
        return None
    return float(abs(corr))


def compute_association_score(
    df: pd.DataFrame,
    inferred_types: dict[str, str],
    column_a: str,
    column_b: str,
) -> float | None:
    type_a = inferred_types.get(column_a)
    type_b = inferred_types.get(column_b)
    series_a = df[column_a]
    series_b = df[column_b]

    if type_a == "numerical" and type_b == "numerical":
        return numeric_numeric_association(series_a, series_b)
    if type_a in CAT_LIKE_TYPES and type_b in CAT_LIKE_TYPES:
        return cramers_v(series_a, series_b)
    if type_a in CAT_LIKE_TYPES and type_b == "numerical":
        return correlation_ratio(series_a, series_b)
    if type_b in CAT_LIKE_TYPES and type_a == "numerical":
        return correlation_ratio(series_b, series_a)
    if type_a == "datetime" and type_b == "numerical":
        return datetime_numeric_association(series_a, series_b)
    if type_b == "datetime" and type_a == "numerical":
        return datetime_numeric_association(series_b, series_a)
    if type_a == "datetime" and type_b in CAT_LIKE_TYPES:
        return datetime_category_association(series_a, series_b)
    if type_b == "datetime" and type_a in CAT_LIKE_TYPES:
        return datetime_category_association(series_b, series_a)
    if type_a == "datetime" and type_b == "datetime":
        return datetime_datetime_association(series_a, series_b)
    return None


def build_column_configs(columns: list[str], analysis: dict) -> dict[str, ColumnConfig]:
    column_configs: dict[str, ColumnConfig] = {}

    for column in columns:
        inferred_type = analysis["inferred_types"].get(column, "categorical")
        inferred_label = TYPE_LABELS.get(inferred_type, inferred_type)

        with st.expander(f"字段：{column}", expanded=False):
            enabled = st.checkbox("参与预处理", value=True, key=f"enabled_{column}")
            override_label = st.selectbox(
                "字段基础类型",
                options=TYPE_OPTIONS,
                index=0,
                key=f"type_{column}",
                help=f"当前自动识别结果：{inferred_label}",
            )

            selected_base_type = TYPE_MAPPING[override_label] or inferred_type
            numeric_controls_enabled = is_numeric_like_base_type(selected_base_type)
            datetime_controls_enabled = is_datetime_like_base_type(selected_base_type)
            categorical_controls_enabled = is_categorical_like_base_type(selected_base_type)

            semantic_options = get_semantic_options(selected_base_type)
            semantic_key = st.selectbox(
                "字段语义类型",
                options=list(semantic_options.keys()),
                index=0,
                format_func=lambda x: semantic_options[x],
                key=f"semantic_{column}",
            )

            custom_semantic_type = None
            if semantic_key == "custom":
                custom_semantic_type = st.text_input(
                    "自定义语义类型",
                    value="",
                    key=f"custom_semantic_{column}",
                    help="例如：城市群名称、学校类型、企业所有制、政策类别。",
                )

            unit = None
            canonical_unit = None
            raw_scale = "未指定"
            legal_min_raw = ""
            legal_max_raw = ""
            soft_min_raw = ""
            soft_max_raw = ""
            allow_negative = True
            integer_only = False
            datetime_granularity = None
            datetime_format = None
            legal_start_time = None
            legal_end_time = None
            soft_start_time = None
            soft_end_time = None

            if numeric_controls_enabled:
                unit = st.text_input("单位", value="", key=f"unit_{column}")
                canonical_unit = st.text_input("标准单位", value="", key=f"canonical_unit_{column}")
                raw_scale = st.selectbox("原始尺度", options=RAW_SCALE_OPTIONS, index=0, key=f"raw_scale_{column}")

                range_col1, range_col2 = st.columns(2)
                with range_col1:
                    legal_min_raw = st.text_input("合法下界", value="", key=f"legal_min_{column}")
                    soft_min_raw = st.text_input("经验下界", value="", key=f"soft_min_{column}")
                with range_col2:
                    legal_max_raw = st.text_input("合法上界", value="", key=f"legal_max_{column}")
                    soft_max_raw = st.text_input("经验上界", value="", key=f"soft_max_{column}")

                allow_negative = st.checkbox("允许负值", value=True, key=f"allow_negative_{column}")
                integer_only = st.checkbox("要求整数", value=False, key=f"integer_only_{column}")
            elif datetime_controls_enabled:
                st.caption("当前字段为时间型，可设置时间粒度、时间格式和时间范围约束。")
                unit = st.text_input("时间单位说明", value="", key=f"unit_{column}")
                canonical_unit = st.text_input("标准时间单位", value="", key=f"canonical_unit_{column}")
                datetime_granularity = st.selectbox(
                    "时间粒度",
                    options=DATETIME_GRANULARITY_OPTIONS,
                    index=0,
                    key=f"datetime_granularity_{column}",
                )
                datetime_format = st.text_input(
                    "时间格式",
                    value="",
                    key=f"datetime_format_{column}",
                    help="例如：%Y-%m-%d、%Y/%m/%d %H:%M:%S",
                )

                time_col1, time_col2 = st.columns(2)
                with time_col1:
                    legal_start_time = st.text_input("合法起始时间", value="", key=f"legal_start_time_{column}")
                    soft_start_time = st.text_input("经验起始时间", value="", key=f"soft_start_time_{column}")
                with time_col2:
                    legal_end_time = st.text_input("合法结束时间", value="", key=f"legal_end_time_{column}")
                    soft_end_time = st.text_input("经验结束时间", value="", key=f"soft_end_time_{column}")
            else:
                st.caption("当前字段不是数值型或时间型，单位、尺度和范围约束不参与配置。")

            description = st.text_area("备注", value="", key=f"description_{column}")

            missing_strategy = st.selectbox(
                "字段级缺失值策略",
                options=["__default__"] + list(MISSING_OPTIONS.keys()),
                index=0,
                format_func=lambda x: "沿用全局默认" if x == "__default__" else MISSING_OPTIONS[x],
                key=f"missing_{column}",
            )
            missing_constant_value = st.text_input("字段级固定填充值", value="", key=f"missing_constant_{column}")

            outlier_strategy = st.selectbox(
                "字段级异常值策略",
                options=["__default__"] + list(OUTLIER_OPTIONS.keys()),
                index=0,
                format_func=lambda x: "沿用全局默认" if x == "__default__" else OUTLIER_OPTIONS[x],
                key=f"outlier_{column}",
                disabled=not numeric_controls_enabled,
            )
            outlier_lower_raw = st.text_input(
                "字段级异常下分位点",
                value="",
                key=f"outlier_lower_{column}",
                disabled=not numeric_controls_enabled,
            )
            outlier_upper_raw = st.text_input(
                "字段级异常上分位点",
                value="",
                key=f"outlier_upper_{column}",
                disabled=not numeric_controls_enabled,
            )
            if not numeric_controls_enabled:
                outlier_strategy = "__default__"
                outlier_lower_raw = ""
                outlier_upper_raw = ""

            encoding = st.selectbox(
                "字段级编码方式",
                options=["__default__"] + list(ENCODING_OPTIONS.keys()),
                index=0,
                format_func=lambda x: "沿用全局默认" if x == "__default__" else ENCODING_OPTIONS[x],
                key=f"encoding_{column}",
                disabled=not categorical_controls_enabled,
            )
            scaling = st.selectbox(
                "字段级缩放方式",
                options=["__default__"] + list(SCALING_OPTIONS.keys()),
                index=0,
                format_func=lambda x: "沿用全局默认" if x == "__default__" else SCALING_OPTIONS[x],
                key=f"scaling_{column}",
                disabled=not numeric_controls_enabled,
            )
            if not categorical_controls_enabled:
                encoding = "__default__"
            if not numeric_controls_enabled:
                scaling = "__default__"

            column_configs[column] = ColumnConfig(
                enabled=enabled,
                inferred_type_override=TYPE_MAPPING[override_label],
                missing_strategy=None if missing_strategy == "__default__" else missing_strategy,
                missing_constant_value=missing_constant_value or None,
                outlier_strategy=None if outlier_strategy == "__default__" else outlier_strategy,
                outlier_lower_quantile=parse_float_or_none(outlier_lower_raw),
                outlier_upper_quantile=parse_float_or_none(outlier_upper_raw),
                encoding=None if encoding == "__default__" else encoding,
                scaling=None if scaling == "__default__" else scaling,
                metadata=ColumnMetadata(
                    semantic_type=semantic_key,
                    custom_semantic_type=custom_semantic_type or None,
                    unit=unit or None,
                    canonical_unit=canonical_unit or None,
                    raw_scale=None if raw_scale == "未指定" else raw_scale,
                    datetime_granularity=None
                    if datetime_granularity in {None, "未指定"}
                    else datetime_granularity,
                    datetime_format=datetime_format or None,
                    legal_start_time=legal_start_time or None,
                    legal_end_time=legal_end_time or None,
                    soft_start_time=soft_start_time or None,
                    soft_end_time=soft_end_time or None,
                    legal_min=parse_float_or_none(legal_min_raw),
                    legal_max=parse_float_or_none(legal_max_raw),
                    soft_min=parse_float_or_none(soft_min_raw),
                    soft_max=parse_float_or_none(soft_max_raw),
                    allow_negative=allow_negative,
                    integer_only=integer_only,
                    description=description or None,
                ),
            )

    return column_configs


def build_history_table(items: list[dict], table_type: str) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return df
    if table_type == "raw":
        return df.rename(
            columns={
                "dataset_id": "数据集 ID",
                "dataset_name": "数据集名称",
                "source_file_name": "源文件名",
                "upload_time": "入库时间",
                "row_count": "行数",
                "column_count": "列数",
            }
        )
    if table_type == "task":
        renamed = df.rename(
            columns={
                "task_id": "任务 ID",
                "task_type": "任务类型",
                "dataset_id": "数据集 ID",
                "status": "状态",
                "source_name": "来源文件",
                "created_time": "创建时间",
                "updated_time": "更新时间",
                "error_message": "错误信息",
            }
        )
        ordered_columns = [
            column
            for column in ["任务 ID", "任务类型", "数据集 ID", "状态", "来源文件", "创建时间", "更新时间", "错误信息"]
            if column in renamed.columns
        ]
        return renamed[ordered_columns]
    return df.rename(
        columns={
            "preprocess_run_id": "预处理运行 ID",
            "dataset_id": "数据集 ID",
            "task_id": "任务 ID",
            "preprocess_time": "处理时间",
            "row_count": "行数",
            "column_count": "列数",
        }
    )


def render_raw_history_panel(raw_items: list[dict]) -> None:
    st.markdown("**原始数据历史记录**")
    st.dataframe(build_history_table(raw_items, "raw"), use_container_width=True, hide_index=True)

    options = {
        f"{item['dataset_name']} | {item['dataset_id']} | {item['row_count']}x{item['column_count']} | {item['upload_time']}": item
        for item in raw_items
    }
    selected_label = st.selectbox("选择 raw_dataset 记录", options=list(options.keys()), key="raw_history_select")
    selected = options[selected_label]
    entry = get_raw_dataset_entry(selected["dataset_id"])

    with st.expander("查看原始数据详情", expanded=True):
        st.json(entry)
        preview_df = load_raw_dataset(selected["dataset_id"]).head(10)
        st.dataframe(preview_df, use_container_width=True)

    if st.button("加载原始数据到当前工作区", key="load_raw_history"):
        df = load_raw_dataset(selected["dataset_id"])
        set_current_dataset(
            df,
            dataset_name=entry["dataset_name"],
            source_file_name=entry["source_file_name"],
            dataset_id=entry["dataset_id"],
            source_mode="raw_sqlite",
        )


def render_preprocessed_history_panel(processed_items: list[dict]) -> None:
    st.markdown("**预处理结果历史记录**")
    st.dataframe(build_history_table(processed_items, "processed"), use_container_width=True, hide_index=True)

    options = {
        f"{item['preprocess_run_id']} | dataset={item['dataset_id']} | {item['row_count']}x{item['column_count']} | {item['preprocess_time']}": item
        for item in processed_items
    }
    selected_label = st.selectbox(
        "选择 preprocessed_data 记录",
        options=list(options.keys()),
        key="processed_history_select",
    )
    selected = options[selected_label]
    entry = get_preprocessed_entry(selected["preprocess_run_id"])

    with st.expander("查看预处理结果详情", expanded=True):
        st.json(entry)
        preview_df = load_preprocessed_dataset(selected["preprocess_run_id"]).head(10)
        st.dataframe(preview_df, use_container_width=True)

    if st.button("加载预处理结果到当前工作区", key="load_processed_history"):
        df = load_preprocessed_dataset(selected["preprocess_run_id"])
        set_current_dataset(
            df,
            dataset_name=f"preprocessed_{selected['dataset_id']}",
            source_file_name=None,
            dataset_id=entry["dataset_id"],
            source_mode="preprocessed_sqlite",
        )


def render_task_history_panel(current_dataset_id: str | None = None) -> None:
    st.markdown("**任务中心**")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        task_type = st.selectbox(
            "任务类型筛选",
            options=["全部", "preprocess", "modeling"],
            index=0,
            key="task_type_filter",
        )
    with filter_col2:
        task_status = st.selectbox(
            "状态筛选",
            options=["全部", "running", "completed", "failed"],
            index=0,
            key="task_status_filter",
        )
    with filter_col3:
        only_current_dataset = st.checkbox(
            "仅查看当前数据集任务",
            value=bool(current_dataset_id),
            disabled=current_dataset_id is None,
            key="task_dataset_filter",
        )
    task_items = list_task_runs(
        task_type=None if task_type == "全部" else task_type,
        dataset_id=current_dataset_id if only_current_dataset else None,
        status=None if task_status == "全部" else task_status,
        limit=100,
    )
    if not task_items:
        st.info("当前没有可展示的任务记录。")
        return

    st.dataframe(build_history_table(task_items, "task"), use_container_width=True, hide_index=True)

    options = {
        f"{item['task_id']} | {item['task_type']} | {item['status']} | {item['updated_time']}": item
        for item in task_items
    }
    selected_label = st.selectbox("选择任务记录", options=list(options.keys()), key="task_history_select")
    selected = options[selected_label]
    entry = get_task_run(selected["task_id"])
    linked_preprocessed_entry = find_preprocessed_entry_by_task_id(selected["task_id"])

    with st.expander("查看任务详情", expanded=True):
        if entry["status"] == "failed" and entry.get("error_message"):
            st.error(f"任务执行失败：{entry['error_message']}")
        elif entry["status"] == "running":
            st.info("该任务仍在运行中，当前可先查看请求参数与基础信息。")
        else:
            st.success("任务已完成。")

        st.json(entry)
        result_payload = entry.get("result_payload_json") or {}
        preprocess_run_id = result_payload.get("preprocess_run_id")
        if preprocess_run_id:
            st.caption(f"关联预处理结果：{preprocess_run_id}")
        elif linked_preprocessed_entry:
            st.caption(f"检测到关联预处理结果：{linked_preprocessed_entry['preprocess_run_id']}")
        else:
            st.caption("当前任务尚未关联预处理结果。")

        if linked_preprocessed_entry:
            st.markdown("**关联预处理结果概览**")
            st.json(
                {
                    "preprocess_run_id": linked_preprocessed_entry["preprocess_run_id"],
                    "dataset_id": linked_preprocessed_entry["dataset_id"],
                    "task_id": linked_preprocessed_entry["task_id"],
                    "preprocess_time": linked_preprocessed_entry["preprocess_time"],
                    "shape": [
                        linked_preprocessed_entry["row_count"],
                        linked_preprocessed_entry["column_count"],
                    ],
                }
            )
            preview_df = load_preprocessed_dataset(linked_preprocessed_entry["preprocess_run_id"]).head(10)
            st.dataframe(preview_df, use_container_width=True)

            if st.button("从该任务加载预处理结果到当前工作区", key=f"load_task_result_{selected['task_id']}"):
                df = load_preprocessed_dataset(linked_preprocessed_entry["preprocess_run_id"])
                set_current_dataset(
                    df,
                    dataset_name=f"preprocessed_{linked_preprocessed_entry['dataset_id']}",
                    source_file_name=entry.get("source_name"),
                    dataset_id=linked_preprocessed_entry["dataset_id"],
                    source_mode="task_preprocessed_sqlite",
                )
                st.success("已从任务中心加载关联预处理结果。")


def render_data_source_panel() -> None:
    st.subheader("数据来源")
    source_mode = st.radio(
        "选择当前工作数据来源",
        options=["上传新数据", "从 SQLite 原始数据加载", "从 SQLite 预处理结果加载"],
        horizontal=True,
    )

    if source_mode == "上传新数据":
        uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_bytes)
                temp_path = tmp.name

            df = read_table(temp_path)
            Path(temp_path).unlink(missing_ok=True)
            dataset_name = Path(uploaded_file.name).stem

            file_signature = f"{uploaded_file.name}:{len(uploaded_bytes)}"
            if st.session_state.get("last_uploaded_signature") != file_signature:
                raw_storage_result = persist_raw_dataframe(
                    df,
                    dataset_name=dataset_name,
                    source_file_name=uploaded_file.name,
                )
                st.session_state["last_uploaded_signature"] = file_signature
                st.session_state["last_raw_storage_result"] = raw_storage_result
                set_current_dataset(
                    df,
                    dataset_name=dataset_name,
                    source_file_name=uploaded_file.name,
                    dataset_id=raw_storage_result["dataset_id"],
                    source_mode="raw_upload",
                )
            elif st.session_state.get("current_df") is None and st.session_state.get("last_raw_storage_result"):
                raw_storage_result = st.session_state["last_raw_storage_result"]
                set_current_dataset(
                    df,
                    dataset_name=dataset_name,
                    source_file_name=uploaded_file.name,
                    dataset_id=raw_storage_result["dataset_id"],
                    source_mode="raw_upload",
                )

    elif source_mode == "从 SQLite 原始数据加载":
        raw_items = list_raw_datasets()
        if not raw_items:
            st.info("当前数据库中还没有原始数据记录。")
            return
        render_raw_history_panel(raw_items)

    else:
        processed_items = list_preprocessed_datasets()
        if not processed_items:
            st.info("当前数据库中还没有预处理结果记录。")
            return
        render_preprocessed_history_panel(processed_items)


def infer_problem_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "regression"

    if pd.api.types.is_numeric_dtype(non_null):
        unique_count = non_null.nunique(dropna=True)
        if unique_count == 2:
            return "binary_classification"
        return "regression"

    unique_count = non_null.astype(str).nunique(dropna=True)
    if unique_count == 2:
        return "binary_classification"
    return "multiclass_classification"


def render_modeling_section(df: pd.DataFrame, analysis: dict, dataset_id: str, source_name: str | None) -> None:
    st.subheader("建模分析")
    all_columns = list(df.columns)
    if len(all_columns) < 2:
        st.info("至少需要 2 个字段才能进行建模。")
        return

    target_column = st.selectbox("选择目标变量", options=all_columns, key="model_target_column")
    candidate_features = [column for column in all_columns if column != target_column]
    default_features = candidate_features[: min(len(candidate_features), 8)]
    feature_columns = st.multiselect(
        "选择特征变量",
        options=candidate_features,
        default=default_features,
        key="model_feature_columns",
    )
    if not feature_columns:
        st.warning("请至少选择一个特征变量。")
        return

    inferred_problem_type = infer_problem_type(df[target_column])
    problem_type_options = {
        "回归": "regression",
        "二分类": "binary_classification",
        "多分类": "multiclass_classification",
    }
    available_problem_labels = ["回归", "二分类", "多分类"]
    default_problem_label = next(
        (label for label, value in problem_type_options.items() if value == inferred_problem_type),
        "回归",
    )

    left, middle, right = st.columns(3)
    with left:
        family_label = st.selectbox("建模家族", options=list(MODELING_FAMILY_OPTIONS.keys()), key="model_family")
    with middle:
        problem_label = st.selectbox(
            "问题类型",
            options=available_problem_labels,
            index=available_problem_labels.index(default_problem_label),
            key="model_problem_type",
        )
    with right:
        if MODELING_FAMILY_OPTIONS[family_label] == "statistical":
            algorithm_label = st.selectbox(
                "建模算法",
                options=list(STATISTICAL_ALGORITHM_OPTIONS.keys()),
                key="stat_algorithm",
            )
        else:
            algorithm_label = st.selectbox(
                "建模算法",
                options=list(ML_ALGORITHM_OPTIONS.keys()),
                key="ml_algorithm",
            )

    problem_type = problem_type_options[problem_label]
    model_family = MODELING_FAMILY_OPTIONS[family_label]
    algorithm = (
        STATISTICAL_ALGORITHM_OPTIONS[algorithm_label]
        if model_family == "statistical"
        else ML_ALGORITHM_OPTIONS[algorithm_label]
    )

    if model_family == "statistical" and algorithm == "ols" and problem_type != "regression":
        st.info("OLS 适用于回归问题，已建议你选择回归类型。")
    if model_family == "statistical" and algorithm == "logit" and problem_type != "binary_classification":
        st.info("Logistic 回归适用于二分类问题，建议将问题类型调整为二分类。")

    save_artifacts = st.checkbox("保存建模产物到 data/model_artifacts", value=True, key="model_save_artifacts")

    advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
    with advanced_col1:
        test_size = st.slider("测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05, key="model_test_size")
    with advanced_col2:
        n_estimators = st.slider("树模型数量", min_value=50, max_value=500, value=200, step=50, key="model_n_estimators")
    with advanced_col3:
        max_depth = st.slider("树深度", min_value=2, max_value=12, value=5, step=1, key="model_max_depth")

    config = ModelingConfig(
        model_family=model_family,
        algorithm=algorithm,
        problem_type=problem_type,
        target_column=target_column,
        feature_columns=feature_columns,
        test_size=test_size,
        save_artifacts=save_artifacts,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    if st.button("执行建模", type="primary", key="run_modeling"):
        result = run_modeling_with_source(
            df,
            config,
            source_name=source_name,
            dataset_id=dataset_id,
        )

        st.subheader("建模结果概览")
        st.json(
            {
                "任务信息": result["task"],
                "模型家族": family_label,
                "算法": algorithm_label,
                "问题类型": problem_label,
                "目标变量": target_column,
                "特征数量": len(feature_columns),
            }
        )

        st.subheader("评估指标")
        st.json(result["result"]["metrics"])

        if result["result"].get("coefficient_table"):
            st.subheader("系数表")
            st.dataframe(pd.DataFrame(result["result"]["coefficient_table"]), use_container_width=True)

        if result["result"].get("feature_importance_table"):
            st.subheader("特征重要性")
            importance_df = pd.DataFrame(result["result"]["feature_importance_table"])
            st.dataframe(importance_df, use_container_width=True)
            if not importance_df.empty:
                fig_importance = px.bar(
                    importance_df.head(15),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="前 15 个特征重要性",
                )
                st.plotly_chart(fig_importance, use_container_width=True)

        st.subheader("预测结果预览")
        st.dataframe(result["prediction_df"].head(20), use_container_width=True)

        if result["artifacts"]:
            st.subheader("建模产物")
            st.json(result["artifacts"])

        st.download_button(
            "下载建模配置",
            data=json.dumps(config.model_dump(), ensure_ascii=False, indent=2),
            file_name="modeling_config.json",
            mime="application/json",
        )
        st.download_button(
            "下载预测结果",
            data=result["prediction_df"].to_csv(index=False).encode("utf-8-sig"),
            file_name="model_prediction.csv",
            mime="text/csv",
        )


def render_association_heatmap(df: pd.DataFrame, analysis: dict) -> None:
    inferred_types = analysis["inferred_types"]
    eligible_columns = [
        col
        for col, kind in inferred_types.items()
        if col in df.columns and kind in {"numerical", "categorical", "boolean", "datetime"}
    ]

    st.markdown("**混合类型关系热力图**")
    if len(eligible_columns) < 2:
        st.info("可用于关系分析的字段不足 2 个。")
        return

    default_columns = eligible_columns[: min(8, len(eligible_columns))]
    selected_columns = st.multiselect(
        "选择参与热力图的字段",
        options=eligible_columns,
        default=default_columns,
        key="association_columns",
    )
    if len(selected_columns) < 2:
        st.info("请至少选择 2 个字段。")
        return

    eda_df, sampled = sample_for_eda(df[selected_columns], max_rows=4000)
    if sampled:
        st.caption("热力图基于 4000 行随机样本计算，用于提升交互速度。")

    matrix = pd.DataFrame(index=selected_columns, columns=selected_columns, dtype=float)
    for column in selected_columns:
        matrix.loc[column, column] = 1.0

    for i, column_a in enumerate(selected_columns):
        for column_b in selected_columns[i + 1 :]:
            score = compute_association_score(eda_df, inferred_types, column_a, column_b)
            if score is None:
                continue
            matrix.loc[column_a, column_b] = round(score, 3)
            matrix.loc[column_b, column_a] = round(score, 3)

    fig = px.imshow(
        matrix.fillna(0.0),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1,
        title="混合类型变量关系强度热力图（0-1）",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("数值-数值使用 Pearson 相关系数绝对值，分类型-分类型使用 Cramer's V，分类型-数值型使用相关比，时间变量按时间序编码参与计算。")


def render_relationship_detail(df: pd.DataFrame, analysis: dict) -> None:
    inferred_types = analysis["inferred_types"]
    eligible_columns = [
        col
        for col, kind in inferred_types.items()
        if col in df.columns and kind in {"numerical", "categorical", "boolean", "datetime"}
    ]

    st.markdown("**双变量关系分析**")
    if len(eligible_columns) < 2:
        st.info("可用于关系分析的字段不足 2 个。")
        return

    x_col = st.selectbox("变量 X", options=eligible_columns, key="relation_x")
    y_candidates = [col for col in eligible_columns if col != x_col]
    y_col = st.selectbox("变量 Y", options=y_candidates, key="relation_y")

    x_type = inferred_types.get(x_col, "unknown")
    y_type = inferred_types.get(y_col, "unknown")
    st.caption(f"当前组合：{x_col}（{TYPE_LABELS.get(x_type, x_type)}） vs {y_col}（{TYPE_LABELS.get(y_type, y_type)}）")

    eda_df, sampled = sample_for_eda(df[[x_col, y_col]], max_rows=5000)
    if sampled:
        st.caption("双变量图形基于 5000 行随机样本生成。")

    if x_type == "numerical" and y_type == "numerical":
        numeric_df = eda_df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if numeric_df.empty:
            st.warning("这两个字段没有足够的有效数值用于绘图。")
            return
        corr = numeric_df[x_col].corr(numeric_df[y_col])
        st.metric("Pearson 相关系数", "-" if pd.isna(corr) else round(float(corr), 4))
        fig = px.scatter(numeric_df, x=x_col, y=y_col, title=f"{x_col} 与 {y_col} 散点图")
        st.plotly_chart(fig, use_container_width=True)
        return

    if x_type in CAT_LIKE_TYPES and y_type == "numerical":
        cat_col, num_col = x_col, y_col
    elif y_type in CAT_LIKE_TYPES and x_type == "numerical":
        cat_col, num_col = y_col, x_col
    else:
        cat_col, num_col = None, None

    if cat_col and num_col:
        frame = pd.DataFrame(
            {
                cat_col: eda_df[cat_col].astype(str),
                num_col: pd.to_numeric(eda_df[num_col], errors="coerce"),
            }
        ).dropna()
        if frame.empty:
            st.warning("该组合没有足够的有效样本。")
            return
        top_categories = frame[cat_col].value_counts().head(20).index
        frame = frame[frame[cat_col].isin(top_categories)]
        eta = correlation_ratio(frame[cat_col], frame[num_col])
        st.metric("关系强度（相关比）", "-" if eta is None else round(float(eta), 4))
        fig = px.box(frame, x=cat_col, y=num_col, points="outliers", title=f"{cat_col} 对 {num_col} 的分布影响")
        st.plotly_chart(fig, use_container_width=True)
        stats_df = (
            frame.groupby(cat_col)[num_col]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
            .rename(columns={"count": "样本数", "mean": "均值", "median": "中位数", "std": "标准差"})
        )
        st.dataframe(stats_df, use_container_width=True)
        return

    if x_type in CAT_LIKE_TYPES and y_type in CAT_LIKE_TYPES:
        frame = pd.DataFrame({x_col: eda_df[x_col].astype(str), y_col: eda_df[y_col].astype(str)}).dropna()
        if frame.empty:
            st.warning("该组合没有足够的有效样本。")
            return
        top_x = frame[x_col].value_counts().head(12).index
        top_y = frame[y_col].value_counts().head(12).index
        frame = frame[frame[x_col].isin(top_x) & frame[y_col].isin(top_y)]
        crosstab = pd.crosstab(frame[x_col], frame[y_col])
        score = cramers_v(frame[x_col], frame[y_col])
        st.metric("关系强度（Cramer's V）", "-" if score is None else round(float(score), 4))
        fig = px.imshow(crosstab, text_auto=True, aspect="auto", title=f"{x_col} 与 {y_col} 列联热力图")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(crosstab, use_container_width=True)
        return

    if x_type == "datetime" and y_type == "numerical":
        dt_col, num_col = x_col, y_col
    elif y_type == "datetime" and x_type == "numerical":
        dt_col, num_col = y_col, x_col
    else:
        dt_col, num_col = None, None

    if dt_col and num_col:
        dt_series = parse_datetime_series(eda_df[dt_col], dt_col)
        numeric_series = pd.to_numeric(eda_df[num_col], errors="coerce")
        frame = pd.DataFrame({"时间": dt_series, "数值": numeric_series}).dropna()
        if frame.empty:
            st.warning("该组合没有足够的有效样本。")
            return
        agg_method = st.selectbox(
            "聚合方式",
            options=["mean", "sum", "median", "max", "min", "count"],
            format_func=lambda x: {
                "mean": "均值",
                "sum": "求和",
                "median": "中位数",
                "max": "最大值",
                "min": "最小值",
                "count": "计数",
            }[x],
            key="relation_time_numeric_agg",
        )
        requested_grain = st.selectbox("时间粒度", options=TIME_GRAIN_OPTIONS, key="relation_time_numeric_grain")
        aggregated, actual_grain = build_time_numeric_frame(frame["时间"], frame["数值"], dt_col, requested_grain, agg_method)
        metric_cols = st.columns(3)
        metric_cols[0].metric("最早时间", format_time_value(frame["时间"].min(), actual_grain))
        metric_cols[1].metric("最晚时间", format_time_value(frame["时间"].max(), actual_grain))
        metric_cols[2].metric("实际粒度", actual_grain)
        fig = px.line(
            aggregated,
            x="时间",
            y=agg_method,
            markers=True,
            title=f"{dt_col} 与 {num_col} 的时间趋势",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(aggregated, use_container_width=True)
        return

    if x_type == "datetime" and y_type in CAT_LIKE_TYPES:
        dt_col, cat_col = x_col, y_col
    elif y_type == "datetime" and x_type in CAT_LIKE_TYPES:
        dt_col, cat_col = y_col, x_col
    else:
        dt_col, cat_col = None, None

    if dt_col and cat_col:
        dt_series = parse_datetime_series(eda_df[dt_col], dt_col)
        category_series = eda_df[cat_col].astype(str)
        frame = pd.DataFrame({"时间": dt_series, "类别": category_series}).dropna()
        if frame.empty:
            st.warning("该组合没有足够的有效样本。")
            return
        requested_grain = st.selectbox("时间粒度", options=TIME_GRAIN_OPTIONS, key="relation_time_category_grain")
        aggregated, actual_grain = build_time_category_frame(frame["时间"], frame["类别"], dt_col, requested_grain)
        metric_cols = st.columns(3)
        metric_cols[0].metric("最早时间", format_time_value(frame["时间"].min(), actual_grain))
        metric_cols[1].metric("最晚时间", format_time_value(frame["时间"].max(), actual_grain))
        metric_cols[2].metric("实际粒度", actual_grain)
        fig = px.line(
            aggregated,
            x="时间",
            y="记录数",
            color="类别",
            markers=True,
            title=f"{dt_col} 与 {cat_col} 的时间分布",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(aggregated, use_container_width=True)
        return

    st.info("当前组合暂未提供专门图形，建议换成数值型、分类型或时间型字段组合。")


def render_statistical_test_section(df: pd.DataFrame, analysis: dict) -> None:
    inferred_types = analysis["inferred_types"]
    eligible_columns = [
        col
        for col, kind in inferred_types.items()
        if col in df.columns and kind in {"numerical", "categorical", "boolean", "datetime"}
    ]

    st.markdown("**统计检验**")
    if len(eligible_columns) < 2:
        st.info("可用于统计检验的字段不足 2 个。")
        return

    test_x = st.selectbox("检验变量 X", options=eligible_columns, key="test_x")
    test_y = st.selectbox(
        "检验变量 Y",
        options=[col for col in eligible_columns if col != test_x],
        key="test_y",
    )

    result = run_statistical_test(df, inferred_types, test_x, test_y)
    if result["status"] == "unsupported":
        st.info(result["summary"])
        return
    if result["status"] == "insufficient":
        st.warning(result["summary"])
        return

    diagnostic = run_statistical_diagnostic(df, inferred_types, test_x, test_y)
    st.markdown("**前提诊断与检验建议**")
    if diagnostic["status"] == "ok":
        st.write(diagnostic["summary"])
        if diagnostic.get("recommended_test"):
            st.caption(f"建议优先采用：{diagnostic['recommended_test']}")
        for item in diagnostic.get("items", []):
            if item["passed"]:
                st.success(f"{item['name']} | {item['target']}：{item['detail']}")
            else:
                st.warning(f"{item['name']} | {item['target']}：{item['detail']}")
        for table_item in diagnostic.get("extra_tables", []):
            st.markdown(f"**{table_item['name']}**")
            st.dataframe(table_item["data"], use_container_width=True)
    elif diagnostic["status"] == "insufficient":
        st.info(diagnostic["summary"])
    else:
        st.info(diagnostic["summary"])

    st.divider()
    st.markdown("**检验结果**")

    left, middle, right = st.columns(3)
    left.metric("检验方法", result["method"])
    middle.metric("样本量", result["sample_size"])
    if result["p_value"] is not None:
        right.metric("p 值", f"{float(result['p_value']):.4g}")

    st.write(result["summary"])

    stat_cols = st.columns(3)
    stat_cols[0].metric(result["statistic_name"], round(float(result["statistic"]), 4))
    if result["effect_size"] is not None:
        stat_cols[1].metric(result["effect_size_name"], round(float(result["effect_size"]), 4))
    stat_cols[2].metric("是否显著", "是" if result["significant"] else "否")

    if result["significant"] is True:
        st.success("在 0.05 显著性水平下，结果具有统计显著性。")
    else:
        st.info("在 0.05 显著性水平下，结果未达到统计显著。")

    for table_item in result.get("extra_tables", []):
        st.markdown(f"**{table_item['name']}**")
        st.dataframe(table_item["data"], use_container_width=True)

    st.divider()
    st.markdown("**多重共线性诊断**")
    numeric_candidates = [
        col
        for col, kind in inferred_types.items()
        if kind == "numerical" and col in df.columns
    ]
    if len(numeric_candidates) < 2:
        st.info("当前数值字段不足 2 个，无法计算 VIF。")
        return

    default_vif_columns = numeric_candidates[: min(8, len(numeric_candidates))]
    selected_vif_columns = st.multiselect(
        "选择参与 VIF 诊断的数值字段",
        options=numeric_candidates,
        default=default_vif_columns,
        key="vif_columns",
    )
    vif_table = compute_vif_table(df, inferred_types, selected_vif_columns)
    if vif_table.empty:
        st.info("有效数值样本不足，或所选字段无法形成可计算的 VIF 矩阵。")
    else:
        st.dataframe(vif_table, use_container_width=True)
        high_risk_count = int((vif_table["VIF"] >= 10).sum())
        medium_risk_count = int(((vif_table["VIF"] >= 5) & (vif_table["VIF"] < 10)).sum())
        if high_risk_count > 0:
            st.warning(f"当前有 {high_risk_count} 个字段 VIF >= 10，建议重点检查共线性。")
        elif medium_risk_count > 0:
            st.info(f"当前有 {medium_risk_count} 个字段 VIF 介于 5 到 10，存在一定共线性风险。")
        else:
            st.success("当前所选数值字段的 VIF 整体处于较低风险水平。")


def render_insight_section(df: pd.DataFrame, analysis: dict) -> None:
    st.markdown("**自动洞察摘要**")
    insights = generate_eda_insights(df, analysis, top_k=10)
    if not insights:
        st.info("当前没有生成明显的自动洞察结果。")
        return

    level_renderers = {
        "warning": st.warning,
        "success": st.success,
        "info": st.info,
    }
    for item in insights:
        renderer = level_renderers.get(item["level"], st.info)
        renderer(f"{item['title']}：{item['detail']}")


def render_eda_section(df: pd.DataFrame, analysis: dict) -> None:
    st.subheader("基础 EDA 可视化")
    inferred_types = analysis["inferred_types"]
    numerical_columns = [col for col, kind in inferred_types.items() if kind == "numerical" and col in df.columns]
    categorical_columns = [col for col, kind in inferred_types.items() if kind in CAT_LIKE_TYPES and col in df.columns]
    datetime_columns = [col for col, kind in inferred_types.items() if kind == "datetime" and col in df.columns]

    metric_cols = st.columns(4)
    metric_cols[0].metric("数值字段数", len(numerical_columns))
    metric_cols[1].metric("分类字段数", len(categorical_columns))
    metric_cols[2].metric("时间字段数", len(datetime_columns))
    metric_cols[3].metric("总字段数", df.shape[1])

    tab_numeric, tab_categorical, tab_datetime, tab_relation, tab_test, tab_insight = st.tabs(
        ["数值变量", "分类变量", "时间变量", "关系分析", "统计检验", "自动洞察"]
    )

    with tab_numeric:
        if numerical_columns:
            selected_num = st.selectbox("选择一个数值字段", options=numerical_columns, key="eda_numeric_column")
            numeric_series = pd.to_numeric(df[selected_num], errors="coerce").dropna()
            if not numeric_series.empty:
                plot_df = pd.DataFrame({"数值": numeric_series})
                plot_col1, plot_col2 = st.columns(2)
                with plot_col1:
                    fig_hist = px.histogram(plot_df, x="数值", nbins=30, title=f"{selected_num} 分布直方图")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with plot_col2:
                    fig_box = px.box(plot_df, y="数值", points="outliers", title=f"{selected_num} 箱线图")
                    st.plotly_chart(fig_box, use_container_width=True)

            if len(numerical_columns) >= 2:
                corr_df = df[numerical_columns].apply(pd.to_numeric, errors="coerce").corr()
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=".2f",
                    aspect="auto",
                    title="数值字段相关系数热力图",
                    color_continuous_scale="Blues",
                    zmin=-1,
                    zmax=1,
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("当前数据中没有可用于数值分析的字段。")

    with tab_categorical:
        if categorical_columns:
            selected_cat = st.selectbox("选择一个分类字段", options=categorical_columns, key="eda_category_column")
            value_counts = df[selected_cat].astype(str).value_counts(dropna=False).head(20).reset_index()
            value_counts.columns = [selected_cat, "频次"]
            fig_bar = px.bar(value_counts, x=selected_cat, y="频次", title=f"{selected_cat} 前 20 类频次分布")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(value_counts, use_container_width=True)
        else:
            st.info("当前数据中没有可用于分类分析的字段。")

    with tab_datetime:
        if datetime_columns:
            selected_dt = st.selectbox("选择一个时间字段", options=datetime_columns, key="eda_datetime_column")
            dt_series = parse_datetime_series(df[selected_dt], selected_dt)
            valid_dt = dt_series.dropna()
            if valid_dt.empty:
                st.warning("该时间字段未能解析出有效时间值。")
            else:
                requested_grain = st.selectbox("时间趋势粒度", options=TIME_GRAIN_OPTIONS, key="eda_datetime_grain")
                trend_df, actual_grain = build_time_coverage_frame(valid_dt, selected_dt, requested_grain)
                summary_cols = st.columns(3)
                summary_cols[0].metric("最早时间", format_time_value(valid_dt.min(), actual_grain))
                summary_cols[1].metric("最晚时间", format_time_value(valid_dt.max(), actual_grain))
                summary_cols[2].metric("实际粒度", actual_grain)

                fig_line = px.line(
                    trend_df,
                    x="时间",
                    y="记录数",
                    markers=True,
                    title=f"{selected_dt} 时间覆盖趋势",
                )
                st.plotly_chart(fig_line, use_container_width=True)
                st.dataframe(trend_df, use_container_width=True)
        else:
            st.info("当前数据中没有可用于时间分析的字段。")

    with tab_relation:
        render_association_heatmap(df, analysis)
        st.divider()
        render_relationship_detail(df, analysis)

    with tab_test:
        render_statistical_test_section(df, analysis)

    with tab_insight:
        render_insight_section(df, analysis)


init_session_state()

st.title("数据预处理原型 V1.8")
st.caption("支持 SQLite 历史浏览、数据回读、字段级人工标注、基础 EDA 与预处理流程。")

render_data_source_panel()
render_task_history_panel(st.session_state.get("current_dataset_id"))

current_df = st.session_state.get("current_df")
current_dataset_id = st.session_state.get("current_dataset_id")
current_dataset_name = st.session_state.get("current_dataset_name")
current_source_file_name = st.session_state.get("current_source_file_name")
current_source_mode = st.session_state.get("current_source_mode")

if current_df is not None and current_dataset_id is not None:
    st.success(
        f"当前工作数据：{current_dataset_name} | dataset_id={current_dataset_id} | 来源={current_source_mode} | 维度={current_df.shape[0]}x{current_df.shape[1]}"
    )

    analysis = analyze_dataframe(current_df)
    overview_tab, eda_tab, preprocess_tab, modeling_tab = st.tabs(["数据概览", "EDA 可视化", "预处理配置", "建模分析"])

    with overview_tab:
        st.subheader("数据预览")
        st.dataframe(current_df.head(20), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("字段类型识别结果")
            st.dataframe(build_type_table(analysis), use_container_width=True)
        with right:
            st.subheader("数据集概况")
            st.json(build_dataset_summary(analysis["quality_report"]["dataset_summary"]))

        type_review_suggestions = analysis.get("type_review_suggestions", {})
        if type_review_suggestions:
            st.warning("以下字段建议人工确认类型或补充语义信息。")
            for column, suggestions in type_review_suggestions.items():
                st.write(f"- {column}：{'；'.join(suggestions)}")

        st.subheader("字段质量报告")
        profile_df = pd.DataFrame(analysis["quality_report"]["column_profiles"])
        st.dataframe(build_quality_table(profile_df), use_container_width=True)

    with eda_tab:
        render_eda_section(current_df, analysis)

    with preprocess_tab:
        st.subheader("全局预处理配置")
        all_columns = list(current_df.columns)
        numeric_strategy = st.selectbox(
            "数值型默认缺失值策略",
            options=list(MISSING_OPTIONS.keys()),
            index=list(MISSING_OPTIONS.keys()).index("median"),
            format_func=lambda x: MISSING_OPTIONS[x],
        )
        categorical_strategy = st.selectbox(
            "分类型默认缺失值策略",
            options=["mode", "constant", "none"],
            index=0,
            format_func=lambda x: MISSING_OPTIONS[x],
        )
        constant_value = st.text_input("默认固定填充值", value="missing")
        outlier_strategy = st.selectbox(
            "默认异常值策略",
            options=list(OUTLIER_OPTIONS.keys()),
            index=0,
            format_func=lambda x: OUTLIER_OPTIONS[x],
        )
        lower_q = st.slider("默认异常下分位点", min_value=0.0, max_value=0.2, value=0.01, step=0.01)
        upper_q = st.slider("默认异常上分位点", min_value=0.8, max_value=1.0, value=0.99, step=0.01)
        encoding = st.selectbox(
            "默认编码方式",
            options=list(ENCODING_OPTIONS.keys()),
            index=list(ENCODING_OPTIONS.keys()).index("onehot"),
            format_func=lambda x: ENCODING_OPTIONS[x],
        )
        scaling = st.selectbox(
            "默认缩放方式",
            options=list(SCALING_OPTIONS.keys()),
            index=0,
            format_func=lambda x: SCALING_OPTIONS[x],
        )
        drop_columns = st.multiselect("直接删除字段", options=all_columns)
        remove_duplicates = st.checkbox("删除重复行", value=True)
        drop_all_null_columns = st.checkbox("删除全空列", value=True)
        drop_single_value_columns = st.checkbox("删除单一值列", value=True)
        save_artifacts = st.checkbox("保存运行产物到 data/artifacts", value=True)
        enable_split = st.checkbox("生成训练集/测试集划分", value=False)
        test_size = st.slider("测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

        st.subheader("字段级配置与元数据标注")
        column_configs = build_column_configs(all_columns, analysis)

        config = PreprocessConfig(
            drop_columns=drop_columns,
            type_overrides={},
            column_configs=column_configs,
            missing_numeric=numeric_strategy,
            missing_categorical=categorical_strategy,
            missing_constant_value=constant_value,
            outlier_strategy=outlier_strategy,
            outlier_lower_quantile=lower_q,
            outlier_upper_quantile=upper_q,
            encoding=encoding,
            scaling=scaling,
            remove_duplicates=remove_duplicates,
            drop_all_null_columns=drop_all_null_columns,
            drop_single_value_columns=drop_single_value_columns,
            save_artifacts=save_artifacts,
            split=SplitConfig(test_size=test_size) if enable_split else None,
        )

        if st.button("执行预处理", type="primary"):
            field_metadata = {
                column: column_config.metadata.model_dump()
                for column, column_config in config.column_configs.items()
            }
            result = run_preprocess_with_source(
                current_df,
                config,
                source_name=current_source_file_name,
                dataset_id=current_dataset_id,
                field_metadata=field_metadata,
                persist_output=True,
            )
            preprocess_storage_result = result["preprocess_storage"]

            st.subheader("处理结果概览")
            st.write(
                {
                    "原始数据维度": list(current_df.shape),
                    "清洗后维度": list(result["cleaned_df"].shape),
                    "转换后维度": list(result["transformed_df"].shape),
                }
            )

            st.subheader("最终字段类型")
            st.json({key: TYPE_LABELS.get(value, value) for key, value in result["resolved_types"].items()})

            if result["validation_findings"]:
                st.subheader("规则校验结果")
                st.warning("以下字段存在元数据约束或语义检查问题，建议优先处理。")
                st.json(result["validation_findings"])

            st.subheader("处理日志")
            st.json(result["run_log"])

            if result["artifacts"]:
                st.subheader("已保存产物")
                st.json(result["artifacts"])

            st.subheader("SQLite 落库信息")
            st.json(
                {
                    "task_run": result["task"],
                    "preprocessed_data": preprocess_storage_result,
                }
            )

            st.subheader("处理后数据预览")
            st.dataframe(result["transformed_df"].head(20), use_container_width=True)

            st.download_button(
                "下载预处理配置",
                data=json.dumps(config.model_dump(), ensure_ascii=False, indent=2),
                file_name="preprocess_config.json",
                mime="application/json",
            )
            st.download_button(
                "下载质量报告",
                data=json.dumps(result["analysis"]["quality_report"], ensure_ascii=False, indent=2, default=str),
                file_name="quality_report.json",
                mime="application/json",
            )
            st.download_button(
                "下载处理后数据",
                data=result["transformed_df"].to_csv(index=False).encode("utf-8-sig"),
                file_name="processed_data.csv",
                mime="text/csv",
            )

    with modeling_tab:
        render_modeling_section(
            current_df,
            analysis,
            dataset_id=current_dataset_id,
            source_name=current_source_file_name,
        )
else:
    st.info("先上传数据或从 SQLite 中加载历史数据，然后再进行 EDA 和预处理。")
