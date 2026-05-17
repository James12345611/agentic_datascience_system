from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from app.core.datetime_utils import parse_datetime_series


CAT_LIKE_TYPES = {"categorical", "boolean"}


def run_statistical_test(
    df: pd.DataFrame,
    inferred_types: dict[str, str],
    x_col: str,
    y_col: str,
) -> dict[str, Any]:
    x_type = inferred_types.get(x_col, "unknown")
    y_type = inferred_types.get(y_col, "unknown")

    if x_type == "numerical" and y_type == "numerical":
        return _pearson_test(df[x_col], df[y_col], x_col, y_col)

    if x_type in CAT_LIKE_TYPES and y_type == "numerical":
        return _categorical_numeric_test(df[x_col], df[y_col], x_col, y_col)
    if y_type in CAT_LIKE_TYPES and x_type == "numerical":
        return _categorical_numeric_test(df[y_col], df[x_col], y_col, x_col)

    if x_type in CAT_LIKE_TYPES and y_type in CAT_LIKE_TYPES:
        return _categorical_categorical_test(df[x_col], df[y_col], x_col, y_col)

    if x_type == "datetime" and y_type == "numerical":
        return _datetime_numeric_test(df[x_col], df[y_col], x_col, y_col)
    if y_type == "datetime" and x_type == "numerical":
        return _datetime_numeric_test(df[y_col], df[x_col], y_col, x_col)

    return {
        "status": "unsupported",
        "title": "当前组合暂不支持统计检验",
        "summary": "目前支持数值-数值、分类-数值、分类-分类、时间-数值四类统计检验。",
    }


def run_statistical_diagnostic(
    df: pd.DataFrame,
    inferred_types: dict[str, str],
    x_col: str,
    y_col: str,
) -> dict[str, Any]:
    x_type = inferred_types.get(x_col, "unknown")
    y_type = inferred_types.get(y_col, "unknown")

    if x_type == "numerical" and y_type == "numerical":
        return _numeric_numeric_diagnostic(df[x_col], df[y_col], x_col, y_col)

    if x_type in CAT_LIKE_TYPES and y_type == "numerical":
        return _categorical_numeric_diagnostic(df[x_col], df[y_col], x_col, y_col)
    if y_type in CAT_LIKE_TYPES and x_type == "numerical":
        return _categorical_numeric_diagnostic(df[y_col], df[x_col], y_col, x_col)

    if x_type in CAT_LIKE_TYPES and y_type in CAT_LIKE_TYPES:
        return _categorical_categorical_diagnostic(df[x_col], df[y_col], x_col, y_col)

    return {
        "status": "unsupported",
        "title": "当前组合暂不支持诊断",
        "summary": "目前支持数值-数值、分类-数值、分类-分类三类前提诊断。",
        "items": [],
        "recommended_test": None,
        "extra_tables": [],
    }


def compute_vif_table(
    df: pd.DataFrame,
    inferred_types: dict[str, str],
    selected_columns: list[str] | None = None,
) -> pd.DataFrame:
    numerical_columns = [
        col
        for col, kind in inferred_types.items()
        if kind == "numerical" and col in df.columns and (selected_columns is None or col in selected_columns)
    ]
    if len(numerical_columns) < 2:
        return pd.DataFrame()

    numeric_df = df[numerical_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if numeric_df.shape[0] < 5 or numeric_df.shape[1] < 2:
        return pd.DataFrame()

    constant_columns = [col for col in numeric_df.columns if numeric_df[col].nunique(dropna=True) <= 1]
    numeric_df = numeric_df.drop(columns=constant_columns, errors="ignore")
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    values = numeric_df.to_numpy(dtype=float)
    for index, column in enumerate(numeric_df.columns):
        try:
            vif_value = float(variance_inflation_factor(values, index))
        except Exception:
            vif_value = float("inf")
        rows.append({"字段": column, "VIF": round(vif_value, 4)})

    result = pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)
    result["风险等级"] = result["VIF"].map(_label_vif_risk)
    return result


def generate_eda_insights(df: pd.DataFrame, analysis: dict, top_k: int = 8) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    inferred_types = analysis["inferred_types"]
    quality_report = analysis.get("quality_report", {})
    column_profiles = quality_report.get("column_profiles", [])

    for profile in column_profiles:
        missing_ratio = float(profile.get("missing_ratio", 0))
        if missing_ratio >= 0.3:
            insights.append(
                {
                    "level": "warning",
                    "title": f"{profile['column_name']} 缺失比例较高",
                    "detail": f"缺失比例为 {missing_ratio:.1%}，建议优先确认该字段是否保留，以及采用何种填补策略。",
                }
            )

    numerical_columns = [col for col, kind in inferred_types.items() if kind == "numerical" and col in df.columns]
    if len(numerical_columns) >= 2:
        corr_frame = df[numerical_columns].apply(pd.to_numeric, errors="coerce")
        corr_matrix = corr_frame.corr()
        pairs: list[tuple[str, str, float]] = []
        for i, column_a in enumerate(numerical_columns):
            for column_b in numerical_columns[i + 1 :]:
                corr = corr_matrix.loc[column_a, column_b]
                if pd.notna(corr):
                    pairs.append((column_a, column_b, float(corr)))
        pairs = sorted(pairs, key=lambda item: abs(item[2]), reverse=True)
        for column_a, column_b, corr in pairs[:2]:
            if abs(corr) >= 0.7:
                insights.append(
                    {
                        "level": "info",
                        "title": f"{column_a} 与 {column_b} 存在较强相关",
                        "detail": f"Pearson 相关系数为 {corr:.3f}。若后续进入回归或机器学习建模，可关注多重共线性风险。",
                    }
                )

    vif_table = compute_vif_table(df, inferred_types)
    if not vif_table.empty:
        high_vif = vif_table[vif_table["VIF"] >= 10].head(3)
        for _, row in high_vif.iterrows():
            insights.append(
                {
                    "level": "warning",
                    "title": f"{row['字段']} 存在较高共线性风险",
                    "detail": f"VIF 为 {row['VIF']:.2f}，若用于回归建模，建议考虑删减、合成或正则化处理。",
                }
            )

    categorical_columns = [col for col, kind in inferred_types.items() if kind in CAT_LIKE_TYPES and col in df.columns]
    for cat_col in categorical_columns[:4]:
        distribution = df[cat_col].astype(str).value_counts(dropna=False, normalize=True)
        if not distribution.empty and float(distribution.iloc[0]) >= 0.8:
            insights.append(
                {
                    "level": "warning",
                    "title": f"{cat_col} 类别分布高度集中",
                    "detail": f"占比最高类别达到 {float(distribution.iloc[0]):.1%}，后续建模时可能带来类别不平衡问题。",
                }
            )

    datetime_columns = [col for col, kind in inferred_types.items() if kind == "datetime" and col in df.columns]
    for dt_col in datetime_columns[:3]:
        dt_series = parse_datetime_series(df[dt_col], dt_col).dropna()
        if dt_series.empty:
            insights.append(
                {
                    "level": "warning",
                    "title": f"{dt_col} 未解析出有效时间",
                    "detail": "建议检查原始格式、字段标注或时间解析规则。",
                }
            )
            continue
        min_time = dt_series.min()
        max_time = dt_series.max()
        insights.append(
            {
                "level": "info",
                "title": f"{dt_col} 时间覆盖范围已识别",
                "detail": f"当前识别到的时间范围为 {min_time.strftime('%Y-%m-%d')} 到 {max_time.strftime('%Y-%m-%d')}。",
            }
        )

    insights.extend(_generate_pairwise_test_insights(df, inferred_types))

    priority = {"warning": 0, "success": 1, "info": 2}
    insights = sorted(insights, key=lambda item: priority.get(item["level"], 9))
    return insights[:top_k]


def _generate_pairwise_test_insights(
    df: pd.DataFrame,
    inferred_types: dict[str, str],
) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    numerical_columns = [col for col, kind in inferred_types.items() if kind == "numerical" and col in df.columns]
    categorical_columns = [col for col, kind in inferred_types.items() if kind in CAT_LIKE_TYPES and col in df.columns]

    strongest_numeric: tuple[str, str, float] | None = None
    for i, x_col in enumerate(numerical_columns[:6]):
        for y_col in numerical_columns[i + 1 : 6]:
            result = _pearson_test(df[x_col], df[y_col], x_col, y_col)
            if result["status"] != "ok":
                continue
            effect = abs(float(result["effect_size"]))
            if strongest_numeric is None or effect > strongest_numeric[2]:
                strongest_numeric = (x_col, y_col, effect)
    if strongest_numeric and strongest_numeric[2] >= 0.7:
        insights.append(
            {
                "level": "success",
                "title": f"{strongest_numeric[0]} 与 {strongest_numeric[1]} 呈现较强线性关系",
                "detail": f"相关强度约为 {strongest_numeric[2]:.3f}，可进一步检查业务解释和建模用途。",
            }
        )

    for cat_col in categorical_columns[:4]:
        for num_col in numerical_columns[:4]:
            result = _categorical_numeric_test(df[cat_col], df[num_col], cat_col, num_col)
            if result["status"] == "ok" and result["p_value"] is not None and float(result["p_value"]) < 0.05:
                insights.append(
                    {
                        "level": "success",
                        "title": f"{cat_col} 对 {num_col} 可能存在显著影响",
                        "detail": f"{result['method']} 的 p 值为 {float(result['p_value']):.4f}，建议继续查看分组分布图和业务含义。",
                    }
                )
                return insights

    return insights


def _pearson_test(series_x: pd.Series, series_y: pd.Series, x_col: str, y_col: str) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            x_col: pd.to_numeric(series_x, errors="coerce"),
            y_col: pd.to_numeric(series_y, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 3:
        return _insufficient_result("有效样本不足，无法执行 Pearson 相关检验。")

    corr, p_value = stats.pearsonr(frame[x_col], frame[y_col])
    return {
        "status": "ok",
        "title": "Pearson 相关检验",
        "summary": f"{x_col} 与 {y_col} 的线性相关系数为 {corr:.4f}，p 值为 {p_value:.4g}。",
        "method": "Pearson 相关检验",
        "sample_size": int(len(frame)),
        "statistic_name": "相关系数",
        "statistic": float(corr),
        "p_value": float(p_value),
        "effect_size_name": "相关强度",
        "effect_size": float(corr),
        "significant": bool(p_value < 0.05),
        "extra_tables": [],
    }


def _categorical_numeric_test(
    category_series: pd.Series,
    numeric_series: pd.Series,
    category_col: str,
    numeric_col: str,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            category_col: category_series.astype(str),
            numeric_col: pd.to_numeric(numeric_series, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return _insufficient_result("有效样本不足，无法执行分组差异检验。")

    counts = frame[category_col].value_counts()
    valid_categories = counts[counts >= 2].index
    frame = frame[frame[category_col].isin(valid_categories)]
    if frame.empty or frame[category_col].nunique() < 2:
        return _insufficient_result("有效类别不足 2 个，无法执行分组差异检验。")

    groups = [group[numeric_col].dropna().to_numpy() for _, group in frame.groupby(category_col)]
    if min(len(group) for group in groups) < 2:
        return _insufficient_result("至少有一个类别样本数不足 2，无法执行分组差异检验。")

    group_summary = (
        frame.groupby(category_col)[numeric_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(columns={"count": "样本数", "mean": "均值", "median": "中位数", "std": "标准差"})
    )

    if frame[category_col].nunique() == 2:
        statistic, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
        effect_size = _cohens_d(groups[0], groups[1])
        return {
            "status": "ok",
            "title": "双样本均值差异检验",
            "summary": f"{category_col} 的两组在 {numeric_col} 上进行 Welch t 检验，p 值为 {p_value:.4g}。",
            "method": "Welch t 检验",
            "sample_size": int(len(frame)),
            "statistic_name": "t 值",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size_name": "Cohen's d",
            "effect_size": None if effect_size is None else float(effect_size),
            "significant": bool(p_value < 0.05),
            "extra_tables": [{"name": "分组统计", "data": group_summary}],
        }

    statistic, p_value = stats.f_oneway(*groups)
    effect_size = _eta_squared(groups)
    return {
        "status": "ok",
        "title": "多组均值差异检验",
        "summary": f"{category_col} 的多组在 {numeric_col} 上进行单因素方差分析，p 值为 {p_value:.4g}。",
        "method": "单因素方差分析（ANOVA）",
        "sample_size": int(len(frame)),
        "statistic_name": "F 值",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_name": "Eta squared",
        "effect_size": None if effect_size is None else float(effect_size),
        "significant": bool(p_value < 0.05),
        "extra_tables": [{"name": "分组统计", "data": group_summary}],
    }


def _categorical_categorical_test(
    series_x: pd.Series,
    series_y: pd.Series,
    x_col: str,
    y_col: str,
) -> dict[str, Any]:
    frame = pd.DataFrame({x_col: series_x.astype(str), y_col: series_y.astype(str)}).dropna()
    if frame.empty:
        return _insufficient_result("有效样本不足，无法执行列联分析。")

    crosstab = pd.crosstab(frame[x_col], frame[y_col])
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return _insufficient_result("列联表维度不足，无法执行卡方检验。")

    chi2, p_value, _, expected = stats.chi2_contingency(crosstab)
    effect_size = _cramers_v_from_chi2(chi2, crosstab)
    expected_df = pd.DataFrame(expected, index=crosstab.index, columns=crosstab.columns)
    return {
        "status": "ok",
        "title": "分类变量独立性检验",
        "summary": f"{x_col} 与 {y_col} 的卡方检验 p 值为 {p_value:.4g}。",
        "method": "卡方独立性检验",
        "sample_size": int(crosstab.to_numpy().sum()),
        "statistic_name": "卡方值",
        "statistic": float(chi2),
        "p_value": float(p_value),
        "effect_size_name": "Cramer's V",
        "effect_size": None if effect_size is None else float(effect_size),
        "significant": bool(p_value < 0.05),
        "extra_tables": [
            {"name": "观测频数", "data": crosstab.reset_index()},
            {"name": "期望频数", "data": expected_df.reset_index()},
        ],
    }


def _datetime_numeric_test(
    datetime_series: pd.Series,
    numeric_series: pd.Series,
    datetime_col: str,
    numeric_col: str,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            datetime_col: parse_datetime_series(datetime_series, datetime_col),
            numeric_col: pd.to_numeric(numeric_series, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 3:
        return _insufficient_result("有效样本不足，无法执行时间趋势检验。")

    time_numeric = frame[datetime_col].astype("int64")
    corr, p_value = stats.pearsonr(time_numeric, frame[numeric_col])
    return {
        "status": "ok",
        "title": "时间趋势相关检验",
        "summary": f"{datetime_col} 与 {numeric_col} 的时间趋势相关检验 p 值为 {p_value:.4g}。",
        "method": "时间序编码 Pearson 相关检验",
        "sample_size": int(len(frame)),
        "statistic_name": "相关系数",
        "statistic": float(corr),
        "p_value": float(p_value),
        "effect_size_name": "趋势强度",
        "effect_size": float(corr),
        "significant": bool(p_value < 0.05),
        "extra_tables": [],
    }


def _numeric_numeric_diagnostic(
    series_x: pd.Series,
    series_y: pd.Series,
    x_col: str,
    y_col: str,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            x_col: pd.to_numeric(series_x, errors="coerce"),
            y_col: pd.to_numeric(series_y, errors="coerce"),
        }
    ).dropna()
    if len(frame) < 8:
        return _insufficient_diagnostic("有效样本不足，无法执行数值变量诊断。")

    x_values = frame[x_col]
    y_values = frame[y_col]
    x_norm = _safe_shapiro(x_values)
    y_norm = _safe_shapiro(y_values)

    items = [
        _diagnostic_item("正态性", x_col, x_norm["summary"], x_norm["passed"]),
        _diagnostic_item("正态性", y_col, y_norm["summary"], y_norm["passed"]),
    ]

    if x_norm["passed"] and y_norm["passed"]:
        recommended = "Pearson 相关检验"
        summary = "两变量均未明显违背正态性假设，优先推荐 Pearson 相关检验。"
    else:
        recommended = "Spearman 秩相关检验"
        summary = "至少有一个变量不满足正态性，建议同时关注 Spearman 秩相关。"

    return {
        "status": "ok",
        "title": "数值变量诊断",
        "summary": summary,
        "items": items,
        "recommended_test": recommended,
        "extra_tables": [],
    }


def _categorical_numeric_diagnostic(
    category_series: pd.Series,
    numeric_series: pd.Series,
    category_col: str,
    numeric_col: str,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            category_col: category_series.astype(str),
            numeric_col: pd.to_numeric(numeric_series, errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return _insufficient_diagnostic("有效样本不足，无法执行分类-数值诊断。")

    counts = frame[category_col].value_counts()
    valid_categories = counts[counts >= 3].index
    frame = frame[frame[category_col].isin(valid_categories)]
    if frame.empty or frame[category_col].nunique() < 2:
        return _insufficient_diagnostic("有效类别不足 2 个，无法执行分类-数值诊断。")

    groups = [group[numeric_col].dropna() for _, group in frame.groupby(category_col)]
    if min(len(group) for group in groups) < 3:
        return _insufficient_diagnostic("至少有一个类别样本数不足 3，无法执行分类-数值诊断。")

    normality_rows = []
    normality_pass = True
    for category_value, group in frame.groupby(category_col):
        normality = _safe_shapiro(group[numeric_col])
        normality_rows.append(
            {
                "类别": category_value,
                "样本数": int(len(group)),
                "Shapiro p值": normality["p_value"],
                "是否近似正态": "是" if normality["passed"] else "否",
            }
        )
        normality_pass = normality_pass and normality["passed"]

    levene_stat, levene_p = stats.levene(*[group.to_numpy() for group in groups], center="median")
    variance_pass = bool(levene_p >= 0.05)
    items = [
        _diagnostic_item(
            "组内正态性",
            numeric_col,
            "；".join(
                [f"{row['类别']} 组 Shapiro p值={_format_p_value(row['Shapiro p值'])}" for row in normality_rows]
            ),
            normality_pass,
        ),
        _diagnostic_item(
            "方差齐性",
            f"{category_col}->{numeric_col}",
            f"Levene 检验 p 值为 {_format_p_value(float(levene_p))}",
            variance_pass,
        ),
    ]

    if frame[category_col].nunique() == 2:
        if normality_pass and variance_pass:
            recommended = "独立样本 t 检验"
            summary = "满足正态性和方差齐性时，可优先使用独立样本 t 检验。"
        elif normality_pass:
            recommended = "Welch t 检验"
            summary = "正态性基本满足但方差不齐，优先使用 Welch t 检验。"
        else:
            recommended = "Mann-Whitney U 检验"
            summary = "正态性不满足，建议使用 Mann-Whitney U 非参数检验。"
    else:
        if normality_pass and variance_pass:
            recommended = "单因素方差分析（ANOVA）"
            summary = "组内正态性与方差齐性基本满足，可优先使用单因素方差分析。"
        else:
            recommended = "Kruskal-Wallis 检验"
            summary = "至少有一项前提不满足，建议使用 Kruskal-Wallis 非参数检验。"

    return {
        "status": "ok",
        "title": "分类-数值诊断",
        "summary": summary,
        "items": items,
        "recommended_test": recommended,
        "extra_tables": [{"name": "分组正态性诊断", "data": pd.DataFrame(normality_rows)}],
    }


def _categorical_categorical_diagnostic(
    series_x: pd.Series,
    series_y: pd.Series,
    x_col: str,
    y_col: str,
) -> dict[str, Any]:
    frame = pd.DataFrame({x_col: series_x.astype(str), y_col: series_y.astype(str)}).dropna()
    if frame.empty:
        return _insufficient_diagnostic("有效样本不足，无法执行分类变量诊断。")

    crosstab = pd.crosstab(frame[x_col], frame[y_col])
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return _insufficient_diagnostic("列联表维度不足，无法执行分类变量诊断。")

    _, _, _, expected = stats.chi2_contingency(crosstab)
    expected_df = pd.DataFrame(expected, index=crosstab.index, columns=crosstab.columns)
    min_expected = float(expected_df.min().min())
    low_expected_ratio = float((expected_df < 5).to_numpy().mean())
    pass_expected = bool(min_expected >= 1 and low_expected_ratio <= 0.2)

    items = [
        _diagnostic_item(
            "期望频数",
            f"{x_col}->{y_col}",
            f"最小期望频数为 {min_expected:.3f}，小于 5 的单元格占比为 {low_expected_ratio:.1%}",
            pass_expected,
        )
    ]

    if pass_expected:
        recommended = "卡方独立性检验"
        summary = "期望频数条件基本满足，可以直接使用卡方独立性检验。"
    else:
        recommended = "合并稀疏类别后再做卡方检验"
        summary = "期望频数条件较弱，建议先合并稀疏类别，再执行卡方检验。"

    return {
        "status": "ok",
        "title": "分类变量诊断",
        "summary": summary,
        "items": items,
        "recommended_test": recommended,
        "extra_tables": [{"name": "期望频数表", "data": expected_df.reset_index()}],
    }


def _safe_shapiro(series: pd.Series) -> dict[str, Any]:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if len(valid) < 3:
        return {"passed": False, "p_value": None, "summary": "有效样本不足，无法执行 Shapiro-Wilk 检验。"}

    sample = valid.sample(n=5000, random_state=42) if len(valid) > 5000 else valid
    statistic, p_value = stats.shapiro(sample)
    passed = bool(p_value >= 0.05)
    return {
        "passed": passed,
        "p_value": float(p_value),
        "summary": f"Shapiro-Wilk 检验 p 值为 {_format_p_value(float(p_value))}",
    }


def _diagnostic_item(name: str, target: str, detail: str, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "target": target,
        "detail": detail,
        "passed": passed,
        "status_label": "通过" if passed else "未通过",
    }


def _label_vif_risk(value: float) -> str:
    if value >= 10:
        return "高风险"
    if value >= 5:
        return "中风险"
    return "低风险"


def _format_p_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4g}"


def _insufficient_diagnostic(message: str) -> dict[str, Any]:
    return {
        "status": "insufficient",
        "title": "诊断样本不足",
        "summary": message,
        "items": [],
        "recommended_test": None,
        "extra_tables": [],
    }


def _cohens_d(group_a: Any, group_b: Any) -> float | None:
    n_a = len(group_a)
    n_b = len(group_b)
    if n_a < 2 or n_b < 2:
        return None

    var_a = float(pd.Series(group_a).var(ddof=1))
    var_b = float(pd.Series(group_b).var(ddof=1))
    pooled_denom = n_a + n_b - 2
    if pooled_denom <= 0:
        return None

    pooled_std = sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / pooled_denom)
    if pooled_std == 0:
        return None
    return (float(pd.Series(group_a).mean()) - float(pd.Series(group_b).mean())) / pooled_std


def _eta_squared(groups: list[Any]) -> float | None:
    flat = [value for group in groups for value in group]
    if len(flat) < 3:
        return None
    grand_mean = sum(flat) / len(flat)
    ss_between = 0.0
    ss_total = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        group_mean = sum(group) / len(group)
        ss_between += len(group) * ((group_mean - grand_mean) ** 2)
        ss_total += sum((value - grand_mean) ** 2 for value in group)
    if ss_total <= 0:
        return None
    return ss_between / ss_total


def _cramers_v_from_chi2(chi2: float, crosstab: pd.DataFrame) -> float | None:
    n = crosstab.to_numpy().sum()
    if n <= 1:
        return None
    r, k = crosstab.shape
    denominator = min(r - 1, k - 1)
    if denominator <= 0:
        return None
    return sqrt((chi2 / n) / denominator)


def _insufficient_result(message: str) -> dict[str, Any]:
    return {
        "status": "insufficient",
        "title": "样本不足",
        "summary": message,
        "method": None,
        "sample_size": None,
        "statistic_name": None,
        "statistic": None,
        "p_value": None,
        "effect_size_name": None,
        "effect_size": None,
        "significant": None,
        "extra_tables": [],
    }
