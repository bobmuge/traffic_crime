# -*- coding: utf-8 -*-
from pathlib import Path
from math import lgamma, sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
FACTOR_DATA_PATH = BASE_DIR / "outputs" / "traffic_factor_analysis_data_with_license.csv"
ACCIDENT_DATA_PATH = BASE_DIR / "outputs" / "traffic_accident_merged_1980_2025.csv"
AGE_ACCIDENT_DATA_PATH = BASE_DIR / "outputs" / "traffic_accidents_by_driver_age.csv"

st.set_page_config(page_title="교통범죄 요인 분석", layout="wide")


st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2.2rem; }
    .hero {
        border-left: 6px solid #e84a5f;
        padding: 0.3rem 0 0.3rem 1.1rem;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin-bottom: 0.3rem;
        font-size: 2.35rem;
        color: #111827;
    }
    .hero p {
        margin: 0;
        color: #4b5563;
        font-size: 1.02rem;
    }
    .metric-card {
        background: #15182b;
        color: white;
        padding: 1.25rem 1.35rem;
        border-radius: 8px;
        border-left: 5px solid #e84a5f;
        min-height: 128px;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.14);
    }
    .metric-card .label {
        color: #cbd5e1;
        font-size: 0.92rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        color: #ff4f6d;
        font-size: 1.85rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .metric-card .note {
        color: #9ca3af;
        font-size: 0.84rem;
    }
    .method-box {
        background: #15182b;
        color: white;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 0.7rem 0 1rem 0;
    }
    .method-box code {
        color: #ff4f6d;
        background: transparent;
        font-size: 1rem;
    }
    .section-title {
        border-left: 5px solid #e84a5f;
        padding-left: 0.8rem;
        margin-top: 1.4rem;
        margin-bottom: 0.7rem;
        font-size: 1.35rem;
        font-weight: 800;
        color: #111827;
    }
    .insight {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.8rem;
    }
    .insight strong { color: #e11d48; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(factor_path: str, factor_modified_ns: int, accident_path: str, accident_modified_ns: int) -> pd.DataFrame:
    # 원자료가 교체되면 캐시도 함께 갱신되도록 각 파일의 수정 시간을 사용함.
    _ = (factor_modified_ns, accident_modified_ns)
    factor = pd.read_csv(factor_path, encoding="utf-8-sig")
    accident = pd.read_csv(accident_path, encoding="utf-8-sig")
    df = factor.merge(accident, on="연도", how="left")

    df["교통범죄 발생건수(만 건)"] = df["교통범죄 발생건수"] / 10_000
    df["검거율(%)"] = df["검거율"] * 100
    df["자동차 등록대수(백만 대)"] = df["자동차 등록대수"] / 1_000_000
    df["운전면허 소지자수(백만 명)"] = df["운전면허 소지자수"] / 1_000_000
    df["CCTV 카메라대수(만 대)"] = df["CCTV 카메라대수"] / 10_000
    df["구간단속 카메라수(천 개)"] = df["구간단속 카메라수"] / 1_000
    df["CCTV 누적 카메라대수(만 대)"] = df["CCTV 카메라대수"].cumsum() / 10_000
    df["구간단속 누적 카메라수(천 개)"] = df["구간단속 카메라수"].cumsum() / 1_000

    df["전체 교통사고(만 건)"] = df["전체_사고건수"] / 10_000
    df["음주운전 사고(천 건)"] = df["음주운전_사고건수"] / 1_000
    df["뺑소니 사고(천 건)"] = df["뺑소니_사고건수"] / 1_000
    license_base = df["운전면허 소지자수"].replace(0, np.nan)
    df["교통범죄 발생률(면허 10만 명당)"] = df["교통범죄 발생건수"] / license_base * 100_000
    df["전체 교통사고율(면허 10만 명당)"] = df["전체_사고건수"] / license_base * 100_000
    df["음주운전 사고율(면허 10만 명당)"] = df["음주운전_사고건수"] / license_base * 100_000
    df["뺑소니 사고율(면허 10만 명당)"] = df["뺑소니_사고건수"] / license_base * 100_000
    return df


@st.cache_data
def load_driver_age_accident_data(data_path: str, modified_ns: int) -> pd.DataFrame:
    # 파일 수정 시간을 캐시 키에 포함해 데이터 교체 후 이전 파일이 재사용되지 않도록 함.
    _ = modified_ns
    age_data = pd.read_csv(data_path, encoding="utf-8-sig")
    age_data["year"] = pd.to_numeric(age_data["year"], errors="coerce").astype("Int64")
    age_data["accidents"] = pd.to_numeric(age_data["accidents"], errors="coerce")
    age_data["age_group"] = (
        age_data["age_group"]
        .astype(str)
        .str.strip()
        .replace({"65세이상": "65세 이상", "19세이하": "19세 이하"})
    )
    return age_data.dropna(subset=["year", "age_group", "accidents"]).copy()


def axis_range(series: pd.Series, pad_ratio: float = 0.12) -> list[float] | None:
    values = series.dropna()
    if values.empty:
        return None
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        return [min_value * 0.9, max_value * 1.1 if max_value else 1]
    pad = (max_value - min_value) * pad_ratio
    return [min_value - pad, max_value + pad]


def dual_axis_chart(
    data: pd.DataFrame,
    title: str,
    left_col: str,
    left_title: str,
    right_col: str,
    right_title: str,
    left_range: list[float] | None = None,
    right_range: list[float] | None = None,
) -> go.Figure:
    chart_data = data[["연도", left_col, right_col]].dropna()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=chart_data["연도"],
            y=chart_data[left_col],
            name=left_title,
            mode="lines+markers",
            line=dict(color="#2f6fd0", width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=chart_data["연도"],
            y=chart_data[right_col],
            name=right_title,
            mode="lines+markers",
            line=dict(color="#e84a5f", width=3),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=22)),
        height=430,
        margin=dict(l=72, r=88, t=78, b=55),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="연도", dtick=1, showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(
        title_text=left_title,
        range=left_range or axis_range(chart_data[left_col]),
        secondary_y=False,
        showgrid=True,
        gridcolor="#e5e7eb",
    )
    fig.update_yaxes(title_text=right_title, range=right_range or axis_range(chart_data[right_col]), secondary_y=True)
    return fig


def single_axis_chart(data: pd.DataFrame, title: str, y_col: str, y_title: str, y_range=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["연도"],
            y=data[y_col],
            name=y_title,
            mode="lines+markers",
            line=dict(color="#2f6fd0", width=3),
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=23)),
        height=390,
        margin=dict(l=72, r=45, t=75, b=55),
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="연도", dtick=1, showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text=y_title, range=y_range, showgrid=True, gridcolor="#e5e7eb")
    return fig


def indexed_chart(data: pd.DataFrame) -> go.Figure:
    base = data[data["연도"] >= 2015].copy()
    columns = {
        "교통범죄 발생건수": "교통범죄",
        "자동차 등록대수": "자동차",
        "운전면허 소지자수": "운전면허",
        "CCTV 카메라대수": "CCTV 신규 설치",
        "구간단속 카메라수": "구간단속 신규 설치",
    }
    fig = go.Figure()
    for col, label in columns.items():
        usable = base[["연도", col]].dropna()
        if usable.empty or 2015 not in usable["연도"].values:
            continue
        base_value = usable.loc[usable["연도"] == 2015, col].iloc[0]
        fig.add_trace(go.Scatter(x=usable["연도"], y=usable[col] / base_value * 100, name=label, mode="lines+markers"))
    fig.update_layout(
        title=dict(text="2015년 기준 변화 비교", x=0.5, xanchor="center", font=dict(size=23)),
        height=470,
        margin=dict(l=70, r=35, t=80, b=55),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="연도", dtick=1, showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="2015년 기준", showgrid=True, gridcolor="#e5e7eb")
    return fig


def traffic_crime_timeline(data: pd.DataFrame) -> go.Figure:
    chart_data = data[["연도", "교통범죄 발생건수"]].dropna()
    fig = go.Figure(
        go.Scatter(
            x=chart_data["연도"],
            y=chart_data["교통범죄 발생건수"],
            mode="lines+markers",
            name="교통범죄 발생건수",
            line=dict(color="#e84a5f", width=3),
            marker=dict(size=8),
        )
    )
    if 2016 in chart_data["연도"].values:
        fig.add_vline(x=2016, line_width=1.5, line_dash="dot", line_color="#f59e0b")
        fig.add_annotation(
            x=2016,
            y=1.04,
            yref="paper",
            text="난폭운전 처벌·단속 강화",
            showarrow=False,
            textangle=-25,
            font=dict(size=11, color="#92400e"),
        )
    fig.update_layout(
        title=dict(text="연도별 교통범죄 발생건수", x=0.5, xanchor="center"),
        height=470,
        margin=dict(l=75, r=40, t=95, b=60),
        plot_bgcolor="white",
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
    )
    fig.update_xaxes(title_text="연도", dtick=1, showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="교통범죄 발생건수", tickformat=",", showgrid=True, gridcolor="#e5e7eb")
    return fig


def accident_count_timeline(data: pd.DataFrame) -> go.Figure:
    chart_data = data[["연도", "전체_사고건수"]].dropna()
    fig = go.Figure(
        go.Scatter(
            x=chart_data["연도"],
            y=chart_data["전체_사고건수"],
            mode="lines+markers",
            name="전체 교통사고 건수",
            line=dict(color="#2f6fd0", width=3),
            marker=dict(size=8),
        )
    )
    events = {
        2011: "면허시험 간소화",
        2016: "난폭운전 처벌·단속 강화",
        2017: "면허시험 강화(2016.12 시행)",
        2018: "능동형 보조장치 확대",
        2019: "보조장치 표준화 확대",
    }
    for year, label in events.items():
        if year not in chart_data["연도"].values:
            continue
        fig.add_vline(x=year, line_width=1.5, line_dash="dot", line_color="#f59e0b")
        fig.add_annotation(
            x=year,
            y=1.04,
            yref="paper",
            text=label,
            showarrow=False,
            textangle=-25,
            font=dict(size=11, color="#92400e"),
        )
    fig.update_layout(
        title=dict(text="연도별 전체 교통사고 건수", x=0.5, xanchor="center"),
        height=500,
        margin=dict(l=75, r=40, t=105, b=60),
        plot_bgcolor="white",
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
    )
    fig.update_xaxes(title_text="연도", dtick=1, showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="전체 교통사고 건수", tickformat=",", showgrid=True, gridcolor="#e5e7eb")
    return fig


def age_group_accident_timeline(age_data: pd.DataFrame) -> go.Figure:
    age_order = ["19세 이하", "20-29세", "30-39세", "40-49세", "50-59세", "60-64세", "65세 이상"]
    colors = {
        "19세 이하": "#cbd5e1",
        "20-29세": "#94a3b8",
        "30-39세": "#64748b",
        "40-49세": "#2f6fd0",
        "50-59세": "#0f766e",
        "60-64세": "#f59e0b",
        "65세 이상": "#e84a5f",
    }
    fig = go.Figure()
    for age_group in age_order:
        group = age_data[age_data["age_group"].eq(age_group)].sort_values("year")
        if group.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=group["year"],
                y=group["accidents"],
                name=age_group,
                mode="lines+markers",
                line=dict(color=colors[age_group], width=4 if age_group == "65세 이상" else 2.5),
                marker=dict(size=8 if age_group == "65세 이상" else 6),
            )
        )
    fig.update_layout(
        title=dict(text="가해운전자 연령대별 교통사고", x=0.5, xanchor="center"),
        height=520,
        margin=dict(l=75, r=40, t=85, b=60),
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.11, x=0),
        font=dict(family="Malgun Gothic, Arial", size=13),
    )
    fig.update_xaxes(title_text="연도", dtick=1, showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="사고건수", tickformat=",", showgrid=True, gridcolor="#e5e7eb")
    return fig


def senior_accident_share_chart(age_data: pd.DataFrame) -> go.Figure:
    totals = age_data.groupby("year", as_index=False)["accidents"].sum().rename(columns={"accidents": "total"})
    seniors = (
        age_data[age_data["age_group"].eq("65세 이상")][["year", "accidents"]]
        .rename(columns={"accidents": "senior"})
    )
    chart_data = totals.merge(seniors, on="year", how="left")
    chart_data["share"] = chart_data["senior"] / chart_data["total"] * 100
    fig = go.Figure(
        go.Scatter(
            x=chart_data["year"],
            y=chart_data["share"],
            mode="lines+markers",
            name="65세 이상 사고 비중",
            line=dict(color="#e84a5f", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title=dict(text="전체 교통사고 중 65세 이상 가해운전자 비중", x=0.5, xanchor="center"),
        height=390,
        margin=dict(l=75, r=40, t=75, b=60),
        plot_bgcolor="white",
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
    )
    fig.update_xaxes(title_text="연도", dtick=1, showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="65세 이상 비중(%)", ticksuffix="%", showgrid=True, gridcolor="#e5e7eb")
    return fig


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        step = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + step) * (a + step))
        d = 1.0 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        result *= d * c

        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + step) * (qap + step))
        d = 1.0 + coefficient * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + coefficient / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    log_term = lgamma(a + b) - lgamma(a) - lgamma(b) + a * np.log(x) + b * np.log(1.0 - x)
    front = float(np.exp(log_term))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_sided_p(t_value: float, degrees_of_freedom: float) -> float:
    if not np.isfinite(t_value) or degrees_of_freedom <= 0:
        return np.nan
    x = degrees_of_freedom / (degrees_of_freedom + t_value**2)
    return min(max(_regularized_beta(x, degrees_of_freedom / 2.0, 0.5), 0.0), 1.0)


def correlation_with_p(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    r_value = float(x.corr(y))
    sample_size = len(x)
    if sample_size < 3 or not np.isfinite(r_value):
        return r_value, np.nan
    if abs(r_value) >= 1:
        return r_value, 0.0
    t_value = r_value * sqrt((sample_size - 2) / (1.0 - r_value**2))
    return r_value, student_t_two_sided_p(t_value, sample_size - 2)


def correlation_result(frame: pd.DataFrame, x_col: str, y_col: str) -> dict:
    sample = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 3 or sample[x_col].nunique() < 2 or sample[y_col].nunique() < 2:
        return {"n": len(sample), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    pearson_r, pearson_p = correlation_with_p(sample[x_col], sample[y_col])
    spearman_r, spearman_p = correlation_with_p(sample[x_col].rank(method="average"), sample[y_col].rank(method="average"))
    return {
        "n": len(sample),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def detrended_correlation_result(frame: pd.DataFrame, x_col: str, y_col: str) -> dict:
    sample = frame[["연도", x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) < 4:
        return {"n": len(sample), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    years = sample["연도"].astype(float).to_numpy()
    x_values = sample[x_col].astype(float).to_numpy()
    y_values = sample[y_col].astype(float).to_numpy()
    x_residual = x_values - np.polyval(np.polyfit(years, x_values, 1), years)
    y_residual = y_values - np.polyval(np.polyfit(years, y_values, 1), years)
    residuals = pd.DataFrame({"x_residual": x_residual, "y_residual": y_residual})
    return correlation_result(residuals, "x_residual", "y_residual")


def correlation_strength(value: float) -> str:
    if pd.isna(value):
        return "계산 불가"
    magnitude = abs(value)
    if magnitude >= 0.7:
        strength = "강한"
    elif magnitude >= 0.4:
        strength = "중간 정도의"
    elif magnitude >= 0.2:
        strength = "약한"
    else:
        strength = "매우 약한"
    direction = "양(+)" if value >= 0 else "음(-)"
    return f"{strength} {direction} 관계"


def scatter_with_fit(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
) -> go.Figure:
    sample = frame[["연도", x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().sort_values(x_col)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample[x_col],
            y=sample[y_col],
            text=sample["연도"],
            mode="markers+text",
            textposition="top center",
            name="연도별 관측값",
            marker=dict(size=10, color="#2f6fd0"),
        )
    )
    if len(sample) >= 2 and sample[x_col].nunique() >= 2:
        slope, intercept = np.polyfit(sample[x_col], sample[y_col], 1)
        fitted = slope * sample[x_col] + intercept
        fig.add_trace(go.Scatter(x=sample[x_col], y=fitted, mode="lines", name="선형 추세선", line=dict(color="#e84a5f", width=3)))
    fig.update_layout(
        title=dict(text=f"{y_title} vs {x_title}", x=0.5, xanchor="center"),
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=430,
        margin=dict(l=70, r=35, t=70, b=60),
        plot_bgcolor="white",
        hovermode="closest",
        font=dict(family="Malgun Gothic, Arial", size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


if not FACTOR_DATA_PATH.exists():
    st.error(f"기존 비교 데이터 파일을 찾을 수 없습니다: {FACTOR_DATA_PATH}")
    st.stop()
if not ACCIDENT_DATA_PATH.exists():
    st.error(f"세부 교통사고 데이터 파일을 찾을 수 없습니다: {ACCIDENT_DATA_PATH}")
    st.stop()
if not AGE_ACCIDENT_DATA_PATH.exists():
    st.error(f"가해운전자 연령대별 사고 데이터 파일을 찾을 수 없습니다: {AGE_ACCIDENT_DATA_PATH}")
    st.stop()

df = load_data(
    str(FACTOR_DATA_PATH),
    FACTOR_DATA_PATH.stat().st_mtime_ns,
    str(ACCIDENT_DATA_PATH),
    ACCIDENT_DATA_PATH.stat().st_mtime_ns,
)
age_accident_df = load_driver_age_accident_data(
    str(AGE_ACCIDENT_DATA_PATH), AGE_ACCIDENT_DATA_PATH.stat().st_mtime_ns
)

with st.sidebar:
    st.header("분석 설정")
    year_min = int(df["연도"].min())
    year_max = int(df["연도"].max())
    year_range = st.slider("연도 범위", year_min, year_max, (year_min, year_max))
    camera_mode = st.radio("카메라 기준", ["연도별 신규 설치 수", "누적 설치 수"])

    st.divider()
    st.header("세부 비교 그래프")
    count_options = {
        "교통범죄 발생건수": ("교통범죄 발생건수(만 건)", "교통범죄(만 건)"),
        "전체 교통사고": ("전체 교통사고(만 건)", "전체 교통사고(만 건)"),
        "음주운전 사고": ("음주운전 사고(천 건)", "음주운전 사고(천 건)"),
        "뺑소니 사고": ("뺑소니 사고(천 건)", "뺑소니 사고(천 건)"),
    }
    if camera_mode == "누적 설치 수":
        sidebar_cctv_name = "CCTV 누적 설치 대수"
        sidebar_cctv_option = ("CCTV 누적 카메라대수(만 대)", "CCTV 누적 설치 대수(만 대)", None)
        sidebar_section_name = "구간단속 누적 설치 수"
        sidebar_section_option = ("구간단속 누적 카메라수(천 개)", "구간단속 누적 설치 수(천 개)", None)
    else:
        sidebar_cctv_name = "CCTV 신규 설치 대수"
        sidebar_cctv_option = ("CCTV 카메라대수(만 대)", "CCTV 신규 설치 대수(만 대)", [0, 5])
        sidebar_section_name = "구간단속 신규 설치 수"
        sidebar_section_option = ("구간단속 카메라수(천 개)", "구간단속 신규 설치 수(천 개)", [0, 8])

    factor_options = {
        "검거율": ("검거율(%)", "검거율(%)", [60, 100]),
        "자동차 등록대수": ("자동차 등록대수(백만 대)", "자동차(백만 대)", [15, 27]),
        sidebar_cctv_name: sidebar_cctv_option,
        sidebar_section_name: sidebar_section_option,
    }
    selected_count_graphs = st.multiselect(
        "비교할 범죄·사고 지표",
        list(count_options.keys()),
        default=["교통범죄 발생건수", "전체 교통사고"],
    )
    selected_factor_graphs = st.multiselect(
        "비교할 영향 요인",
        list(factor_options.keys()),
        default=["검거율"],
    )

data = df[(df["연도"] >= year_range[0]) & (df["연도"] <= year_range[1])].copy()

if camera_mode == "누적 설치 수":
    cctv_col = "CCTV 누적 카메라대수(만 대)"
    cctv_title = "CCTV 누적 설치 대수(만 대)"
    cctv_range = None
    section_col = "구간단속 누적 카메라수(천 개)"
    section_title_label = "구간단속 누적 설치 수(천 개)"
    section_range = None
else:
    cctv_col = "CCTV 카메라대수(만 대)"
    cctv_title = "CCTV 신규 설치 대수(만 대)"
    cctv_range = [0, 5]
    section_col = "구간단속 카메라수(천 개)"
    section_title_label = "구간단속 신규 설치 수(천 개)"
    section_range = [0, 8]

first = df[df["연도"] == 2007].iloc[0]
latest = df[df["연도"] == 2023].iloc[0]
drop_rate = (latest["교통범죄 발생건수"] / first["교통범죄 발생건수"] - 1) * 100
drop_2016_2018 = (
    df.loc[df["연도"].eq(2018), "교통범죄 발생건수"].iloc[0]
    / df.loc[df["연도"].eq(2016), "교통범죄 발생건수"].iloc[0]
    - 1
) * 100
corr_columns = {
    "검거율": "검거율",
    "자동차 등록대수": "자동차 등록대수",
    "운전면허 소지자수": "운전면허 소지자수",
    "CCTV 신규 설치 대수": "CCTV 카메라대수",
    "구간단속 신규 설치 수": "구간단속 카메라수",
    "전체 교통사고": "전체_사고건수",
    "음주운전 사고": "음주운전_사고건수",
    "뺑소니 사고": "뺑소니_사고건수",
}
corr_rows = []
for label, column in corr_columns.items():
    sub = data[["교통범죄 발생건수", column]].dropna()
    corr_rows.append({"비교 변수": label, "상관계수": sub["교통범죄 발생건수"].corr(sub[column]), "사용 연도 수": len(sub)})
corr_df = pd.DataFrame(corr_rows).sort_values("상관계수", key=lambda s: s.abs(), ascending=False)
top_corr = corr_df.iloc[0]

st.markdown(
    """
    <div class="hero">
        <h1>교통범죄 감소 요인 분석 대시보드</h1>
        <p>2007–2023년 공공데이터를 바탕으로 교통범죄 발생건수와 단속, 교통환경, 사고 지표의 관계를 비교합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

card1, card2, card3 = st.columns(3)
with card1:
    metric_card("2023년 교통범죄", f"{int(latest['교통범죄 발생건수']):,}건", "분석 기간 마지막 연도")
with card2:
    metric_card("2007년 대비 변화", f"{drop_rate:.1f}%", "교통범죄 발생건수 감소폭")
with card3:
    metric_card("2016–2018 변화", f"{drop_2016_2018:.1f}%", "감소폭이 두드러진 구간")

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["개요", "추세", "1대1 비교", "세부 비교", "추가 원인 분석", "통계 분석", "결론", "데이터"]
)

with tab0:
    section_title("분석 주제")
    st.markdown(
        """
        <div class="insight">
        <strong>주제</strong><br>
        공공데이터를 활용해 교통범죄 발생건수 감소에 영향을 미친 요인을 분석.
        </div>
        <div class="insight">
        <strong>가설</strong><br>
        검거율 상승, 무인단속 장비 확대, 교통환경 변화는 교통범죄 발생건수 감소와 관련이 있을 것이다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("분석 방법")
    st.markdown(
        """
        <div class="method-box">
        <code>검거율 = 검거건수 / 교통범죄 발생건수 × 100</code><br><br>
        1. 연도별 추세 비교<br>
        2. 이중축 그래프를 통한 1대1 비교<br>
        3. 세부 사고 지표와 단속·환경 요인의 조합 비교<br>
        4. 상관계수로 함께 움직이는 정도 확인<br>
        5. 2016–2018년 정책 변화 시점과 감소 흐름 해석
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("핵심 분석 결과")
    st.markdown(
        """
        <div class="insight">
        <strong>1. 검거율</strong><br>교통범죄 발생건수와 유의미한 관계가 나타남.
        </div>
        <div class="insight">
        <strong>2. 무인단속 장비</strong><br>CCTV·구간단속 확대는 교통범죄 감소에 영향을 준 환경 요인으로 작용.
        </div>
        <div class="insight">
        <strong>3. 해석 범위</strong><br>시간에 따라 계속 증가하는 변수는 공통 추세의 영향을 받으므로 단순 상관관계와 인과관계를 구분해야 함.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab1:
    st.plotly_chart(single_axis_chart(data, "교통범죄 발생건수 추세", "교통범죄 발생건수(만 건)", "발생건수(만 건)", [20, 90]), width="stretch")
    st.plotly_chart(single_axis_chart(data, "검거율 추세", "검거율(%)", "검거율(%)", [60, 100]), width="stretch")

with tab2:
    st.plotly_chart(indexed_chart(df), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            dual_axis_chart(
                data,
                "교통범죄 발생건수 vs 검거율",
                "교통범죄 발생건수(만 건)",
                "교통범죄(만 건)",
                "검거율(%)",
                "검거율(%)",
                [20, 90],
                [60, 100],
            ),
            width="stretch",
        )
        vehicle_data = data.dropna(subset=["자동차 등록대수(백만 대)"])
        st.plotly_chart(
            dual_axis_chart(vehicle_data, "교통범죄 발생건수 vs 자동차 등록대수", "교통범죄 발생건수(만 건)", "교통범죄(만 건)", "자동차 등록대수(백만 대)", "자동차(백만 대)", [20, 70], [15, 27]),
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            dual_axis_chart(data, "교통범죄 발생건수 vs 운전면허 소지자수", "교통범죄 발생건수(만 건)", "교통범죄(만 건)", "운전면허 소지자수(백만 명)", "운전면허(백만 명)", [20, 90], [24, 36]),
            width="stretch",
        )
        st.plotly_chart(
            dual_axis_chart(data, f"교통범죄 발생건수 vs {cctv_title}", "교통범죄 발생건수(만 건)", "교통범죄(만 건)", cctv_col, cctv_title, [20, 90], cctv_range),
            width="stretch",
        )
    st.plotly_chart(
        dual_axis_chart(data, f"교통범죄 발생건수 vs {section_title_label}", "교통범죄 발생건수(만 건)", "교통범죄(만 건)", section_col, section_title_label, [20, 90], section_range),
        width="stretch",
    )

with tab3:
    section_title("범죄·사고 건수와 영향 요인 비교")
    st.caption("왼쪽 사이드바에서 비교할 범죄·사고 지표와 영향 요인을 각각 선택하면 조합 그래프가 생성됩니다.")
    if not selected_count_graphs or not selected_factor_graphs:
        st.info("왼쪽 목록에서 범죄·사고 지표와 영향 요인을 각각 하나 이상 선택해 주세요.")
    else:
        for count_name in selected_count_graphs:
            for factor_name in selected_factor_graphs:
                left_col, left_title = count_options[count_name]
                right_col, right_title, right_range = factor_options[factor_name]
                graph_label = f"{count_name} vs {factor_name}"
                st.plotly_chart(
                    dual_axis_chart(
                        data,
                        graph_label,
                        left_col,
                        left_title,
                        right_col,
                        right_title,
                        axis_range(data[left_col]),
                        right_range,
                    ),
                    width="stretch",
                )

with tab4:
    section_title("교통범죄·교통사고 변화와 정책·차량 기술")
    st.caption("교통범죄와 전체 교통사고의 연도별 변화에 정책 시행 및 주행보조장치 보급 시점을 함께 표시했습니다.")
    st.plotly_chart(traffic_crime_timeline(df), width="stretch")
    st.plotly_chart(accident_count_timeline(df), width="stretch")

    accidents_by_year = df.set_index("연도")["전체_사고건수"]

    def accident_change(start_year: int, end_year: int) -> float:
        return (accidents_by_year.loc[end_year] / accidents_by_year.loc[start_year] - 1) * 100

    change1, change2, change3, change4 = st.columns(4)
    change1.metric("2015 → 2016", f"{accidents_by_year.loc[2016]:,.0f}건", f"{accident_change(2015, 2016):.1f}%")
    change2.metric("2017 → 2018", f"{accidents_by_year.loc[2018]:,.0f}건", f"{accident_change(2017, 2018):.1f}%")
    change3.metric("2018 → 2019", f"{accidents_by_year.loc[2019]:,.0f}건", f"{accident_change(2018, 2019):.1f}%")
    change4.metric("2019 → 2020", f"{accidents_by_year.loc[2020]:,.0f}건", f"{accident_change(2019, 2020):.1f}%")

    st.info(
        "전체 교통사고는 2014–2015년 증가 후 2016–2017년 감소했습니다. 2018–2019년에는 다시 증가했지만 "
        "2020–2022년 연속 감소했으며, 2023년에는 소폭 증가했습니다."
    )

    section_title("가해운전자 연령대별 사고 변화")
    st.caption("자료 범위: 경찰 교통사고 통계의 전체 가해운전자 연령대별 사고건수입니다.")
    st.plotly_chart(age_group_accident_timeline(age_accident_df), width="stretch")
    st.plotly_chart(senior_accident_share_chart(age_accident_df), width="stretch")

    age_pivot = age_accident_df.pivot(index="year", columns="age_group", values="accidents")
    age_totals = age_accident_df.groupby("year")["accidents"].sum()

    def age_change(age_group: str, start_year: int, end_year: int) -> float:
        return (age_pivot.loc[end_year, age_group] / age_pivot.loc[start_year, age_group] - 1) * 100

    senior_share_2007 = age_pivot.loc[2007, "65세 이상"] / age_totals.loc[2007] * 100
    senior_share_2023 = age_pivot.loc[2023, "65세 이상"] / age_totals.loc[2023] * 100
    total_change_2015 = (age_totals.loc[2015] / age_totals.loc[2013] - 1) * 100
    total_change_2019 = (age_totals.loc[2019] / age_totals.loc[2018] - 1) * 100

    age1, age2, age3, age4 = st.columns(4)
    age1.metric("65세 이상 2013 → 2015", f"{age_pivot.loc[2015, '65세 이상']:,.0f}건", f"{age_change('65세 이상', 2013, 2015):.1f}%")
    age2.metric("65세 이상 2018 → 2019", f"{age_pivot.loc[2019, '65세 이상']:,.0f}건", f"{age_change('65세 이상', 2018, 2019):.1f}%")
    age3.metric("65세 이상 사고 비중", f"{senior_share_2023:.1f}%", f"2007년 {senior_share_2007:.1f}%")
    age4.metric("전체 사고 2018 → 2019", f"{age_totals.loc[2019]:,.0f}건", f"{total_change_2019:.1f}%")

    st.markdown(
        f"""
        <div class="insight">
        <strong>2013–2015년:</strong> 전체 사고는 {total_change_2015:.1f}% 증가. 같은 기간 65세 이상은
        {age_change('65세 이상', 2013, 2015):.1f}%, 60–64세는 {age_change('60-64세', 2013, 2015):.1f}% 증가해
        연령대 가운데 가장 큰 증가폭을 기록. 고령 운전자 사고 증가가 이 구간의 주요 구조적 요인으로 나타남.
        </div>
        <div class="insight">
        <strong>2018–2019년:</strong> 모든 연령대의 사고가 증가. 65세 이상은
        {age_change('65세 이상', 2018, 2019):.1f}%, 60–64세는 {age_change('60-64세', 2018, 2019):.1f}% 증가해
        전체 증가율 {total_change_2019:.1f}%를 상회. 고령 운전자 사고 증가가 전체 상승폭을 확대한 것으로 나타남.
        </div>
        <div class="insight">
        <strong>장기 변화:</strong> 전체 사고 중 65세 이상 가해운전자 비중은 2007년 {senior_share_2007:.1f}%에서
        2023년 {senior_share_2023:.1f}%로 상승. 60–64세 사고도 장기적으로 증가해 고령 운전자 안전대책의 중요성이 확대됨.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("시기별 주요 변화와 해석")
    st.markdown(
        """
        <div class="insight">
        <strong>2011–2015년 · 면허시험 간소화와 사고 증가</strong><br>
        <b>사고 흐름:</b> 2011–2013년 감소 후 2014–2015년 증가<br>
        <b>제도 변화:</b> 2011년 운전면허시험 간소화 시행<br>
        <b>분석:</b> 실제 도로의 돌발상황과 복합 주행환경을 경험할 기회가 축소됨. 사고 증가는 개정 직후가 아닌 2014년부터 나타나 다른 교통환경 요인도 함께 작용한 것으로 분석됨.
        </div>
        <div class="insight">
        <strong>2016년 · 난폭운전 처벌·단속 강화와 초기 주행보조장치</strong><br>
        <b>사고 흐름:</b> 232,035건에서 220,917건으로 4.8% 감소<br>
        <b>제도·기술 변화:</b> 난폭운전 처벌·단속 강화, AEB·SCC·LDWS·FCWS·BSD 선택품목 등장<br>
        <b>분석:</b> 단속 강화와 초기 주행보조장치 도입이 동시에 진행된 시기로, 제도와 차량 기술의 복합 효과가 사고 감소 요인으로 작용.
        </div>
        <div class="insight">
        <strong>2017–2019년 · 면허시험 강화와 능동형 보조장치 확대</strong><br>
        <b>제도 변화:</b> 강화된 운전면허시험이 2016년 12월 22일 시행돼 2017년부터 본격 적용<br>
        <b>기술 변화:</b> 전방 충돌방지, 차로 이탈방지, 스마트 크루즈 등 제동·조향 개입 기능 확대<br>
        <b>사고 흐름:</b> 2017–2018년 0.4%, 2018–2019년 5.7% 증가<br>
        <b>분석:</b> 능동형 기능이 선택 패키지 중심이었던 시기로, 낮은 초기 보급률 때문에 사고 감소 효과가 전체 차량에 즉시 반영되지 않음.
        </div>
        <div class="insight">
        <strong>2019–2023년 · 기본 트림까지 안전기능 표준화 확대</strong><br>
        <b>기술 변화:</b> 전방 충돌방지, 차로 유지·이탈방지, 운전자 주의 경고가 기본 트림으로 확대<br>
        <b>사고 흐름:</b> 2020–2022년 3년 연속 감소, 2023년 소폭 증가<br>
        <b>분석:</b> 안전기능의 기본화와 사고 감소 시점이 일치함. 차량 교체에 따른 보급 시차와 교통정책이 함께 작용하므로 단일 요인보다 복합 효과로 해석.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("연도별 주행보조장치 적용 변화")
    evidence_rows = [
        {
            "연도": "2014",
            "차종": "쏘나타",
            "적용 형태": "기본·일부 선택",
            "가격표에서 확인한 기능": "차체자세제어장치(VSM), 급제동 경보, 후방 주차보조, 전자식 파킹브레이크·Auto Hold(상위 트림)",
            "변화의 의미": "차량 안정성과 주차 편의 중심이며 전방 충돌 회피나 차로 제어 기능은 확인되지 않음",
        },
        {
            "연도": "2014",
            "차종": "아반떼",
            "적용 형태": "선택 패키지",
            "가격표에서 확인한 기능": "고급형 주차조향 보조시스템(Advanced PAS: 직각주차·평행주차·평행출차)",
            "변화의 의미": "주행 중 사고 회피보다 저속 주차 보조에 가까운 단계",
        },
        {
            "연도": "2015",
            "차종": "쏘나타",
            "적용 형태": "선택 패키지",
            "가격표에서 확인한 기능": "어드밴스드 스마트 크루즈, 전방 추돌 경보(FCWS), 차선이탈 경보(LDWS), 후측방 경보(BSD), 스마트 하이빔, 어드밴스드 주차조향 보조",
            "변화의 의미": "주행보조 기능이 등장했지만 패키지 선택 차량에 한정",
        },
        {
            "연도": "2015–2016",
            "차종": "아반떼",
            "적용 형태": "하이테크 패키지",
            "가격표에서 확인한 기능": "자동 긴급제동(AEB·보행자 보호 포함), 스마트 크루즈(SCC), 차선이탈 경보(LDWS), 스마트 하이빔",
            "변화의 의미": "충돌 회피와 속도 제어 기능이 소형 승용차 선택품목으로 확대",
        },
        {
            "연도": "2018",
            "차종": "쏘나타 뉴 라이즈",
            "적용 형태": "스마트센스 패키지",
            "가격표에서 확인한 기능": "자동 긴급제동, 스마트 크루즈(Stop & Go), 차선이탈 경보, 차로 이탈방지 보조, 운전자 주의 경고, 고속도로 주행보조",
            "변화의 의미": "경고 중심에서 제동·조향에 직접 개입하는 능동형 기능으로 확장",
        },
        {
            "연도": "2018",
            "차종": "아반떼",
            "적용 형태": "기본·스마트센스 패키지",
            "가격표에서 확인한 기능": "전방 충돌방지, 차로 이탈방지, 차로 이탈 경고, 운전자 주의 경고, 스마트 크루즈, 후측방 충돌 경고·후방 교차충돌 경고",
            "변화의 의미": "일부 능동 안전기술이 기본 품목으로 들어가고 패키지 구성이 세분화됨",
        },
        {
            "연도": "2019",
            "차종": "쏘나타 센슈어스",
            "적용 형태": "기본 트림",
            "가격표에서 확인한 기능": "전방 충돌방지, 차로 이탈방지, 차로 유지보조, 운전자 주의 경고, 하이빔 보조, 전방 차량 출발 알림",
            "변화의 의미": "주요 능동 안전기술이 가장 낮은 Smart 트림의 기본 품목으로 표준화",
        },
        {
            "연도": "2020",
            "차종": "아반떼",
            "적용 형태": "기본 트림",
            "가격표에서 확인한 기능": "전방 충돌방지, 차로 유지보조, 차로 이탈방지, 차로 이탈 경고, 운전자 주의 경고, 하이빔 보조, 전방 차량 출발 알림",
            "변화의 의미": "소형 승용차 기본 트림까지 능동 안전기술의 표준 적용이 확대됨",
        },
    ]
    st.dataframe(
        pd.DataFrame(evidence_rows),
        width="stretch",
        height=455,
        hide_index=True,
        column_config={
            "연도": st.column_config.TextColumn(width="small"),
            "차종": st.column_config.TextColumn(width="medium"),
            "적용 형태": st.column_config.TextColumn(width="medium"),
            "가격표에서 확인한 기능": st.column_config.TextColumn(width="large"),
            "변화의 의미": st.column_config.TextColumn(width="large"),
        },
    )

with tab5:
    section_title("상관분석")
    st.caption("분석 기준: 원자료와 전년 대비 증감률을 함께 비교하며, p-value 0.05 미만을 통계적 유의 기준으로 적용합니다.")

    analysis_outcomes = {
        "교통범죄 발생건수": "교통범죄 발생건수",
        "교통범죄 발생률(면허 10만 명당)": "교통범죄 발생률(면허 10만 명당)",
        "전체 교통사고 건수": "전체_사고건수",
        "전체 교통사고율(면허 10만 명당)": "전체 교통사고율(면허 10만 명당)",
        "음주운전 사고 건수": "음주운전_사고건수",
        "음주운전 사고율(면허 10만 명당)": "음주운전 사고율(면허 10만 명당)",
        "뺑소니 사고 건수": "뺑소니_사고건수",
        "뺑소니 사고율(면허 10만 명당)": "뺑소니 사고율(면허 10만 명당)",
    }
    analysis_factors = {
        "검거율": "검거율(%)",
        "자동차 등록대수": "자동차 등록대수",
        "운전면허 소지자수": "운전면허 소지자수",
        cctv_title: cctv_col,
        section_title_label: section_col,
    }

    select_col1, select_col2 = st.columns(2)
    with select_col1:
        selected_outcome_name = st.selectbox("결과변수", list(analysis_outcomes.keys()), index=0)
    with select_col2:
        selected_factor_name = st.selectbox("비교변수", list(analysis_factors.keys()), index=0)

    selected_outcome_col = analysis_outcomes[selected_outcome_name]
    selected_factor_col = analysis_factors[selected_factor_name]
    raw_result = correlation_result(data, selected_factor_col, selected_outcome_col)

    change_data = data[["연도", selected_factor_col, selected_outcome_col]].sort_values("연도").copy()
    change_x_col = "비교변수 증감률(%)"
    change_y_col = "결과변수 증감률(%)"
    change_data[change_x_col] = change_data[selected_factor_col].pct_change(fill_method=None) * 100
    change_data[change_y_col] = change_data[selected_outcome_col].pct_change(fill_method=None) * 100
    change_result = correlation_result(change_data, change_x_col, change_y_col)
    detrended_result = detrended_correlation_result(data, selected_factor_col, selected_outcome_col)

    st.markdown(
        """
        <div class="method-box">
        <b>Pearson r</b>: 두 변수의 직선형 관계. -1에 가까우면 반대 방향, +1에 가까우면 같은 방향, 0에 가까우면 관계가 약함.<br><br>
        <b>Spearman ρ</b>: 값의 순위를 이용한 관계. 일부 연도의 큰 변동에 덜 민감하며 두 변수가 전반적으로 같은 순서로 움직이는지 확인.<br><br>
        <b>p-value</b>: 관찰된 관계가 우연히 나타날 가능성. 0.05 미만이면 통계적으로 유의한 관계로 판정.<br><br>
        <b>분석법 선택 이유</b>: 연도별 자료가 17개로 적어 Pearson과 Spearman을 함께 사용하고, 전년 대비 증감률과 추세 제거 분석으로 시간에 따라 함께 증가·감소하는 효과를 분리.
        </div>
        """,
        unsafe_allow_html=True,
    )

    def selected_result_summary(analysis_name: str, result: dict) -> str:
        if pd.isna(result["pearson_p"]):
            return f"<strong>{analysis_name}:</strong> 분석 가능한 연도 수 부족."
        significance = "통계적으로 유의함" if result["pearson_p"] < 0.05 else "통계적으로 유의하지 않음"
        return (
            f"<strong>{analysis_name}:</strong> {selected_factor_name}과 {selected_outcome_name}은 "
            f"{correlation_strength(result['pearson_r'])}를 보임 "
            f"(Pearson r={result['pearson_r']:.3f}, p={result['pearson_p']:.4f}). "
            f"결과는 <b>{significance}</b>."
        )

    section_title("선택 변수 분석 결과")
    raw1, raw2, raw3, raw4 = st.columns(4)
    raw1.metric("선형상관계수 (Pearson r)", f"{raw_result['pearson_r']:.3f}" if not pd.isna(raw_result["pearson_r"]) else "-")
    raw2.metric("유의확률 (p-value)", f"{raw_result['pearson_p']:.4f}" if not pd.isna(raw_result["pearson_p"]) else "-")
    raw3.metric("순위상관계수 (Spearman ρ)", f"{raw_result['spearman_r']:.3f}" if not pd.isna(raw_result["spearman_r"]) else "-")
    raw4.metric("분석 연도", f"{raw_result['n']}개")
    st.markdown(f'<div class="insight">{selected_result_summary("전체 수준 비교", raw_result)}</div>', unsafe_allow_html=True)

    change1, change2, change3, change4 = st.columns(4)
    change1.metric("증감률 선형상관계수", f"{change_result['pearson_r']:.3f}" if not pd.isna(change_result["pearson_r"]) else "-")
    change2.metric("증감률 유의확률", f"{change_result['pearson_p']:.4f}" if not pd.isna(change_result["pearson_p"]) else "-")
    change3.metric("증감률 순위상관계수", f"{change_result['spearman_r']:.3f}" if not pd.isna(change_result["spearman_r"]) else "-")
    change4.metric("분석 연도", f"{change_result['n']}개")
    st.markdown(f'<div class="insight">{selected_result_summary("전년 대비 변화 비교", change_result)}</div>', unsafe_allow_html=True)

    trend1, trend2, trend3, trend4 = st.columns(4)
    trend1.metric("추세 제거 선형상관계수", f"{detrended_result['pearson_r']:.3f}" if not pd.isna(detrended_result["pearson_r"]) else "-")
    trend2.metric("추세 제거 유의확률", f"{detrended_result['pearson_p']:.4f}" if not pd.isna(detrended_result["pearson_p"]) else "-")
    trend3.metric("추세 제거 순위상관계수", f"{detrended_result['spearman_r']:.3f}" if not pd.isna(detrended_result["spearman_r"]) else "-")
    trend4.metric("분석 연도", f"{detrended_result['n']}개")
    st.markdown(f'<div class="insight">{selected_result_summary("연도 추세 제거 후 비교", detrended_result)}</div>', unsafe_allow_html=True)

    selected_significant = {
        "전체 수준": not pd.isna(raw_result["pearson_p"]) and raw_result["pearson_p"] < 0.05,
        "전년 대비 변화": not pd.isna(change_result["pearson_p"]) and change_result["pearson_p"] < 0.05,
        "연도 추세 제거": not pd.isna(detrended_result["pearson_p"]) and detrended_result["pearson_p"] < 0.05,
    }
    significant_names = [name for name, is_significant in selected_significant.items() if is_significant]
    if not significant_names:
        selected_conclusion = f"{selected_factor_name}은 세 분석 모두에서 {selected_outcome_name}과 통계적으로 유의한 관계가 확인되지 않음."
    elif significant_names == ["전체 수준"]:
        selected_conclusion = f"{selected_factor_name}은 장기 추세에서는 관계가 나타나지만 전년 대비 변화와 추세 제거 후에는 유의하지 않아 직접적인 영향은 확인되지 않음."
    else:
        selected_conclusion = f"통계적으로 유의한 분석: {', '.join(significant_names)}. 해당 조건에서 {selected_factor_name}과 {selected_outcome_name}의 관계가 확인됨."
    st.markdown(f'<div class="insight"><strong>선택 변수 결론:</strong> {selected_conclusion}</div>', unsafe_allow_html=True)

    st.plotly_chart(
        scatter_with_fit(
            data,
            selected_factor_col,
            selected_outcome_col,
            selected_factor_name,
            selected_outcome_name,
        ),
        width="stretch",
    )

    matrix_rows = []
    for factor_name, factor_col in analysis_factors.items():
        raw = correlation_result(data, factor_col, selected_outcome_col)
        change = data[[factor_col, selected_outcome_col]].copy()
        change["x_change"] = change[factor_col].pct_change(fill_method=None) * 100
        change["y_change"] = change[selected_outcome_col].pct_change(fill_method=None) * 100
        changed = correlation_result(change, "x_change", "y_change")
        detrended = detrended_correlation_result(data, factor_col, selected_outcome_col)
        matrix_rows.append(
            {
                "비교변수": factor_name,
                "원자료 Pearson r": raw["pearson_r"],
                "원자료 p-value": raw["pearson_p"],
                "증감률 Pearson r": changed["pearson_r"],
                "증감률 p-value": changed["pearson_p"],
                "추세 제거 Pearson r": detrended["pearson_r"],
                "추세 제거 p-value": detrended["pearson_p"],
                "표본 수": raw["n"],
            }
        )
    matrix_df = pd.DataFrame(matrix_rows)

    section_title("기간별 관계 분석 설정")
    available_years = sorted(data["연도"].dropna().astype(int).unique().tolist())
    breakpoint_options = available_years[2:-1] if len(available_years) >= 4 else available_years
    default_breakpoint = 2016 if 2016 in breakpoint_options else breakpoint_options[len(breakpoint_options) // 2]
    breakpoint = st.selectbox(
        "기간 구분 연도",
        breakpoint_options,
        index=breakpoint_options.index(default_breakpoint),
        help="선택한 연도부터 이후 기간으로 구분해 요인별 상관관계를 비교합니다.",
    )

    before_start, before_end = available_years[0], breakpoint - 1
    after_start, after_end = breakpoint, available_years[-1]

    st.caption(f"분석 기간: {before_start}–{before_end}년 / {after_start}–{after_end}년")

    section_title("비교변수 전체 결과")
    st.dataframe(
        matrix_df.style.format(
            {
                "원자료 Pearson r": "{:.3f}",
                "원자료 p-value": "{:.4f}",
                "증감률 Pearson r": "{:.3f}",
                "증감률 p-value": "{:.4f}",
                "추세 제거 Pearson r": "{:.3f}",
                "추세 제거 p-value": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    section_title("종합 결론")

    def compact_relationship(result: dict) -> str:
        if pd.isna(result["pearson_p"]):
            return "자료 부족"
        direction = "양(+)" if result["pearson_r"] >= 0 else "음(-)"
        significance = "유의" if result["pearson_p"] < 0.05 else "비유의"
        return f"{significance} · {direction} · r={result['pearson_r']:.3f}"

    summary_rows = []
    conclusion_groups = {
        "추세 제거 후에도 유의": [],
        "전년 대비 변화에서 유의": [],
        "특정 기간에서만 유의": [],
        "장기 추세에서만 유의": [],
        "유의한 관계 없음": [],
    }
    for factor_name, factor_col in analysis_factors.items():
        full_row = matrix_df.loc[matrix_df["비교변수"].eq(factor_name)].iloc[0]
        before_result = correlation_result(
            data[data["연도"].between(before_start, before_end)], factor_col, selected_outcome_col
        )
        after_result = correlation_result(
            data[data["연도"].between(after_start, after_end)], factor_col, selected_outcome_col
        )

        full_significant = full_row["원자료 p-value"] < 0.05
        change_significant = full_row["증감률 p-value"] < 0.05
        trend_significant = full_row["추세 제거 p-value"] < 0.05
        before_significant = not pd.isna(before_result["pearson_p"]) and before_result["pearson_p"] < 0.05
        after_significant = not pd.isna(after_result["pearson_p"]) and after_result["pearson_p"] < 0.05

        full_result = {
            "pearson_r": full_row["원자료 Pearson r"],
            "pearson_p": full_row["원자료 p-value"],
        }
        change_result_row = {
            "pearson_r": full_row["증감률 Pearson r"],
            "pearson_p": full_row["증감률 p-value"],
        }
        trend_result_row = {
            "pearson_r": full_row["추세 제거 Pearson r"],
            "pearson_p": full_row["추세 제거 p-value"],
        }

        if trend_significant:
            raw_trend_reversed = full_significant and full_row["원자료 Pearson r"] * full_row["추세 제거 Pearson r"] < 0
            judgment = "추세 제거 후 유의·방향 반전" if raw_trend_reversed else "추세 제거 후에도 유의"
            group = "추세 제거 후에도 유의"
        elif change_significant:
            judgment = "전년 대비 변화에서 유의"
            group = "전년 대비 변화에서 유의"
        elif before_significant and after_significant:
            if before_result["pearson_r"] * after_result["pearson_r"] < 0:
                judgment = "기간별로 유의하지만 방향 반전"
            else:
                judgment = "두 기간 모두 같은 방향으로 유의"
            group = "특정 기간에서만 유의"
        elif before_significant:
            judgment = f"{before_start}–{before_end}년에만 유의"
            group = "특정 기간에서만 유의"
        elif after_significant:
            judgment = f"{after_start}–{after_end}년에만 유의"
            group = "특정 기간에서만 유의"
        elif full_significant:
            judgment = "장기 추세에서만 유의"
            group = "장기 추세에서만 유의"
        else:
            judgment = "통계적으로 유의한 관계 없음"
            group = "유의한 관계 없음"

        conclusion_groups[group].append(factor_name)
        summary_rows.append(
            {
                "요인": factor_name,
                "전체 기간": compact_relationship(full_result),
                f"{before_start}–{before_end}년": compact_relationship(before_result),
                f"{after_start}–{after_end}년": compact_relationship(after_result),
                "전년 대비": compact_relationship(change_result_row),
                "추세 제거": compact_relationship(trend_result_row),
                "최종 판정": judgment,
            }
        )

    st.dataframe(
        pd.DataFrame(summary_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "요인": st.column_config.TextColumn(width="medium"),
            "전체 기간": st.column_config.TextColumn(width="medium"),
            f"{before_start}–{before_end}년": st.column_config.TextColumn(width="medium"),
            f"{after_start}–{after_end}년": st.column_config.TextColumn(width="medium"),
            "전년 대비": st.column_config.TextColumn(width="medium"),
            "추세 제거": st.column_config.TextColumn(width="medium"),
            "최종 판정": st.column_config.TextColumn(width="large"),
        },
    )

    conclusion_lines = [
        f"<b>{group}:</b> {', '.join(factors)}"
        for group, factors in conclusion_groups.items()
        if factors
    ]
    st.markdown(
        '<div class="insight"><strong>발표 결론</strong><br>' + "<br>".join(conclusion_lines) + "</div>",
        unsafe_allow_html=True,
    )

with tab6:
    section_title("교통범죄·교통사고 감소를 위한 정책 우선순위")
    st.markdown(
        """
        <div class="insight">
        <strong>핵심 방향</strong><br>
        교통범죄는 범죄 다발 구간의 단속 실효성 강화, 교통사고는 고령 운전자 중심의 예방정책으로 구분해 추진.
        시설의 단순 확대보다 위험 구간과 위험 집단을 선별하고 시행 전후 효과를 검증하는 방식이 우선됨.
        </div>
        """,
        unsafe_allow_html=True,
    )

    investment_rows = [
        {
            "우선순위": "1순위",
            "투자 요인": "집중 단속과 단속 실효성 강화",
            "데이터 근거": "검거율은 2016–2023년에 교통범죄 발생건수와 유의한 음(-)의 관계",
            "투자 방향": "난폭운전·상습 위반·사고 다발 지역에 단속 인력과 시간을 집중하고 적발 이후 재범 관리를 강화",
            "판정": "최우선 검토",
        },
        {
            "우선순위": "2순위",
            "투자 요인": "구간단속의 선별적 확대",
            "데이터 근거": "구간단속 설치 수는 2016–2023년에 교통범죄 발생건수와 유의한 음(-)의 관계",
            "투자 방향": "전국 일괄 확대보다 과속·난폭운전 빈도가 높은 구간에 우선 설치하고 설치 전후 효과를 측정",
            "판정": "조건부 확대",
        },
        {
            "우선순위": "3순위",
            "투자 요인": "고령 운전자 예방교육과 능동형 안전기술",
            "데이터 근거": "65세 이상 가해운전자 사고 비중이 2007년 3.9%에서 2023년 20.0%로 상승",
            "투자 방향": "고령 운전자 맞춤형 안전교육과 운전능력 점검을 강화하고 충돌방지·차로유지 기능의 보급을 지원",
            "판정": "예방 투자",
        },
        {
            "우선순위": "보조",
            "투자 요인": "일반 CCTV 확대",
            "데이터 근거": "전체 기간 상관관계는 유의하지만 증감률과 추세 제거 분석에서는 유의하지 않음",
            "투자 방향": "설치 대수 확대보다 사각지대와 위반 집중 지역의 위치 최적화에 활용",
            "판정": "보조 수단",
        },
    ]
    st.dataframe(
        pd.DataFrame(investment_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "우선순위": st.column_config.TextColumn(width="small"),
            "투자 요인": st.column_config.TextColumn(width="medium"),
            "데이터 근거": st.column_config.TextColumn(width="large"),
            "투자 방향": st.column_config.TextColumn(width="large"),
            "판정": st.column_config.TextColumn(width="small"),
        },
    )

    section_title("최종 결론")
    st.markdown(
        """
        <div class="insight">
        <strong>자료를 통해 우선 검토할 정책 조합</strong><br>
        교통범죄 대책은 ① 범죄 다발 구간 집중 단속과 ② 구간단속 장비의 선별적 배치를 중심으로 구성.
        교통사고 대책은 사고 비중이 빠르게 증가한 65세 이상 운전자를 대상으로 안전교육·운전능력 점검·능동형 안전기술 지원을 병행.
        일반 CCTV는 설치 대수보다 위치 최적화에 집중하고, 모든 사업은 시행 전후 교통범죄 발생건수·사고건수·재범률을 비교해 확대 여부를 결정.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab7:
    section_title("교통범죄·교통사고 원자료")
    st.dataframe(data, width="stretch", hide_index=True)
    st.download_button(
        "현재 선택한 데이터 CSV 다운로드",
        data.to_csv(index=False).encode("utf-8-sig"),
        file_name="traffic_factor_selected.csv",
        mime="text/csv",
    )
    section_title("가해운전자 연령대별 사고 원자료")
    age_display_df = age_accident_df.rename(
        columns={
            "year": "연도",
            "age_group": "연령대",
            "accidents": "발생건수",
            "deaths": "사망자수",
            "injuries": "부상자수",
        }
    )
    st.dataframe(age_display_df, width="stretch", hide_index=True)
    st.download_button(
        "가해운전자 연령대별 사고 CSV 다운로드",
        age_display_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="traffic_accidents_by_driver_age.csv",
        mime="text/csv",
    )
    st.caption("자료 출처: 경찰청 범죄통계, 공공데이터포털 CCTV 및 단속 관련 데이터, 가해운전자 연령대별 교통사고 통계. 공공데이터 이용허락범위: 저작자표시(CC BY)")
