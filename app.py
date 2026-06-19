# -*- coding: utf-8 -*-
from pathlib import Path
from itertools import combinations
from math import lgamma, sqrt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
FACTOR_DATA_PATH = BASE_DIR / "outputs" / "traffic_factor_analysis_data_with_license.csv"
ACCIDENT_DATA_PATH = BASE_DIR / "outputs" / "traffic_accident_merged_1980_2025.csv"

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
def load_data() -> pd.DataFrame:
    factor = pd.read_csv(FACTOR_DATA_PATH, encoding="utf-8-sig")
    accident = pd.read_csv(ACCIDENT_DATA_PATH, encoding="utf-8-sig")
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
    event_year: int | None = None,
    event_label: str | None = None,
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
    if event_year is not None and event_year in chart_data["연도"].values:
        fig.add_vline(x=event_year, line_width=2, line_dash="dash", line_color="#f59e0b")
        fig.add_annotation(
            x=event_year,
            y=1.04,
            yref="paper",
            text=event_label or str(event_year),
            showarrow=False,
            font=dict(color="#b45309", size=12),
            bgcolor="rgba(255,247,237,0.9)",
        )
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


def welch_t_test(before: pd.Series, after: pd.Series) -> tuple[float, float, float]:
    before_values = before.astype(float).to_numpy()
    after_values = after.astype(float).to_numpy()
    n_before, n_after = len(before_values), len(after_values)
    variance_before = float(np.var(before_values, ddof=1))
    variance_after = float(np.var(after_values, ddof=1))
    term_before = variance_before / n_before
    term_after = variance_after / n_after
    standard_error = sqrt(term_before + term_after)
    if standard_error == 0:
        return np.nan, np.nan, np.nan
    t_value = (float(np.mean(before_values)) - float(np.mean(after_values))) / standard_error
    degrees_of_freedom = (term_before + term_after) ** 2 / (
        term_before**2 / (n_before - 1) + term_after**2 / (n_after - 1)
    )
    return t_value, degrees_of_freedom, student_t_two_sided_p(t_value, degrees_of_freedom)


def exact_permutation_p(before: pd.Series, after: pd.Series) -> float:
    before_values = before.astype(float).to_numpy()
    after_values = after.astype(float).to_numpy()
    combined = np.concatenate([before_values, after_values])
    before_size = len(before_values)
    observed = abs(float(np.mean(before_values) - np.mean(after_values)))
    extreme = 0
    total = 0
    all_indexes = np.arange(len(combined))
    for selected in combinations(range(len(combined)), before_size):
        selected_indexes = np.array(selected, dtype=int)
        other_indexes = np.setdiff1d(all_indexes, selected_indexes, assume_unique=True)
        difference = abs(float(np.mean(combined[selected_indexes]) - np.mean(combined[other_indexes])))
        extreme += difference >= observed - 1e-12
        total += 1
    return extreme / total if total else np.nan


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


def p_value_text(p_value: float) -> str:
    if pd.isna(p_value):
        return "계산 불가"
    return "통계적으로 유의함" if p_value < 0.05 else "통계적으로 유의하지 않음"


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

df = load_data()

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

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["개요", "추세", "1대1 비교", "세부 비교", "추가 원인 분석", "통계 분석", "데이터"]
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
                event_year=2016,
                event_label="난폭운전 단속 시행",
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
    section_title("교통사고 감소에 영향을 준 제도와 차량 기술")
    st.caption("연도별 교통사고 건수와 면허시험 개정, 단속 강화, 주행보조장치 보급 시점을 비교했습니다.")
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

    section_title("선택 변수 분석 결과")
    raw1, raw2, raw3, raw4 = st.columns(4)
    raw1.metric("Pearson r", f"{raw_result['pearson_r']:.3f}" if not pd.isna(raw_result["pearson_r"]) else "-")
    raw2.metric("Pearson p-value", f"{raw_result['pearson_p']:.4f}" if not pd.isna(raw_result["pearson_p"]) else "-")
    raw3.metric("Spearman ρ", f"{raw_result['spearman_r']:.3f}" if not pd.isna(raw_result["spearman_r"]) else "-")
    raw4.metric("사용 연도 수", f"{raw_result['n']}개")
    st.info(
        f"원자료 Pearson 분석: **{correlation_strength(raw_result['pearson_r'])}**, "
        f"**{p_value_text(raw_result['pearson_p'])}** (p={raw_result['pearson_p']:.4f})."
        if not pd.isna(raw_result["pearson_p"])
        else "분석 가능한 연도 수가 부족합니다."
    )

    change1, change2, change3, change4 = st.columns(4)
    change1.metric("증감률 Pearson r", f"{change_result['pearson_r']:.3f}" if not pd.isna(change_result["pearson_r"]) else "-")
    change2.metric("증감률 p-value", f"{change_result['pearson_p']:.4f}" if not pd.isna(change_result["pearson_p"]) else "-")
    change3.metric("증감률 Spearman ρ", f"{change_result['spearman_r']:.3f}" if not pd.isna(change_result["spearman_r"]) else "-")
    change4.metric("증감률 연도 수", f"{change_result['n']}개")
    st.warning("해석 기준: 원자료보다 증감률 상관계수가 크게 낮으면 공통된 시간 추세가 원자료 상관관계에 포함된 것으로 판단합니다.")

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
        matrix_rows.append(
            {
                "비교변수": factor_name,
                "원자료 Pearson r": raw["pearson_r"],
                "원자료 p-value": raw["pearson_p"],
                "증감률 Pearson r": changed["pearson_r"],
                "증감률 p-value": changed["pearson_p"],
                "표본 수": raw["n"],
            }
        )
    matrix_df = pd.DataFrame(matrix_rows)

    section_title("기준 연도 전후 차이 검정")
    test_col1, test_col2 = st.columns(2)
    with test_col1:
        test_outcome_name = st.selectbox("검정할 결과변수", list(analysis_outcomes.keys()), index=0, key="test_outcome")
    with test_col2:
        available_years = sorted(data["연도"].dropna().astype(int).unique().tolist())
        breakpoint_options = available_years[2:-1] if len(available_years) >= 4 else available_years
        default_breakpoint = 2016 if 2016 in breakpoint_options else breakpoint_options[len(breakpoint_options) // 2]
        breakpoint = st.selectbox(
            "비교 기준 연도",
            breakpoint_options,
            index=breakpoint_options.index(default_breakpoint),
            help="기준 연도는 이후 기간에 포함됩니다. 앞뒤에 최소 2개 연도가 남는 범위에서 선택할 수 있습니다.",
        )

    before_start, before_end = available_years[0], breakpoint - 1
    after_start, after_end = breakpoint, available_years[-1]

    if selected_factor_name == "검거율" and breakpoint == 2016:
        st.success("2016년 난폭운전 단속 시행 시점을 기준으로 이전·이후 기간을 비교합니다.")
    else:
        st.caption("선택한 기준 연도는 이후 기간의 첫해로 포함됩니다.")

    test_col = analysis_outcomes[test_outcome_name]
    before_values = data.loc[data["연도"].between(before_start, before_end), test_col].dropna()
    after_values = data.loc[data["연도"].between(after_start, after_end), test_col].dropna()
    welch_p = np.nan
    mean_change = np.nan

    if len(before_values) >= 2 and len(after_values) >= 2:
        welch_t, welch_df, welch_p = welch_t_test(before_values, after_values)
        permutation_p = exact_permutation_p(before_values, after_values)
        before_mean = float(before_values.mean())
        after_mean = float(after_values.mean())
        mean_change = (after_mean / before_mean - 1) * 100 if before_mean != 0 else np.nan

        test1, test2, test3, test4 = st.columns(4)
        test1.metric(f"이전 평균 ({before_start}–{before_end})", f"{before_mean:,.2f}")
        test2.metric(f"이후 평균 ({after_start}–{after_end})", f"{after_mean:,.2f}", f"{mean_change:.1f}%")
        test3.metric("Welch t-test p", f"{welch_p:.4f}")
        test4.metric("순열검정 p", f"{permutation_p:.4f}")
        st.info(
            f"Welch t-test 결과, 두 기간 평균 차이는 **{p_value_text(welch_p)}**입니다. "
            f"(t={welch_t:.3f}, 자유도={welch_df:.2f}, p={welch_p:.4f})"
        )
    else:
        st.warning("선택한 결과변수에 전후 검정을 수행할 연도별 자료가 충분하지 않습니다.")

    section_title("비교변수 전체 결과")
    st.dataframe(
        matrix_df.style.format(
            {
                "원자료 Pearson r": "{:.3f}",
                "원자료 p-value": "{:.4f}",
                "증감률 Pearson r": "{:.3f}",
                "증감률 p-value": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    section_title("종합 결론")

    def relationship_text(result: dict) -> str:
        if pd.isna(result["pearson_p"]):
            return "표본 부족으로 판정 불가"
        direction = "양(+)" if result["pearson_r"] >= 0 else "음(-)"
        significance = "유의함" if result["pearson_p"] < 0.05 else "유의하지 않음"
        return f"{direction} 관계, {significance} (r={result['pearson_r']:.3f}, p={result['pearson_p']:.4f})"

    factor_cards = []
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
        before_significant = not pd.isna(before_result["pearson_p"]) and before_result["pearson_p"] < 0.05
        after_significant = not pd.isna(after_result["pearson_p"]) and after_result["pearson_p"] < 0.05

        if before_significant and after_significant:
            same_direction = before_result["pearson_r"] * after_result["pearson_r"] > 0
            if same_direction:
                period_summary = "기준 연도 이전과 이후 모두 같은 방향의 유의한 관계 확인."
            else:
                period_summary = "기준 연도 이전과 이후 모두 유의하지만 관계 방향이 반전돼 일관된 영향은 확인되지 않음."
        elif before_significant:
            period_summary = f"{before_start}–{before_end}년에만 유의하며 {after_start}–{after_end}년에는 영향이 확인되지 않음."
        elif after_significant:
            period_summary = f"{after_start}–{after_end}년에만 유의하며 {before_start}–{before_end}년에는 영향이 확인되지 않음."
        else:
            period_summary = "기준 연도 이전과 이후 모두 통계적으로 유의한 영향이 확인되지 않음."

        if change_significant:
            factor_summary = "전년 대비 변화에서도 유의해 연도별 변동과 직접적인 관련성이 확인됨."
        elif full_significant:
            factor_summary = "전체 기간의 장기 추세에서는 유의하지만 전년 대비 변화에서는 유의하지 않아 단기적 직접 영향은 확인되지 않음."
        else:
            factor_summary = "전체 기간과 전년 대비 변화 모두 유의하지 않아 이 자료에서는 독립적인 영향이 확인되지 않음."

        change_direction = "양(+)" if full_row["증감률 Pearson r"] >= 0 else "음(-)"
        change_significance_text = "유의함" if change_significant else "유의하지 않음"
        factor_cards.append(
            f"""
            <div class="insight">
            <strong>{factor_name}</strong><br>
            <b>전체 기간:</b> {('유의함' if full_significant else '유의하지 않음')} (r={full_row['원자료 Pearson r']:.3f}, p={full_row['원자료 p-value']:.4f})<br>
            <b>{before_start}–{before_end}년:</b> {relationship_text(before_result)}<br>
            <b>{after_start}–{after_end}년:</b> {relationship_text(after_result)}<br>
            <b>전년 대비 변화:</b> {change_direction} 관계, {change_significance_text} (r={full_row['증감률 Pearson r']:.3f}, p={full_row['증감률 p-value']:.4f})<br>
            <b>결론:</b> {period_summary} {factor_summary}
            </div>
            """
        )

    st.markdown("".join(factor_cards), unsafe_allow_html=True)

    if pd.isna(welch_p):
        welch_conclusion = "표본 부족으로 기준 연도 전후 평균 차이를 판정할 수 없음."
    else:
        change_direction = "감소" if mean_change < 0 else "증가"
        significance = "유의함" if welch_p < 0.05 else "유의하지 않음"
        welch_conclusion = (
            f"{test_outcome_name} 평균은 {breakpoint}년 이후 {abs(mean_change):.1f}% {change_direction}했으며 "
            f"차이는 통계적으로 {significance} (Welch t-test, p={welch_p:.4f})."
        )
    st.markdown(
        f'<div class="insight"><strong>기준 연도 전후 결론</strong><br>{welch_conclusion}</div>',
        unsafe_allow_html=True,
    )

with tab6:
    section_title("원자료 및 출처")
    st.dataframe(data, width="stretch", hide_index=True)
    st.download_button(
        "현재 선택한 데이터 CSV 다운로드",
        data.to_csv(index=False, encoding="utf-8-sig"),
        file_name="traffic_factor_selected.csv",
        mime="text/csv",
    )
    st.caption("자료 출처: 경찰청 범죄통계, 공공데이터포털 CCTV 및 단속 관련 데이터, 교통사고 통계 자료. 공공데이터 이용허락범위: 저작자표시(CC BY)")
