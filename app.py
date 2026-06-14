# -*- coding: utf-8 -*-
from pathlib import Path

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
    df["도로 포장률(%)"] = df["도로_계_포장률_pct"]
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
        title=dict(text="2015년=100 기준 변화 비교", x=0.5, xanchor="center", font=dict(size=23)),
        height=470,
        margin=dict(l=70, r=35, t=80, b=55),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="연도", dtick=1, showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(title_text="2015년=100", showgrid=True, gridcolor="#e5e7eb")
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
    factor_options = {
        "검거율": ("검거율(%)", "검거율(%)", [60, 100]),
        "자동차 등록대수": ("자동차 등록대수(백만 대)", "자동차(백만 대)", [15, 27]),
        "CCTV 신규 설치 대수": ("CCTV 카메라대수(만 대)", "CCTV 신규 설치(만 대)", [0, 5]),
        "구간단속 신규 설치 수": ("구간단속 카메라수(천 개)", "구간단속 신규 설치(천 개)", [0, 8]),
        "도로 포장률": ("도로 포장률(%)", "도로 포장률(%)", [85, 100]),
    }
    selected_count_graphs = st.multiselect(
        "비교할 범죄·사고 지표",
        list(count_options.keys()),
        default=["교통범죄 발생건수", "전체 교통사고"],
    )
    selected_factor_graphs = st.multiselect(
        "비교할 영향 요인",
        list(factor_options.keys()),
        default=["검거율", "도로 포장률"],
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
    "도로 포장률": "도로_계_포장률_pct",
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
        <p>2007~2023년 공공데이터를 바탕으로 교통범죄 발생건수와 단속, 교통환경, 사고 지표의 관계를 비교합니다.</p>
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
    metric_card("2016~2018 변화", f"{drop_2016_2018:.1f}%", "감소폭이 두드러진 구간")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["개요", "추세", "1대1 비교", "세부 비교", "상관계수", "데이터"])

with tab0:
    section_title("연구 개요")
    st.markdown(
        """
        <div class="insight">
        <strong>주제</strong><br>
        공공데이터를 활용해 교통범죄 발생건수 감소에 영향을 미친 요인을 분석합니다.
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
        5. 2016~2018년 정책 변화 시점과 감소 흐름 해석
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("핵심 해석")
    st.markdown(
        """
        <div class="insight">
        <strong>1. 2016~2018년에 교통범죄 발생건수가 크게 감소했습니다.</strong><br>
        이 시기에는 난폭운전 처벌 강화 이후 단속 강화 분위기, 무인단속 장비 확대, 교통안전 정책 확산이 함께 나타났습니다.
        </div>
        <div class="insight">
        <strong>2. 검거율은 교통범죄 발생건수와 뚜렷한 관계를 보입니다.</strong><br>
        다만 검거율은 발생건수를 기준으로 계산되므로, 검거율 상승이 직접적인 원인이라고 단정하기보다 단속 강화의 신호로 해석하는 것이 적절합니다.
        </div>
        <div class="insight">
        <strong>3. CCTV 신규 설치, 구간단속 신규 설치, 도로 포장률은 단독 원인보다 환경 요인으로 보는 것이 자연스럽습니다.</strong><br>
        장비와 도로환경은 장기적으로 운전자 행동을 바꾸는 요인일 수 있으므로, 발생건수와 함께 누적 추세를 보는 것이 중요합니다.
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
            dual_axis_chart(data, "교통범죄 발생건수 vs 검거율", "교통범죄 발생건수(만 건)", "교통범죄(만 건)", "검거율(%)", "검거율(%)", [20, 90], [60, 100]),
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
                    dual_axis_chart(data, graph_label, left_col, left_title, right_col, right_title, axis_range(data[left_col]), right_range),
                    width="stretch",
                )

with tab4:
    section_title("교통범죄 발생건수와의 상관계수")
    st.dataframe(corr_df.style.format({"상관계수": "{:.3f}"}), width="stretch", hide_index=True)
    st.info("상관계수는 같이 움직이는 정도만 보여줍니다. 원인과 결과를 증명하려면 정책 변화, 시차 효과, 집계 방식까지 함께 해석해야 합니다.")

with tab5:
    section_title("원자료 및 출처")
    st.dataframe(data, width="stretch", hide_index=True)
    st.download_button(
        "현재 선택한 데이터 CSV 다운로드",
        data.to_csv(index=False, encoding="utf-8-sig"),
        file_name="traffic_factor_selected.csv",
        mime="text/csv",
    )
    st.caption("자료 출처: 경찰청 범죄통계, 공공데이터포털 CCTV 및 단속 관련 데이터, 교통사고 통계 자료. 공공데이터 이용허락범위: 저작자표시(CC BY)")
