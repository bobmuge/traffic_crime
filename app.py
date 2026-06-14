# -*- coding: utf-8 -*-
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "traffic_factor_analysis_data_with_license.csv"


st.set_page_config(page_title="교통범죄 요인 분석", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    df["교통범죄 발생건수(만 건)"] = df["교통범죄 발생건수"] / 10_000
    df["검거율(%)"] = df["검거율"] * 100
    df["자동차 등록대수(백만 대)"] = df["자동차 등록대수"] / 1_000_000
    df["운전면허 소지자수(백만 명)"] = df["운전면허 소지자수"] / 1_000_000
    df["CCTV 카메라대수(만 대)"] = df["CCTV 카메라대수"] / 10_000
    df["구간단속 카메라수(천 개)"] = df["구간단속 카메라수"] / 1_000
    df["CCTV 누적 카메라대수(만 대)"] = df["CCTV 카메라대수"].cumsum() / 10_000
    df["구간단속 누적 카메라수(천 개)"] = df["구간단속 카메라수"].cumsum() / 1_000

    return df


def dual_axis_chart(
    data: pd.DataFrame,
    title: str,
    right_col: str,
    right_title: str,
    right_range: list[float] | None = None,
    left_range: list[float] | None = None,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=data["연도"],
            y=data["교통범죄 발생건수(만 건)"],
            name="교통범죄 발생건수(만 건)",
            mode="lines+markers",
            line=dict(color="#2F6FD0", width=3),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data["연도"],
            y=data[right_col],
            name=right_title,
            mode="lines+markers",
            line=dict(color="#E94B3C", width=3),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=23)),
        height=430,
        margin=dict(l=70, r=85, t=80, b=55),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
    )
    fig.update_xaxes(title_text="연도", dtick=1)
    fig.update_yaxes(
        title_text="발생건수(만 건)",
        range=left_range or [20, 90],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=right_title,
        range=right_range,
        secondary_y=True,
    )
    return fig


def single_axis_chart(
    data: pd.DataFrame,
    title: str,
    y_col: str,
    y_title: str,
    y_range: list[float] | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["연도"],
            y=data[y_col],
            name=y_title,
            mode="lines+markers",
            line=dict(color="#2F6FD0", width=3),
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=23)),
        height=390,
        margin=dict(l=70, r=45, t=75, b=55),
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
    )
    fig.update_xaxes(title_text="연도", dtick=1)
    fig.update_yaxes(title_text=y_title, range=y_range)
    return fig


def indexed_chart(data: pd.DataFrame) -> go.Figure:
    base = data[data["연도"] >= 2015].copy()
    columns = {
        "교통범죄 발생건수": "교통범죄",
        "자동차 등록대수": "자동차",
        "운전면허 소지자수": "운전면허",
        "CCTV 카메라대수": "CCTV",
        "구간단속 카메라수": "구간단속",
    }

    fig = go.Figure()
    for col, label in columns.items():
        usable = base[["연도", col]].dropna()
        if usable.empty:
            continue
        base_value = usable.loc[usable["연도"] == 2015, col].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=usable["연도"],
                y=usable[col] / base_value * 100,
                name=label,
                mode="lines+markers",
            )
        )

    fig.update_layout(
        title=dict(text="2015년=100 기준 변화 비교", x=0.5, font=dict(size=23)),
        height=470,
        margin=dict(l=70, r=35, t=80, b=55),
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Arial", size=13),
    )
    fig.update_xaxes(title_text="연도", dtick=1)
    fig.update_yaxes(title_text="2015년=100")
    return fig


if not DATA_PATH.exists():
    st.error(f"데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    st.stop()

df = load_data()

st.title("교통범죄 발생건수 영향 요인 분석")
st.write("2007년부터 2023년까지 교통범죄 발생건수와 검거율, 자동차 등록대수, 운전면허 소지자수, CCTV, 구간단속 카메라를 비교합니다.")

with st.sidebar:
    st.header("설정")
    year_min = int(df["연도"].min())
    year_max = int(df["연도"].max())
    year_range = st.slider("연도 범위", year_min, year_max, (year_min, year_max))
    camera_mode = st.radio("카메라 기준", ["연도별 신규 설치 수", "누적 설치 수"])

data = df[(df["연도"] >= year_range[0]) & (df["연도"] <= year_range[1])].copy()

if camera_mode == "누적 설치 수":
    cctv_col = "CCTV 누적 카메라대수(만 대)"
    cctv_range = None
    section_col = "구간단속 누적 카메라수(천 개)"
    section_range = None
else:
    cctv_col = "CCTV 카메라대수(만 대)"
    cctv_range = [0, 5]
    section_col = "구간단속 카메라수(천 개)"
    section_range = [0, 8]

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("최근 교통범죄", f"{int(data['교통범죄 발생건수'].iloc[-1]):,}건")
metric2.metric("최근 검거율", f"{data['검거율(%)'].iloc[-1]:.1f}%")
metric3.metric("최근 운전면허", f"{int(data['운전면허 소지자수'].iloc[-1]):,}명")
metric4.metric("최근 구간단속", f"{int(data['구간단속 카메라수'].iloc[-1]):,}개")

tab1, tab2, tab3, tab4 = st.tabs(["추세", "1대1 비교", "상관계수", "데이터"])

with tab1:
    st.plotly_chart(
        single_axis_chart(
            data,
            "교통범죄 발생건수 추세",
            "교통범죄 발생건수(만 건)",
            "발생건수(만 건)",
            [20, 90],
        ),
        width="stretch",
    )
    st.plotly_chart(
        single_axis_chart(
            data,
            "검거율 추세",
            "검거율(%)",
            "검거율(%)",
            [60, 100],
        ),
        width="stretch",
    )

with tab2:
    st.plotly_chart(indexed_chart(df), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            dual_axis_chart(
                data,
                "교통범죄 발생건수 vs 검거율",
                "검거율(%)",
                "검거율(%)",
                right_range=[60, 100],
            ),
            width="stretch",
        )

        vehicle_data = data.dropna(subset=["자동차 등록대수(백만 대)"])
        st.plotly_chart(
            dual_axis_chart(
                vehicle_data,
                "교통범죄 발생건수 vs 자동차 등록대수",
                "자동차 등록대수(백만 대)",
                "자동차(백만 대)",
                right_range=[15, 27],
                left_range=[20, 70],
            ),
            width="stretch",
        )

    with col2:
        st.plotly_chart(
            dual_axis_chart(
                data,
                "교통범죄 발생건수 vs 운전면허 소지자수",
                "운전면허 소지자수(백만 명)",
                "운전면허(백만 명)",
                right_range=[24, 36],
            ),
            width="stretch",
        )

        st.plotly_chart(
            dual_axis_chart(
                data,
                "교통범죄 발생건수 vs CCTV 카메라대수",
                cctv_col,
                cctv_col,
                right_range=cctv_range,
            ),
            width="stretch",
        )

    st.plotly_chart(
        dual_axis_chart(
            data,
            "교통범죄 발생건수 vs 구간단속 카메라수",
            section_col,
            section_col,
            right_range=section_range,
        ),
        width="stretch",
    )

with tab3:
    st.subheader("교통범죄 발생건수와의 상관계수")
    compare_columns = [
        "검거율",
        "자동차 등록대수",
        "운전면허 소지자수",
        "CCTV 카메라대수",
        "구간단속 카메라수",
    ]
    corr_rows = []
    for column in compare_columns:
        sub = data[["교통범죄 발생건수", column]].dropna()
        corr_rows.append(
            {
                "비교 변수": column,
                "상관계수": sub["교통범죄 발생건수"].corr(sub[column]),
                "사용 연도 수": len(sub),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    st.dataframe(corr_df.style.format({"상관계수": "{:.3f}"}), width="stretch", hide_index=True)
    st.info("상관계수는 같이 움직이는 정도만 보여줍니다. 원인과 결과를 증명하려면 추가 분석이 필요합니다.")

with tab4:
    st.subheader("원자료")
    st.dataframe(data, width="stretch", hide_index=True)
    st.download_button(
        "현재 선택한 데이터 CSV 다운로드",
        data.to_csv(index=False, encoding="utf-8-sig"),
        file_name="traffic_factor_selected.csv",
        mime="text/csv",
    )
st.caption("자료 출처: 공공데이터포털, 경찰청 범죄 통계, KOSIS, CCTV 정보 데이터, 공공데이터 이용허락범위: 저작자표시(CC BY)")