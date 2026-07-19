from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st

from asset_lab.data.moex import MoexApiError, MoexClient
from asset_lab.localization import install_russian_interface


st.set_page_config(page_title="Asset Lab v6", page_icon="📈", layout="wide")
install_russian_interface()

INTERVAL_OPTIONS = ("1 day", "1 hour")
INSTRUMENTS: dict[str, str | None] = {
    "Индекс МосБиржи": "IMOEX",
    "Доллар/рубль": "USDRUB_TOM",
    "Сбербанк": "SBER",
    "Другой инструмент": None,
}


@st.cache_data(ttl=3600, show_spinner=False)
def search_instruments(query: str) -> pd.DataFrame:
    return MoexClient().search_securities(query, limit=20)


def instrument_label(row: pd.Series) -> str:
    secid = str(row.get("secid", "")).strip().upper()
    shortname = str(row.get("shortname", "")).strip()
    board = str(row.get("primary_boardid", "")).strip()
    name = shortname if shortname and shortname.lower() != "nan" else secid
    suffix = f" · {board}" if board and board.lower() != "nan" else ""
    return f"{name} — {secid}{suffix}"


def render_data_selector() -> tuple[str, str]:
    with st.sidebar:
        st.header("Данные")

        instrument = st.selectbox(
            "Инструмент",
            list(INSTRUMENTS),
            key="instrument_choice",
        )

        preset_secid = INSTRUMENTS[instrument]
        if preset_secid is None:
            secid = st.text_input(
                "Код инструмента (SECID)",
                value=st.session_state.get("custom_secid", ""),
                key="custom_secid",
                placeholder="Например, GAZP",
            ).strip().upper()
        else:
            secid = preset_secid

        with st.expander("Найти код по названию", expanded=False):
            query = st.text_input(
                "Название или код",
                key="instrument_search_query",
                placeholder="Например, Газпром",
            ).strip()
            if query:
                try:
                    results = search_instruments(query)
                except MoexApiError as exc:
                    st.error(str(exc))
                    results = pd.DataFrame()

                if results.empty:
                    st.info("Ничего не найдено.")
                else:
                    records = results.to_dict("records")
                    options = [str(record.get("secid", "")).strip().upper() for record in records]
                    labels = {
                        option: instrument_label(pd.Series(record))
                        for option, record in zip(options, records, strict=True)
                    }
                    found_secid = st.selectbox(
                        "Результаты",
                        options,
                        format_func=lambda value: labels.get(value, value),
                        key="instrument_search_result",
                    )
                    if st.button("Выбрать", use_container_width=True):
                        st.session_state["instrument_choice"] = "Другой инструмент"
                        st.session_state["custom_secid"] = found_secid
                        st.rerun()

        interval = st.selectbox(
            "Интервал",
            INTERVAL_OPTIONS,
            key="analysis_interval",
        )

    st.session_state["selected_secid"] = secid
    return secid, interval


def prepare_page_source(page_path: Path, interval: str) -> str:
    source = page_path.read_text(encoding="utf-8")
    source = re.sub(
        r'^st\.set_page_config\([^\n]*\)\n',
        "",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    source = source.replace('    st.header("Data")\n', "", 1)
    source = source.replace('    st.header("Hourly data")\n', "", 1)
    source = re.sub(
        r'(?m)^(?P<indent>\s*)secid = st\.text_input\("SECID", value="SBER"\)\.strip\(\)\.upper\(\)\s*$',
        r'\g<indent>secid = st.session_state["selected_secid"]',
        source,
        count=1,
    )
    source = source.replace(
        '    interval_label = st.selectbox("Interval", list(INTERVALS), index=0)\n',
        "",
        1,
    )
    if interval == "1 day" and 'st.session_state["selected_secid"]' not in source:
        raise RuntimeError("Не удалось подключить выбранный инструмент к дневному анализу.")
    if interval == "1 hour" and 'st.session_state["selected_secid"]' not in source:
        raise RuntimeError("Не удалось подключить выбранный инструмент к часовому анализу.")
    return source


def run_asset_lab() -> None:
    _, selected_interval = render_data_selector()
    page_path = Path(__file__).parent / (
        "pages/1_Daily_Lab.py"
        if selected_interval == "1 day"
        else "pages/hourly_moments.py"
    )
    source = prepare_page_source(page_path, selected_interval)
    namespace = {
        "__name__": "__main__",
        "__file__": str(page_path),
        "__package__": None,
        "interval_label": selected_interval,
    }
    exec(compile(source, str(page_path), "exec"), namespace)


navigation = st.navigation(
    [st.Page(run_asset_lab, title="Asset Lab", default=True)],
    position="hidden",
)
navigation.run()
