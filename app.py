from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd
import streamlit as st

import asset_lab.localization as localization
from asset_lab.data.moex import MoexApiError, MoexClient
from asset_lab.localization_content import EXTRA_PHRASES, EXTRA_TEXT
from asset_lab.ui import bounded_number_input


st.set_page_config(page_title="Asset Lab v6", page_icon="📈", layout="wide")
localization.EXACT_TEXT.update(EXTRA_TEXT)
localization.PHRASE_REPLACEMENTS = tuple(
    dict.fromkeys((*localization.PHRASE_REPLACEMENTS, *EXTRA_PHRASES))
)
# Streamlit may recreate its exported UI methods between script runs. Reinstall the
# wrappers instead of relying on module state from the previous run.
localization._PATCHED = False
localization.install_russian_interface()

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


def _localized_format_func(
    existing: Callable[[Any], Any] | None,
) -> Callable[[Any], Any]:
    def render(value: Any) -> Any:
        displayed = existing(value) if existing is not None else value
        return localization.translate_text(displayed)

    return render


def _bounded_slider_input(
    label: str,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
    value: int | float | None = None,
    step: int | float | None = None,
    format: str | None = None,
    key: str | None = None,
    help: str | None = None,
    on_change: Callable[..., Any] | None = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    disabled: bool = False,
    label_visibility: str = "visible",
    **_: Any,
) -> int | float:
    if min_value is None or max_value is None or value is None:
        raise ValueError("Для числового параметра должны быть заданы минимум, максимум и значение.")
    if isinstance(value, (tuple, list)):
        raise ValueError("Диапазон из двух значений пока не поддерживается числовым полем.")
    return bounded_number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        key=key,
        format=format,
        help=help,
        on_change=on_change,
        args=args,
        kwargs=kwargs,
        disabled=disabled,
        label_visibility=label_visibility,
    )


def render_data_selector() -> tuple[str, str]:
    pending_secid = st.session_state.pop("pending_secid", None)
    if pending_secid:
        st.session_state["instrument_choice"] = "Другой инструмент"
        st.session_state["custom_secid"] = str(pending_secid).strip().upper()
    st.session_state.setdefault("instrument_choice", "Индекс МосБиржи")
    st.session_state.setdefault("custom_secid", "")

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
                        st.session_state["pending_secid"] = found_secid
                        st.rerun()

        interval = st.selectbox(
            "Интервал",
            INTERVAL_OPTIONS,
            key="analysis_interval",
        )

    st.session_state["selected_secid"] = secid
    return secid, interval


def _call_name(function: ast.expr) -> str:
    parts: list[str] = []
    current: ast.expr | None = function
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _translate_expression(expression: ast.expr) -> ast.expr:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return ast.copy_location(
            ast.Constant(value=localization.translate_text(expression.value)),
            expression,
        )
    if isinstance(expression, ast.JoinedStr):
        expression.values = [
            _translate_expression(value) if isinstance(value, ast.expr) else value
            for value in expression.values
        ]
        return expression
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        expression.elts = [_translate_expression(item) for item in expression.elts]
        return expression
    if isinstance(expression, ast.Dict):
        expression.keys = [
            _translate_expression(key) if isinstance(key, ast.expr) else key
            for key in expression.keys
        ]
        return expression
    return expression


class _RussianUiTransformer(ast.NodeTransformer):
    _label_methods = {
        "title",
        "header",
        "subheader",
        "caption",
        "markdown",
        "write",
        "info",
        "warning",
        "error",
        "success",
        "metric",
        "text_input",
        "text_area",
        "date_input",
        "time_input",
        "number_input",
        "checkbox",
        "toggle",
        "button",
        "download_button",
        "file_uploader",
        "expander",
        "spinner",
        "status",
        "toast",
    }
    _choice_methods = {"selectbox", "radio", "multiselect", "pills", "segmented_control"}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        name = _call_name(node.func)
        method = name.rsplit(".", 1)[-1]

        if method == "slider":
            if node.args:
                node.args[0] = _translate_expression(node.args[0])
            for keyword in node.keywords:
                if keyword.arg in {"label", "help", "placeholder"}:
                    keyword.value = _translate_expression(keyword.value)
            node.func = ast.copy_location(
                ast.Name(id="_bounded_slider_input", ctx=ast.Load()),
                node.func,
            )

        elif name.startswith("st.") and method in self._label_methods:
            if node.args:
                node.args[0] = _translate_expression(node.args[0])
            for keyword in node.keywords:
                if keyword.arg in {"label", "help", "placeholder"}:
                    keyword.value = _translate_expression(keyword.value)

        elif name.startswith("st.") and method in self._choice_methods:
            if node.args:
                node.args[0] = _translate_expression(node.args[0])
            for keyword in node.keywords:
                if keyword.arg in {"label", "help", "placeholder"}:
                    keyword.value = _translate_expression(keyword.value)
            format_keyword = next(
                (keyword for keyword in node.keywords if keyword.arg == "format_func"),
                None,
            )
            existing = format_keyword.value if format_keyword is not None else ast.Constant(None)
            composed = ast.Call(
                func=ast.Name(id="_localized_format_func", ctx=ast.Load()),
                args=[existing],
                keywords=[],
            )
            if format_keyword is None:
                node.keywords.append(ast.keyword(arg="format_func", value=composed))
            else:
                format_keyword.value = composed

        elif name == "st.tabs" and node.args:
            node.args[0] = _translate_expression(node.args[0])

        elif method == "update_layout":
            for keyword in node.keywords:
                if keyword.arg in {
                    "title",
                    "title_text",
                    "xaxis_title",
                    "yaxis_title",
                    "legend_title",
                }:
                    keyword.value = _translate_expression(keyword.value)

        elif name in {"go.Scatter", "go.Bar", "go.Candlestick"}:
            for keyword in node.keywords:
                if keyword.arg in {"name", "hovertemplate", "texttemplate"}:
                    keyword.value = _translate_expression(keyword.value)

        elif method == "line_figure":
            if len(node.args) > 1:
                node.args[1] = _translate_expression(node.args[1])
            if len(node.args) > 2:
                node.args[2] = _translate_expression(node.args[2])
            if len(node.args) > 3:
                node.args[3] = _translate_expression(node.args[3])

        elif method in {"price_figure", "heatmap_figure"} and len(node.args) > 1:
            node.args[1] = _translate_expression(node.args[1])

        return node


def prepare_page_tree(page_path: Path, interval: str) -> ast.Module:
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

    tree = ast.parse(source, filename=str(page_path))
    tree = _RussianUiTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def run_asset_lab() -> None:
    _, selected_interval = render_data_selector()
    page_path = Path(__file__).parent / (
        "pages/1_Daily_Lab.py"
        if selected_interval == "1 day"
        else "pages/hourly_moments.py"
    )
    tree = prepare_page_tree(page_path, selected_interval)
    namespace = {
        "__name__": "__main__",
        "__file__": str(page_path),
        "__package__": None,
        "interval_label": selected_interval,
        "_localized_format_func": _localized_format_func,
        "_bounded_slider_input": _bounded_slider_input,
    }
    exec(compile(tree, str(page_path), "exec"), namespace)


navigation = st.navigation(
    [st.Page(run_asset_lab, title="Asset Lab", default=True)],
    position="hidden",
)
navigation.run()
