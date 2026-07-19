from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import asset_lab.localization as localization
from asset_lab.data.moex import MoexApiError, MoexClient
from asset_lab.localization_content import EXTRA_PHRASES, EXTRA_TEXT
from asset_lab.ui import apply_asset_lab_style, bounded_number_input


st.set_page_config(page_title="Asset Lab v6", page_icon="📈", layout="wide")
apply_asset_lab_style()
localization.EXACT_TEXT.update(EXTRA_TEXT)
localization.EXACT_TEXT.update(
    {
        "Rolling volatility window": "Окно расчёта исторической волатильности",
        "Daily variance proxies shown": "Показываемые оценки дисперсии",
        "Rolling annualised volatility estimates": "Оценки исторической волатильности",
        "Close-to-close squared return": "Квадрат логарифмической доходности",
        "Parkinson": "Паркинсон",
        "Garman–Klass": "Гарман—Класс",
        "Rogers–Satchell": "Роджерс—Сатчелл",
        "Gap² + Rogers–Satchell": "Ночной разрыв² + Роджерс—Сатчелл",
        "Yang–Zhang daily contribution": "Вклад Янга—Чжана",
        "Yang–Zhang rolling": "Янг—Чжан",
        "Volatility": "Годовая волатильность",
    }
)
localization.PHRASE_REPLACEMENTS = tuple(
    dict.fromkeys((*localization.PHRASE_REPLACEMENTS, *EXTRA_PHRASES))
)
localization._PATCHED = False
localization.install_russian_interface()

INTERVAL_OPTIONS = ("1 day", "1 hour")
INSTRUMENTS: dict[str, str | None] = {
    "Индекс МосБиржи": "IMOEX",
    "Доллар/рубль": "USD000UTSTOM",
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


def _has_meaningful_volume(candles: pd.DataFrame) -> bool:
    if "volume" not in candles.columns:
        return False
    volume = pd.to_numeric(candles["volume"], errors="coerce").fillna(0.0)
    return bool(volume.abs().gt(0.0).any())


def _price_figure_candles(candles: pd.DataFrame) -> pd.DataFrame:
    if _has_meaningful_volume(candles):
        return candles
    return candles.drop(columns=["volume"], errors="ignore")


def _price_figure_title(title: str, candles: pd.DataFrame) -> str:
    if _has_meaningful_volume(candles):
        return title
    return title.replace("hourly price and volume", "hourly price").replace(
        "price and volume", "price"
    )


def _clean_log_returns_for_display(
    time_axis: pd.Index,
    raw_returns: pd.Series,
    max_gap_days: int,
) -> pd.Series:
    timestamps = pd.DatetimeIndex(pd.to_datetime(time_axis, errors="coerce"))
    values = pd.Series(
        pd.to_numeric(raw_returns, errors="coerce").to_numpy(),
        index=timestamps,
        dtype=float,
    )
    gaps = timestamps.to_series(index=timestamps).diff().dt.total_seconds() / 86_400.0
    return values.mask(gaps > max_gap_days)


def _log_return_figure(
    time_axis: pd.Index,
    raw_returns: pd.Series,
    max_gap_days: int,
) -> go.Figure:
    returns = _clean_log_returns_for_display(time_axis, raw_returns, max_gap_days)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns,
            mode="lines",
            connectgaps=True,
            name="Логарифмическая доходность",
            hovertemplate=(
                "Дата: %{x|%Y-%m-%d}<br>"
                "Логарифмическая доходность: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0)
    figure.update_layout(
        yaxis_title="log(Pₜ/Pₜ₋₁)",
        hovermode="x unified",
        showlegend=False,
        margin={"t": 8},
    )
    return figure


def _render_log_return_chart(
    time_axis: pd.Index,
    raw_returns: pd.Series,
    max_gap_days: int,
) -> None:
    st.subheader(
        "Логарифмическая доходность",
        help=(
            "Значение за день: log(Pₜ/Pₜ₋₁), где Pₜ — цена закрытия в день t. "
            f"Переходы через перерывы более {max_gap_days} календарных дней "
            "не учитываются."
        ),
    )
    st.plotly_chart(
        _log_return_figure(time_axis, raw_returns, max_gap_days),
        width="stretch",
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
                        format_func=lambda item: labels.get(item, item),
                        key="instrument_search_result",
                    )
                    if st.button("Выбрать", width="stretch"):
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


def _modern_width_value(expression: ast.expr) -> ast.expr:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, bool):
        return ast.copy_location(
            ast.Constant(value="stretch" if expression.value else "content"),
            expression,
        )
    return ast.copy_location(
        ast.IfExp(
            test=expression,
            body=ast.Constant(value="stretch"),
            orelse=ast.Constant(value="content"),
        ),
        expression,
    )


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

        for keyword in node.keywords:
            if keyword.arg == "use_container_width":
                keyword.arg = "width"
                keyword.value = _modern_width_value(keyword.value)

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
        elif method == "price_figure":
            if len(node.args) > 1:
                node.args[1] = _translate_expression(node.args[1])
                node.args[1] = ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_price_figure_title", ctx=ast.Load()),
                        args=[node.args[1], node.args[0]],
                        keywords=[],
                    ),
                    node.args[1],
                )
            if node.args:
                node.args[0] = ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_price_figure_candles", ctx=ast.Load()),
                        args=[node.args[0]],
                        keywords=[],
                    ),
                    node.args[0],
                )
        elif method == "heatmap_figure" and len(node.args) > 1:
            node.args[1] = _translate_expression(node.args[1])

        return node


def _rewrite_daily_summary(source: str) -> str:
    pattern = re.compile(
        r'metric_columns = st\.columns\(6\)\n'
        r'.*?'
        r'    with st\.expander\("Long calendar gaps"\):\n'
        r'        st\.dataframe\(gap_report, use_container_width=True, hide_index=True\)\n',
        flags=re.DOTALL,
    )
    replacement = '''excluded_gap_returns = int(raw_returns.notna().sum() - returns.notna().sum())
summary_columns = st.columns(2)
summary_columns[0].metric("Торговых дней", f"{len(candles):,}")
summary_columns[1].metric("Исключено из-за разрывов", excluded_gap_returns)

for warning in warnings:
    st.warning(warning)

if not gap_report.empty:
    expander_title = (
        "Исключено из-за разрывов — подробнее"
        if exclude_long_gaps
        else "Длинные календарные разрывы — подробнее"
    )
    with st.expander(expander_title):
        if exclude_long_gaps:
            st.markdown(
                f"Между {len(gap_report)} парами соседних торговых дней прошло больше "
                f"{max_gap_days} календарных дней. Доходность через такие перерывы не "
                "используется в показателях, зависящих от предыдущей цены закрытия: "
                "логарифмической доходности, ночном разрыве и оценках волатильности, "
                "которые учитывают этот разрыв. Сами дневные свечи из данных не удаляются."
            )
        else:
            st.markdown(
                f"Найдено {len(gap_report)} разрывов длиннее {max_gap_days} календарных "
                "дней. Сейчас доходности через них оставлены в расчётах."
            )
        displayed_gaps = gap_report.rename(
            columns={
                "previous_timestamp": "Предыдущий торговый день",
                "timestamp": "Следующий торговый день",
                "gap_days": "Разрыв, календарных дней",
                "log_return": "Доходность через разрыв",
            }
        )
        st.dataframe(displayed_gaps, width="stretch", hide_index=True)
'''
    rewritten, count = pattern.subn(replacement, source, count=1)
    if count != 1:
        raise RuntimeError("Не удалось обновить сводку дневных данных.")
    return rewritten


def _rewrite_daily_return_chart(source: str) -> str:
    old_block = '''    st.plotly_chart(
        line_figure(
            time_axis,
            {"cleaned log return": returns, "raw log return": raw_returns},
            "Logarithmic close-to-close returns",
            "log(Pₜ/Pₜ₋₁)",
            zero_line=True,
        ),
        use_container_width=True,
    )
'''
    new_block = '''    _render_log_return_chart(
        time_axis,
        raw_returns,
        max_gap_days,
    )
'''
    rewritten, count = source.replace(old_block, new_block, 1), source.count(old_block)
    if count != 1:
        raise RuntimeError("Не удалось обновить график логарифмической доходности.")
    return rewritten


def _rewrite_volatility_overview(source: str) -> str:
    old_window = '    overview_window = st.slider("Rolling volatility window", 5, 120, 20, key="overview_window")\n'
    new_window = '''    overview_window = st.slider(
        "Окно расчёта исторической волатильности",
        5,
        120,
        20,
        key="overview_window",
        help="Для каждой даты используются последние N торговых периодов.",
    )
'''
    source, window_count = source.replace(old_window, new_window, 1), source.count(old_window)
    source = source.replace(
        '        "Daily variance proxies shown",\n',
        '        "Показываемые оценки дисперсии",\n',
        1,
    )
    source = source.replace(
        '            "Rolling annualised volatility estimates",\n',
        '            "Оценки исторической волатильности",\n',
        1,
    )
    if window_count != 1:
        raise RuntimeError("Не удалось обновить параметры исторической волатильности.")
    return source


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
    if interval == "1 day":
        source = _rewrite_daily_summary(source)
        source = _rewrite_daily_return_chart(source)
        source = _rewrite_volatility_overview(source)
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
        "_price_figure_candles": _price_figure_candles,
        "_price_figure_title": _price_figure_title,
        "_render_log_return_chart": _render_log_return_chart,
    }
    exec(compile(tree, str(page_path), "exec"), namespace)


navigation = st.navigation(
    [st.Page(run_asset_lab, title="Asset Lab", default=True)],
    position="hidden",
)
navigation.run()
