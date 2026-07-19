from __future__ import annotations

from copy import deepcopy
from functools import wraps
from typing import Any, Callable

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


EXACT_TEXT: dict[str, str] = {
    # Общие элементы
    "Data": "Данные",
    "Hourly data": "Данные",
    "Interval": "Интервал",
    "SECID": "Код инструмента (SECID)",
    "Board (optional)": "Торговая доска (необязательно)",
    "From": "С",
    "Till": "По",
    "Source": "Источник",
    "MOEX live": "MOEX ISS",
    "Multi-asset ZIP": "ZIP с несколькими активами",
    "Overview": "Обзор",
    "Raw data": "Исходные данные",
    "Load / refresh": "Загрузить / обновить",
    "Download prepared hourly moments as CSV": "Скачать подготовленные часовые данные (CSV)",
    "1 day": "1 день",
    "1 hour": "1 час",
    # Заголовки
    "Asset Lab v6 · Daily": "Asset Lab v6 · Дневной анализ",
    "Asset Lab v6 · Hourly Moments": "Asset Lab v6 · Часовые моменты",
    "Volatility laboratory": "Волатильность",
    "M2 / asymmetry": "M2 и асимметрия",
    "Trend and forecast": "Тренд и прогноз",
    "Shock and relaxation": "Шоки и релаксация",
    "M2 / M3 and sessions": "M2, M3 и торговые сессии",
    "Mobility and Kurbakovsky M3": "Мобильность и M3 Курбаковского",
    "Two-process shocks": "Два типа шоков",
    "Market moments": "Рыночные моменты",
    # Боковая панель
    "Daily-data cleaning": "Очистка дневных данных",
    "Exclude gap-dependent observations after long calendar gaps": "Исключать наблюдения после длинных календарных разрывов",
    "Maximum calendar gap kept": "Максимальный допустимый разрыв, дней",
    "Exclude the still-forming current daily candle": "Исключать незавершённую дневную свечу",
    "Yang–Zhang contribution centering window": "Окно центрирования Yang–Zhang",
    "Session and scaling": "Сессия и нормировка",
    "Regular session starts": "Начало основной сессии",
    "Regular session ends": "Конец основной сессии",
    "Past robust-scale window": "Окно робастной нормировки",
    "Hourly ZIP": "ZIP с часовыми свечами",
    "Displayed asset": "Отображаемый инструмент",
    # Метрики
    "Candles": "Свечей",
    "All candles": "Всего свечей",
    "Regular candles": "Свечей основной сессии",
    "Missing close": "Пропусков цены закрытия",
    "Excluded long-gap returns": "Исключено из-за разрывов",
    "Excluded live candle": "Исключена текущая свеча",
    "Years": "Лет",
    "Panel assets": "Инструментов в панели",
    "Aligned panel assets": "Согласованных инструментов",
    "Observations": "Наблюдений",
    "Regression N": "Наблюдений в регрессии",
    "HAR-only R²": "R² только HAR",
    "Full R²": "Полный R²",
    "ΔR² from trend": "ΔR² от тренда",
    "p-value b− = b+": "p-value для b− = b+",
    "Selected shocks": "Выбранных шоков",
    "Matched shocks": "Шоков с контролями",
    "Abrupt-like": "Резких",
    "Preheated-like": "С разогревом",
    "Negative shocks": "Отрицательных шоков",
    "All selected shocks": "Всего выбранных шоков",
    "Negative-price shocks": "Шоков с падением цены",
    "Positive-price shocks": "Шоков с ростом цены",
    "Common threshold": "Общий порог",
    "HAR complete events": "Полных событий HAR",
    "Matched complete events": "Полных событий с контролями",
    "Events with controls": "Событий с контролями",
    "Local-projection horizons": "Горизонтов локальной проекции",
    "Robust overnight skewness": "Робастная ночная асимметрия",
    "Spearman: recent market M3 vs future M2": "Спирмен: недавний рыночный M3 и будущий M2",
    # Дневной анализ
    "Rolling volatility window": "Окно скользящей волатильности",
    "Daily variance proxies shown": "Показываемые оценки дневной дисперсии",
    "What each estimator measures": "Что измеряет каждая оценка",
    "Comparison window": "Окно сравнения",
    "Dates with the strongest estimator disagreement": "Даты с наибольшим расхождением оценок",
    "Moment / asymmetry window": "Окно моментов и асимметрии",
    "Raw M3 diagnostic": "Диагностика исходного M3",
    "Display scale": "Масштаб отображения",
    "Signed cube-root": "Знаковый кубический корень",
    "Central 98% clipped": "Центральные 98%",
    "Raw full scale": "Полный масштаб",
    "Does trend add information beyond volatility persistence?": "Добавляет ли тренд информацию сверх инерции волатильности?",
    "Variance proxy": "Оценка дисперсии",
    "Past trend window": "Окно прошлого тренда",
    "Future variance horizon": "Горизонт будущей дисперсии",
    "Walk-forward out-of-sample comparison": "Последовательная вневыборочная проверка",
    "Volatility shocks: identification by three counterfactual methods": "Шоки волатильности: три контрфактических метода",
    "Daily event variance proxy": "Оценка дисперсии для дневных событий",
    "Initial HAR training": "Начальная выборка HAR",
    "Refit HAR every N days": "Переоценивать HAR каждые N дней",
    "Innovation MAD window": "Окно MAD для инноваций",
    "Common shock-score quantile": "Общий квантиль силы шока",
    "Event cooldown": "Минимальный интервал между событиями",
    "Forward variance window": "Окно будущей дисперсии",
    "Post-shock horizon": "Горизонт после шока",
    "Trajectory step": "Шаг траектории",
    "Matched-control and local-projection settings": "Параметры контролей и локальных проекций",
    "Controls per event": "Контролей на событие",
    "Pre-event trend window": "Окно тренда до события",
    "Maximum local date distance": "Максимальное расстояние по времени",
    "Maximum control shock score": "Максимальная сила шока у контроля",
    "Exclude days near selected shocks": "Исключать дни рядом с выбранными шоками",
    "Detected shock table": "Таблица найденных шоков",
    "Cross-method early-response summary": "Ранний отклик по всем методам",
    "Local-projection coefficient table": "Коэффициенты локальных проекций",
    "Decay models for the matched-control negative response": "Модели затухания отрицательного отклика",
    # Часовой анализ
    "Rolling moment window (regular hours)": "Окно моментов основной сессии",
    "Overnight M3": "Ночной M3",
    "Price quantile bins per year": "Ценовых квантилей на год",
    "Leave-one-year-out mobility comparison": "Проверка мобильности с исключением одного года",
    "Finite-horizon M3 implied by affine mobility": "M3 на конечном горизонте из аффинной мобильности",
    "Analysis parameters": "Параметры анализа",
    "|z| shock quantile": "Квантиль шока |z|",
    "Cooldown (hours)": "Интервал между шоками, ч",
    "Preheating window": "Окно предварительного разогрева",
    "Matched controls": "Подобранных контролей",
    "Market-wide synchrony": "Порог общерыночной синхронности",
    "Isolated synchrony": "Порог изолированного движения",
    "Process fitted": "Тип процесса",
    "Price direction": "Направление цены",
    "Event-level held-out relaxation comparison": "Сравнение релаксации на отложенных событиях",
    "Event and matching tables": "События и качество сопоставления",
    "Detected events": "Найденные события",
    "Match quality": "Качество сопоставления",
    "Future market-M2 horizon": "Горизонт будущего рыночного M2",
    "Why Landau is not constrained mobility": "Почему модель Ландау и ограниченная мобильность — разные модели",
    "Kurbakovsky constrained mobility": "Ограниченная мобильность Курбаковского",
    "Amplitude-aware Landau": "Амплитудная модель Ландау",
    # Графики и подписи
    "Hourly offset": "Смещение, часов",
    "Price-bin estimate": "Оценка по ценовому диапазону",
    "Conditional price mobility": "Условная мобильность цены",
    "Price F": "Цена F",
    "Estimated mobility of ΔF": "Оценка мобильности ΔF",
    "Matched response": "Отклик относительно контроля",
    "x₀ window": "окно x₀",
    "Hours after shock": "Часов после шока",
    "Abnormal M2": "Избыточный M2",
    "Abnormal M3": "Избыточный M3",
    "Model": "Модель",
    "Hourly within-candle returns": "Часовые доходности внутри свечи",
    "Past-only hour-of-day standardisation": "Нормировка по часу только на прошлых данных",
    "Standardised return / scale": "Нормированная доходность / масштаб",
    "Rolling standardised M2": "Скользящий нормированный M2",
    "Rolling M3 and standardised skewness": "Скользящие M3 и нормированная асимметрия",
    "Moment / skewness": "Момент / асимметрия",
    "Negative-return share of M2": "Доля отрицательных доходностей в M2",
    "Share": "Доля",
    "Skewness by candle start hour": "Асимметрия по часу начала свечи",
    "Hour": "Час",
    "Robust skewness": "Робастная асимметрия",
    "Affine prediction": "Аффинный прогноз",
    "Observed": "Наблюдаемое",
    "Does state-dependent mobility explain M3?": "Объясняет ли зависимая от цены мобильность M3?",
    "Horizon, regular-session hours": "Горизонт, часов основной сессии",
    "Skewness": "Асимметрия",
    "Matched-control M2 trajectories": "Траектории M2 относительно контролей",
    "Matched-control M3 trajectories": "Траектории M3 относительно контролей",
    "Cross-asset market moments": "Рыночные моменты по нескольким инструментам",
    "Standardised moment": "Нормированный момент",
    "Share of assets in their own extreme-return tail": "Доля инструментов с экстремальной доходностью",
    "Synchrony": "Синхронность",
    "Logarithmic close-to-close returns": "Логарифмические доходности закрытие–закрытие",
    "Rolling annualised volatility estimates": "Скользящие оценки годовой волатильности",
    "Volatility": "Волатильность",
    "All supported OHLC volatility estimates": "Все поддерживаемые оценки волатильности OHLC",
    "Annualised volatility": "Годовая волатильность",
    "Spearman correlation between daily variance proxies": "Корреляция Спирмена между оценками дневной дисперсии",
    "Daily overnight and intraday variance components": "Ночная и внутридневная компоненты дисперсии",
    "Squared-log-return units": "Квадраты логарифмических доходностей",
    "Rolling second central moment": "Скользящий второй центральный момент",
    "Rolling standardised skewness M3 / M2^(3/2)": "Скользящая нормированная асимметрия M3 / M2^(3/2)",
    "Share of squared returns contributed by negative days": "Доля квадратов доходностей в отрицательные дни",
    "Downside share": "Доля снижения",
    "Robust asymmetry diagnostics": "Робастные показатели асимметрии",
    "Asymmetry": "Асимметрия",
    "Rolling raw third central moment": "Скользящий третий центральный момент",
    # Оценки дисперсии
    "Close-to-close squared return": "Квадрат доходности закрытие–закрытие",
    "Yang–Zhang daily contribution": "Дневной вклад Yang–Zhang",
    "Yang–Zhang rolling": "Скользящая Yang–Zhang",
    # Категории и модели
    "negative": "отрицательное",
    "positive": "положительное",
    "zero": "нулевое",
    "abrupt-like": "резкий",
    "preheated-like": "с предварительным разогревом",
    "external-like": "внешний",
    "internal-like": "внутренний",
    "intermediate": "промежуточный",
    "market-wide": "общерыночный",
    "isolated": "изолированный",
    "not available": "нет данных",
    "other": "прочее",
    "constant": "постоянная",
    "proportional": "пропорциональная",
    "affine": "аффинная",
    "exponential": "экспоненциальная",
    "kurbakovsky": "ограниченная мобильность Курбаковского",
    "landau": "модель Ландау",
    "double_exponential": "двойная экспонента",
    "shifted_power": "сдвинутый степенной закон",
    "Exponential": "Экспоненциальная",
    "Kurbakovsky constrained mobility": "Ограниченная мобильность Курбаковского",
    "Two-reservoir / double exponential": "Два резервуара / двойная экспонента",
    "Shifted power law": "Сдвинутый степенной закон",
}


PHRASE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("price and volume", "цена и объём"),
    ("hourly price and volume", "часовая цена и объём"),
    ("Amplitude-aware relaxation", "Амплитудная релаксация"),
    ("abrupt-like", "резкий"),
    ("preheated-like", "с предварительным разогревом"),
    ("external-like", "внешний"),
    ("internal-like", "внутренний"),
    ("negative shocks", "отрицательные шоки"),
    ("positive shocks", "положительные шоки"),
    ("uploaded hourly panel", "загруженная часовая панель"),
    ("Current setup:", "Текущие параметры:"),
    ("cooldown", "интервал"),
    ("preheat", "разогрев"),
    ("horizon", "горизонт"),
    ("controls", "контролей"),
    ("Process split", "Разделение процессов"),
    ("Mean observed-minus-affine skewness", "Средняя разность наблюдаемой и аффинной асимметрии"),
    ("Best-to-worst held-out RMSE spread", "Разброс RMSE от лучшей до худшей модели"),
    ("Held-out winner:", "Лучшая модель на отложенных событиях:"),
    ("Amplitude-aware Landau leads held-out RMSE for", "Амплитудная модель Ландау даёт лучший RMSE для"),
    ("Relaxation models", "Модели релаксации"),
    ("Hours after shock", "Часов после шока"),
)


COLUMN_NAMES: dict[str, str] = {
    "begin": "Начало",
    "end": "Конец",
    "date": "Дата",
    "year": "Год",
    "hour": "Час",
    "open": "Открытие",
    "close": "Закрытие",
    "high": "Максимум",
    "low": "Минимум",
    "volume": "Объём",
    "value": "Оборот",
    "observations": "Наблюдений",
    "model": "Модель",
    "label": "Модель",
    "parameters": "Параметры",
    "sample_size": "Размер выборки",
    "events": "Событий",
    "winner": "Лучшая",
    "mean": "Среднее",
    "median": "Медиана",
    "std": "Стандартное отклонение",
    "rmse": "RMSE",
    "sse": "SSE",
    "r_squared": "R²",
    "direction": "Направление",
    "process_class": "Тип процесса",
    "heating_class": "Предыстория",
    "synchrony_class": "Синхронность",
    "timestamp": "Время события",
    "event_id": "ID события",
    "position": "Позиция",
    "offset": "Смещение",
    "threshold": "Порог",
    "controls": "Контролей",
    "mean_distance": "Среднее расстояние",
    "price": "Цена",
    "mobility": "Мобильность",
    "held_out_year": "Исключённый год",
    "normalised_rmse": "Нормированный RMSE",
    "horizon_hours": "Горизонт, ч",
    "predicted_affine_skewness": "Аффинный прогноз асимметрии",
    "observed_skewness": "Наблюдаемая асимметрия",
    "skewness": "Асимметрия",
    "m3": "M3",
    "m2": "M2",
    "filter": "Фильтр",
    "burden_mae_12": "MAE нагрузки за 12 ч",
}


_PATCHED = False


def translate_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    translated = EXACT_TEXT.get(value, value)
    for source, target in PHRASE_REPLACEMENTS:
        translated = translated.replace(source, target)
    return translated


def _translated_format_func(existing: Callable[[Any], Any] | None) -> Callable[[Any], Any]:
    def format_value(value: Any) -> Any:
        rendered = existing(value) if existing is not None else value
        return translate_text(rendered)

    return format_value


def _localize_frame(data: Any) -> Any:
    if isinstance(data, pd.DataFrame):
        localized = data.copy()
        localized = localized.rename(columns=lambda column: COLUMN_NAMES.get(str(column), str(column)))
        for column in localized.select_dtypes(include=["object", "string"]).columns:
            localized[column] = localized[column].map(translate_text)
        return localized
    if isinstance(data, pd.Series):
        localized = data.copy()
        localized.name = COLUMN_NAMES.get(str(localized.name), str(localized.name))
        if localized.dtype == "object" or pd.api.types.is_string_dtype(localized.dtype):
            localized = localized.map(translate_text)
        return localized
    return data


def _localize_figure(figure: Any) -> Any:
    try:
        localized = deepcopy(figure)
        if localized.layout.title.text:
            localized.layout.title.text = translate_text(localized.layout.title.text)
        if localized.layout.legend.title.text:
            localized.layout.legend.title.text = translate_text(localized.layout.legend.title.text)
        for key in localized.layout:
            if not (str(key).startswith("xaxis") or str(key).startswith("yaxis")):
                continue
            axis = localized.layout[key]
            if axis.title.text:
                axis.title.text = translate_text(axis.title.text)
            if axis.ticktext is not None:
                axis.ticktext = [translate_text(item) for item in axis.ticktext]
        for annotation in localized.layout.annotations or ():
            annotation.text = translate_text(annotation.text)
        for trace in localized.data:
            if getattr(trace, "name", None):
                trace.name = translate_text(trace.name)
            if getattr(trace, "legendgroup", None):
                trace.legendgroup = translate_text(trace.legendgroup)
        return localized
    except Exception:
        return figure


def _wrap_module_function(name: str, mode: str) -> None:
    original = getattr(st, name, None)
    if original is None or getattr(original, "_asset_lab_ru", False):
        return

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        mutable = list(args)
        if mode == "label":
            if mutable:
                mutable[0] = translate_text(mutable[0])
            for key in ("label", "help", "placeholder"):
                if key in kwargs:
                    kwargs[key] = translate_text(kwargs[key])
        elif mode == "choice":
            if mutable:
                mutable[0] = translate_text(mutable[0])
            kwargs["format_func"] = _translated_format_func(kwargs.get("format_func"))
            for key in ("label", "help", "placeholder"):
                if key in kwargs:
                    kwargs[key] = translate_text(kwargs[key])
        elif mode == "tabs":
            if mutable:
                mutable[0] = [translate_text(item) for item in mutable[0]]
        elif mode == "dataframe":
            if mutable:
                mutable[0] = _localize_frame(mutable[0])
        elif mode == "plotly":
            if mutable:
                mutable[0] = _localize_figure(mutable[0])
        return original(*mutable, **kwargs)

    wrapped._asset_lab_ru = True  # type: ignore[attr-defined]
    setattr(st, name, wrapped)


def _wrap_delta_method(name: str, mode: str) -> None:
    original = getattr(DeltaGenerator, name, None)
    if original is None or getattr(original, "_asset_lab_ru", False):
        return

    @wraps(original)
    def wrapped(self: DeltaGenerator, *args: Any, **kwargs: Any) -> Any:
        mutable = list(args)
        if mode == "label":
            if mutable:
                mutable[0] = translate_text(mutable[0])
            for key in ("label", "help", "placeholder"):
                if key in kwargs:
                    kwargs[key] = translate_text(kwargs[key])
        elif mode == "choice":
            if mutable:
                mutable[0] = translate_text(mutable[0])
            kwargs["format_func"] = _translated_format_func(kwargs.get("format_func"))
            for key in ("label", "help", "placeholder"):
                if key in kwargs:
                    kwargs[key] = translate_text(kwargs[key])
        elif mode == "tabs":
            if mutable:
                mutable[0] = [translate_text(item) for item in mutable[0]]
        elif mode == "dataframe":
            if mutable:
                mutable[0] = _localize_frame(mutable[0])
        elif mode == "plotly":
            if mutable:
                mutable[0] = _localize_figure(mutable[0])
        return original(self, *mutable, **kwargs)

    wrapped._asset_lab_ru = True  # type: ignore[attr-defined]
    setattr(DeltaGenerator, name, wrapped)


def install_russian_interface() -> None:
    global _PATCHED
    if _PATCHED:
        return

    label_methods = (
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
        "slider",
        "checkbox",
        "toggle",
        "button",
        "download_button",
        "file_uploader",
        "expander",
        "spinner",
        "status",
        "toast",
    )
    choice_methods = ("selectbox", "radio", "multiselect", "pills", "segmented_control")

    for method in label_methods:
        _wrap_module_function(method, "label")
        _wrap_delta_method(method, "label")
    for method in choice_methods:
        _wrap_module_function(method, "choice")
        _wrap_delta_method(method, "choice")
    for method, mode in (("tabs", "tabs"), ("dataframe", "dataframe"), ("plotly_chart", "plotly")):
        _wrap_module_function(method, mode)
        _wrap_delta_method(method, mode)

    _PATCHED = True
