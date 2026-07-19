from __future__ import annotations


EXTRA_TEXT: dict[str, str] = {
    # Краткие описания страниц
    "MOEX OHLC → multiple variance estimators → HAR persistence → asymmetric trend forecast → volatility innovations → three-method shock identification": (
        "Данные MOEX → оценки волатильности → HAR-прогноз → влияние тренда → выявление шоков тремя методами"
    ),
    "Hourly MOEX OHLC → past-only standardisation → M2/M3 → mobility → abrupt/preheated processes → amplitude-aware relaxation": (
        "Часовые данные MOEX → нормировка по прошлому → M2/M3 → мобильность → тип шока → амплитудная релаксация"
    ),
    "Daily page preserves the validated v5 counterfactual workflow. Use Hourly Moments for M2/M3, mobility and process classification.": (
        "Дневной режим сохраняет проверенный контрфактический анализ v5."
    ),
    "Default regular session: 10:00–18:59. Opening, evening and overnight stay separate.": (
        "Основная сессия по умолчанию: 10:00–18:59. Открытие, вечер и ночь анализируются отдельно."
    ),
    # Подсказки и проверки ввода
    "Leave empty to use the primary board. SBER commonly resolves to TQBR.": (
        "Оставьте поле пустым, чтобы использовать основную торговую доску."
    ),
    "Supports the moex_hourly_core exporter layout and ordinary CSV/CSV.GZ candle files.": (
        "Поддерживаются архивы экспортера moex_hourly_core и обычные файлы свечей CSV/CSV.GZ."
    ),
    "`From` must be earlier than `Till`.": "Дата начала должна быть раньше даты окончания.",
    "Enter a MOEX SECID.": "Укажите код инструмента MOEX (SECID).",
    "MOEX ISS returned no candles for this route and period.": "MOEX ISS не вернул свечи за выбранный период.",
    "Only an unfinished current-day candle was available.": "За период доступна только незавершённая текущая свеча.",
    "Upload the hourly ZIP produced by the exporter.": "Загрузите ZIP с часовыми свечами.",
    "No usable hourly candle file was found in the ZIP.": "В архиве не найдено подходящих файлов с часовыми свечами.",
    "At least 500 regular-session candles are needed for the hourly laboratory.": (
        "Для часового анализа нужно не менее 500 свечей основной сессии."
    ),
    # Дневная методология
    "**Default rolling state:** Yang–Zhang, because it includes opening gaps and is drift-robust over a window. **Default daily event proxy:** gap² + Rogers–Satchell, because it separates overnight and intraday variation. Parkinson and Garman–Klass remain efficiency-oriented robustness checks, not the sole definition of volatility.": (
        "**Основная скользящая оценка:** Yang–Zhang — она учитывает гэпы открытия и устойчива к дрейфу. "
        "**Основная оценка дневного события:** gap² + Rogers–Satchell — она разделяет ночную и внутридневную дисперсию. "
        "Parkinson и Garman–Klass используются как дополнительные проверки."
    ),
    "Not enough observations for trend analysis.": "Недостаточно наблюдений для анализа тренда.",
    "Not enough complete observations for HAR + asymmetric trend regression.": (
        "Недостаточно полных наблюдений для регрессии HAR с асимметричным трендом."
    ),
    "The sample is too short for a yearly expanding-window evaluation.": (
        "Выборка слишком короткая для ежегодной последовательной проверки."
    ),
    "Volatility shocks: identification by three counterfactual methods": (
        "Шоки волатильности: проверка тремя контрфактическими методами"
    ),
    "The event definition is unchanged: a daily OHLC variance proxy must exceed a strictly past HAR forecast by an unusually large robust residual. The response is now estimated independently by **(1) a non-compounding HAR conditional-median path, (2) nearest matched no-shock days, and (3) HAC local projections**. A decay law is fitted only when the methods agree on a positive early response.": (
        "Событие определяется как крупное превышение дневной дисперсии над HAR-прогнозом, построенным только по прошлым данным. "
        "Отклик оценивается независимо тремя способами: **HAR-контрфактическая траектория, подобранные дни без шока и HAC-локальные проекции**. "
        "Закон затухания оценивается только при согласии методов о положительном раннем отклике."
    ),
    "Step 1 gives the most detailed curve. Setting the step equal to the forward window yields non-overlapping displayed windows; HAC/local bootstrap still handles dependence in estimation.": (
        "Шаг 1 даёт самую подробную кривую. Шаг, равный окну будущей дисперсии, убирает перекрытие отображаемых окон."
    ),
    "Trading observations; 504 is approximately two years.": "Торговые наблюдения; 504 — примерно два года.",
    "No volatility innovations exceed the selected threshold.": "Нет инноваций волатильности выше выбранного порога.",
    "No complete response estimates are available. Lower the threshold or shorten the horizon.": (
        "Нет полных оценок отклика. Снизьте порог или сократите горизонт."
    ),
    # Часовая методология
    "Mobility relaxes exponentially and variance is its square. This is a tightly constrained double exponential: both time scales and weights are linked.": (
        "Мобильность затухает экспоненциально, а дисперсия равна её квадрату. Поэтому масштабы времени и веса двух экспонент жёстко связаны."
    ),
    "The fractional decay rate depends nonlinearly on the event amplitude x₀. It is a separate model, not a special case hidden inside constrained mobility.": (
        "Скорость затухания нелинейно зависит от начальной амплитуды x₀. Это отдельная модель, а не частный случай ограниченной мобильности."
    ),
    "The sample is too short for mobility estimation by year and price bin.": (
        "Выборка слишком короткая для оценки мобильности по годам и ценовым диапазонам."
    ),
    "The proportional geometric law M(F)=bF is the best out-of-year model.": (
        "Пропорциональный закон M(F)=bF лучше всего работает на исключённых годах."
    ),
    "The full affine law M(F)=a+bF adds out-of-year information.": (
        "Аффинный закон M(F)=a+bF улучшает прогноз на исключённых годах."
    ),
    "A price-independent mobility is the best out-of-year model for this asset.": (
        "Для этого инструмента лучше всего работает мобильность, не зависящая от цены."
    ),
    "Upload a multi-asset ZIP to activate market-wide versus isolated classification.": (
        "Загрузите ZIP с несколькими инструментами, чтобы различать общерыночные и изолированные шоки."
    ),
    "Type exact values. The thin strip shows the value's position inside the admissible range; it is a scale, not a good/bad score.": (
        "Введите точные значения. Полоса показывает положение параметра в допустимом диапазоне."
    ),
    "No complete matched shock trajectories were available under the selected settings.": (
        "При выбранных параметрах нет полных траекторий шоков с контролями."
    ),
    "The market-moment panel requires a ZIP with at least three usable hourly assets.": (
        "Для рыночных моментов нужен ZIP минимум с тремя подходящими инструментами."
    ),
    "The 25-stock research panel found market-level M3 more useful than spectral graph features for future market M2. This panel exposes the same state variables live.": (
        "В исследовании 25 акций рыночный M3 лучше прогнозировал будущий M2, чем спектральные признаки графа."
    ),
    # Модели
    "The earlier research win was specific: Landau was most useful for preheated negative events and their integrated 12-hour excess burden. For abrupt or ordinary moderate shocks, the improvement was small, so the app keeps the held-out comparison visible instead of declaring a universal law.": (
        "Преимущество модели Ландау было локальным: оно проявлялось прежде всего у отрицательных шоков с предварительным разогревом и в суммарном избыточном отклике за 12 часов. "
        "Для резких и умеренных шоков выигрыш был мал, поэтому приложение всегда показывает проверку на отложенных событиях."
    ),
    "Treat the practical 12-hour burden column as the stronger decision metric.": (
        "Главный практический показатель — ошибка суммарного избыточного отклика за 12 часов."
    ),
    "Landau is retained as a candidate, not forced as a universal winner.": (
        "Модель Ландау остаётся кандидатом, но не объявляется универсально лучшей."
    ),
    "Small spreads mean persistence is identified more clearly than a unique law.": (
        "Малый разброс означает, что устойчивость отклика определяется лучше, чем конкретный закон затухания."
    ),
    "A large residual indicates mechanisms beyond diffusion.": "Большой остаток указывает на механизмы вне диффузионной модели.",
    # Фильтры таблиц
    "all": "все данные",
    "exclude_2022": "без 2022 года",
    "absolute_gap_below_20pct": "гэп меньше 20%",
    "exclude_2022_and_20pct": "без 2022 года и гэпов выше 20%",
    "overnight-dominated": "преимущественно ночной",
    "intraday-dominated": "преимущественно внутридневной",
    "mixed": "смешанный",
    # Графики из figures.py
    "Volume": "Объём",
    "Price": "Цена",
    "Volatility-normalised past trend": "Прошлый тренд, нормированный на волатильность",
    "Future annualised volatility": "Будущая годовая волатильность",
    "Past trend and future volatility": "Прошлый тренд и будущая волатильность",
    "Mean": "Среднее",
    "Median": "Медиана",
    "Future volatility by trend quantile": "Будущая волатильность по квантилям тренда",
    "Strong decline → strong rise": "Сильное падение → сильный рост",
    "Negative trend coefficient": "Коэффициент отрицательного тренда",
    "Positive trend coefficient": "Коэффициент положительного тренда",
    "Asymmetric trend coefficients by period (95% HAC CI)": "Коэффициенты асимметричного тренда по периодам (95% HAC-интервал)",
    "Sample": "Период",
    "Coefficient in log future variance model": "Коэффициент в модели логарифма будущей дисперсии",
    "HAR innovation score": "Сила инновации HAR",
    "log-variance innovation": "Инновация логарифма дисперсии",
    "Robust score": "Робастная оценка",
    "Innovation": "Инновация",
    "Unexpected volatility innovations relative to a past-only HAR model": "Неожиданные инновации волатильности относительно HAR-прогноза по прошлым данным",
    "Negative-price volatility shocks": "Шоки волатильности с падением цены",
    "Positive-price volatility shocks": "Шоки волатильности с ростом цены",
    "Abnormal future variance relative to a recursive no-shock HAR counterfactual": "Избыточная будущая дисперсия относительно HAR-сценария без шока",
    "Trading periods after the shock": "Торговых периодов после шока",
    "Actual variance / no-shock variance − 1": "Фактическая дисперсия / дисперсия без шока − 1",
    "Observed exceedance rate": "Наблюдаемая частота превышений",
    "Rate of later volatility-shock exceedances": "Частота последующих превышений порога шока",
    "Trading periods after the main shock": "Торговых периодов после основного шока",
    "Fraction of exposed events with another exceedance": "Доля событий с повторным превышением",
    "Negative-price shocks": "Шоки с падением цены",
    "Positive-price shocks": "Шоки с ростом цены",
    "HAR conditional median": "Условная медиана HAR",
    "Matched controls": "Подобранные контроли",
    "Local projection": "Локальная проекция",
}


EXTRA_PHRASES: tuple[tuple[str, str], ...] = (
    ("Loading candles from MOEX ISS…", "Загрузка свечей из MOEX ISS…"),
    ("Loading hourly candles from MOEX ISS…", "Загрузка часовых свечей из MOEX ISS…"),
    ("Preparing the cross-asset moment panel…", "Подготовка панели рыночных моментов…"),
    ("Estimating conditional price mobility…", "Оценка условной мобильности цены…"),
    ("Detecting shocks and matching no-shock controls…", "Поиск шоков и подбор контрольных наблюдений…"),
    ("Fitting expanding past-only HAR forecasts…", "Оценка расширяющихся HAR-прогнозов по прошлым данным…"),
    ("Building non-compounding HAR median counterfactuals…", "Построение HAR-контрфактических траекторий…"),
    ("Selecting matched no-shock control days…", "Подбор контрольных дней без шока…"),
    ("Estimating HAC local projections by horizon…", "Оценка HAC-локальных проекций по горизонтам…"),
    ("Found ", "Найдено "),
    (" close-to-close observation(s) spanning more than ", " наблюдений закрытие–закрытие с разрывом более "),
    (" calendar days; they are currently excluded from gap-dependent measures.", " календарных дней; они исключены из показателей, зависящих от гэпа."),
    (" calendar days; they are currently kept.", " календарных дней; они оставлены в расчёте."),
    ("Average full-minus-baseline ΔQLIKE", "Средняя разность полная модель − базовая по ΔQLIKE"),
    ("Negative values mean the trend model improved forecasts.", "Отрицательные значения означают улучшение прогноза за счёт тренда."),
    ("Matched-control balance is good:", "Баланс подобранных контролей хороший:"),
    ("Some matched features remain imbalanced:", "Часть признаков после сопоставления остаётся несбалансированной:"),
    ("Negative-shock response identified across methods:", "Отрицательный отклик подтверждён всеми методами:"),
    ("No cross-method negative-shock identification:", "Методы не дали согласованного отрицательного отклика:"),
    ("common threshold", "общий порог"),
    ("negative price shock", "шок с падением цены"),
    ("positive price shock", "шок с ростом цены"),
    ("event-bootstrap CI", "bootstrap-интервал по событиям"),
    ("Omori fit", "модель Омори"),
    ("Group ", "Группа "),
    ("Future vol:", "Будущая волатильность:"),
    ("Mean trend:", "Средний тренд:"),
)
