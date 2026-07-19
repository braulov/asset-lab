from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests


class MoexApiError(RuntimeError):
    """Raised when MOEX ISS returns an invalid or unsuccessful response."""


@dataclass(frozen=True)
class InstrumentRoute:
    secid: str
    engine: str
    market: str
    board: str
    board_title: str | None = None


class MoexClient:
    """Small explicit client for the public MOEX ISS JSON API."""

    base_url = "https://iss.moex.com/iss"
    secid_aliases = {
        # MOEX shows USDRUB_TOM as the contract code, while ISS uses this SECID.
        "USDRUB_TOM": "USD000UTSTOM",
    }

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "asset-lab/0.1 (+research prototype)",
                "Accept": "application/json",
            }
        )

    @classmethod
    def normalize_secid(cls, secid: str) -> str:
        normalized = secid.strip().upper()
        return cls.secid_aliases.get(normalized, normalized)

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise MoexApiError(f"Не удалось получить данные MOEX ISS: {exc}") from exc

        if not isinstance(payload, dict):
            raise MoexApiError("MOEX ISS вернул JSON неожиданного формата.")
        return payload

    @staticmethod
    def table_to_frame(payload: dict[str, Any], table: str) -> pd.DataFrame:
        block = payload.get(table)
        if not isinstance(block, dict):
            raise MoexApiError(f"В ответе MOEX ISS нет таблицы '{table}'.")

        columns = block.get("columns", [])
        data = block.get("data", [])
        if not isinstance(columns, list) or not isinstance(data, list):
            raise MoexApiError(f"Таблица '{table}' имеет неожиданный формат.")
        return pd.DataFrame(data, columns=columns)

    def search_securities(self, query: str, limit: int = 25) -> pd.DataFrame:
        query = query.strip()
        if not query:
            return pd.DataFrame()

        payload = self._get_json(
            "securities.json",
            {
                "q": query,
                "iss.meta": "off",
                "iss.only": "securities",
                "securities.columns": (
                    "secid,shortname,name,type,group,primary_boardid,"
                    "marketprice_boardid,is_traded"
                ),
            },
        )
        frame = self.table_to_frame(payload, "securities")
        if frame.empty:
            return frame

        if "is_traded" in frame.columns:
            frame = frame.sort_values("is_traded", ascending=False, na_position="last")
        return frame.head(limit).reset_index(drop=True)

    def get_boards(self, secid: str) -> pd.DataFrame:
        secid = self.normalize_secid(secid)
        payload = self._get_json(
            f"securities/{secid}.json",
            {
                "iss.meta": "off",
                "iss.only": "boards",
                "boards.columns": (
                    "secid,boardid,title,market,engine,is_primary,decimals"
                ),
            },
        )
        return self.table_to_frame(payload, "boards")

    def resolve_route(self, secid: str, preferred_board: str | None = None) -> InstrumentRoute:
        secid = self.normalize_secid(secid)
        boards = self.get_boards(secid)
        if boards.empty:
            raise MoexApiError(f"Для {secid} не найдены торговые доски.")

        selected = pd.DataFrame()
        if preferred_board:
            selected = boards[
                boards["boardid"].astype(str).str.upper() == preferred_board.strip().upper()
            ]
            if selected.empty:
                raise MoexApiError(
                    f"Для {secid} не найдена доска {preferred_board.strip().upper()}."
                )

        if selected.empty and "is_primary" in boards.columns:
            selected = boards[pd.to_numeric(boards["is_primary"], errors="coerce") == 1]

        if selected.empty:
            selected = boards

        row = selected.iloc[0]
        required = ("engine", "market", "boardid")
        missing = [name for name in required if pd.isna(row.get(name))]
        if missing:
            raise MoexApiError(
                f"MOEX ISS не вернул маршрут для {secid}; отсутствуют поля: {', '.join(missing)}."
            )

        return InstrumentRoute(
            secid=secid,
            engine=str(row["engine"]),
            market=str(row["market"]),
            board=str(row["boardid"]),
            board_title=None if pd.isna(row.get("title")) else str(row.get("title")),
        )

    def load_candles(
        self,
        route: InstrumentRoute,
        start_date: date | str,
        end_date: date | str,
        interval: int = 24,
    ) -> pd.DataFrame:
        if interval not in {1, 10, 60, 24, 7, 31, 4}:
            raise ValueError("Неподдерживаемый интервал свечей MOEX ISS.")

        path = (
            f"engines/{route.engine}/markets/{route.market}/boards/{route.board}/"
            f"securities/{route.secid}/candles.json"
        )
        parts: list[pd.DataFrame] = []
        offset = 0

        while True:
            payload = self._get_json(
                path,
                {
                    "from": str(start_date),
                    "till": str(end_date),
                    "interval": interval,
                    "start": offset,
                    "iss.meta": "off",
                    "iss.only": "candles",
                },
            )
            page = self.table_to_frame(payload, "candles")
            if page.empty:
                break
            parts.append(page)
            offset += len(page)

        if not parts:
            return pd.DataFrame()

        frame = pd.concat(parts, ignore_index=True)
        frame.columns = [str(column).lower() for column in frame.columns]

        for column in ("begin", "end"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")

        numeric_columns = ("open", "close", "high", "low", "value", "volume")
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

        sort_column = "begin" if "begin" in frame.columns else frame.columns[0]
        frame = frame.sort_values(sort_column).drop_duplicates(subset=[sort_column], keep="last")
        return frame.reset_index(drop=True)
