import pandas as pd
import pytest

from asset_lab.data.moex import MoexApiError, MoexClient


def test_table_to_frame() -> None:
    payload = {"securities": {"columns": ["secid", "name"], "data": [["SBER", "Sber"]]}}
    frame = MoexClient.table_to_frame(payload, "securities")
    assert frame.to_dict("records") == [{"secid": "SBER", "name": "Sber"}]


def test_table_to_frame_missing_block() -> None:
    with pytest.raises(MoexApiError):
        MoexClient.table_to_frame({}, "candles")


class FakeClient(MoexClient):
    def get_boards(self, secid: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "secid": secid,
                    "boardid": "TEST",
                    "title": "Secondary",
                    "market": "shares",
                    "engine": "stock",
                    "is_primary": 0,
                },
                {
                    "secid": secid,
                    "boardid": "TQBR",
                    "title": "Primary",
                    "market": "shares",
                    "engine": "stock",
                    "is_primary": 1,
                },
            ]
        )


def test_resolve_route_uses_primary() -> None:
    route = FakeClient().resolve_route("sber")
    assert route.secid == "SBER"
    assert route.board == "TQBR"
    assert route.engine == "stock"
    assert route.market == "shares"
