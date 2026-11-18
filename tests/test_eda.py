"""Small tests for date filtering and field selection helpers."""

import pandas as pd

from src.eda import date_mask, first_existing


def test_date_mask_includes_2016_through_2021():
    dates = pd.Series([
        1451606400000,  # 2016-01-01 UTC
        1640908800000,  # 2021-12-31 UTC
        1640995200000,  # 2022-01-01 UTC
    ])
    result = date_mask(dates)
    assert result.tolist() == [True, True, False]


def test_first_existing_returns_first_available_candidate():
    columns = ["OBJECTID", "TC_ID", "Can_P"]
    assert first_existing(columns, ["TC_ID", "OBJECTID"]) == "TC_ID"
    assert first_existing(columns, ["missing", "Can_P"]) == "Can_P"
    assert first_existing(columns, ["missing"]) is None
