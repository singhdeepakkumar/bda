import pandas as pd
from unittest.mock import patch
import pytest

from src.feature_engineering import load_data, process, save_data


@patch("pandas.read_csv")
def test_load_data(mock_read_csv):
    dummy_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    mock_read_csv.return_value = dummy_df

    result = load_data("fake_path.csv")
    mock_read_csv.assert_called_once_with("fake_path.csv")
    pd.testing.assert_frame_equal(result, dummy_df)


def test_process_removes_columns_and_encodes():
    raw_data = pd.DataFrame({
        'Customer ID': ['0001', '0002'],
        'City': ['CityA', 'CityB'],
        'Zip Code': [12345, 67890],
        'Latitude': [12.34, 56.78],
        'Longitude': [98.76, 54.32],
        'Customer Status': ['Churned', 'Joined'],
        'Contract': ['Month-to-month', 'One year']
    })

    processed = process(raw_data)

    # Check removed row
    assert processed.shape[0] == 1  # 'Joined' should be removed

    # Check dropped columns
    for col in ['Customer ID', 'City', 'Zip Code', 'Latitude', 'Longitude']:
        assert col not in processed.columns

    # Check label encoding applied
    assert pd.api.types.is_integer_dtype(processed['Contract'])


def test_save_data(tmp_path):
    df = pd.DataFrame({"A": [1], "B": [2]})
    file_path = tmp_path / "output.csv"

    save_data(df, file_path)
    loaded = pd.read_csv(file_path)

    pd.testing.assert_frame_equal(df, loaded)
