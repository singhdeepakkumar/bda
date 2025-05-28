import os
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open

from src.pipeline.data import data_preprocessing

@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "Internet Service": ["No", "DSL", "No"],
        "Phone Service": ["No", "Yes", "No"],
        "Avg Monthly GB Download": [None, 15.5, None],
        "Avg Monthly Long Distance Charges": [None, 20.0, None],
        "Internet Type": [None, "DSL", None],
        "Online Security": [None, "Yes", None],
        "Online Backup": [None, "No", None],
        "Device Protection Plan": [None, "Yes", None],
        "Premium Tech Support": [None, "No", None],
        "Streaming TV": [None, "Yes", None],
        "Streaming Movies": [None, "Yes", None],
        "Streaming Music": [None, "No", None],
        "Unlimited Data": [None, "Yes", None],
        "Multiple Lines": [None, "Yes", None],
        "Offer": [None, "Offer A", None],
        "Churn Category": ["Competitor", "Dissatisfaction", "Other"],
        "Churn Reason": ["Reason A", "Reason B", "Reason C"]
    })

def test_load_data():
    with patch("pandas.read_csv") as mock_read:
        mock_df = pd.DataFrame({"a": [1, 2]})
        mock_read.return_value = mock_df
        df = data_preprocessing.load_data("some_path.csv")
        assert df.equals(mock_df)
        mock_read.assert_called_once_with("some_path.csv")

def test_preprocess(raw_df):
    with patch("src.data_pipeline.data_preprocessing.log_metric") as mock_log_metric:
        cleaned_df = data_preprocessing.preprocess(raw_df.copy())

        # Check no nulls in processed columns
        assert cleaned_df['Avg Monthly GB Download'].isnull().sum() == 0
        assert cleaned_df['Avg Monthly Long Distance Charges'].isnull().sum() == 0
        assert (cleaned_df['Internet Type'] == "No internet service").sum() == 2
        assert (cleaned_df['Multiple Lines'] == "No phone service").sum() == 2
        assert cleaned_df['Offer'].isnull().sum() == 0

        # Ensure Churn columns are dropped
        assert 'Churn Category' not in cleaned_df.columns
        assert 'Churn Reason' not in cleaned_df.columns

        # Check metrics logged
        mock_log_metric.assert_any_call("rows:", cleaned_df.shape[0])
        mock_log_metric.assert_any_call("columns:", cleaned_df.shape[1])

def test_save_data(tmp_path):
    df = pd.DataFrame({"x": [1, 2]})
    file_path = tmp_path / "output.csv"
    data_preprocessing.save_data(df, file_path)

    loaded = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(df, loaded)

@patch("src.pipeline.data.data_preprocessing.save_data")
@patch("src.pipeline.data.data_preprocessing.preprocess")
@patch("src.pipeline.data.data_preprocessing.load_data")
@patch("src.pipeline.data.data_preprocessing.load_params")
@patch("src.pipeline.data.data_preprocessing.start_run")
@patch("src.pipeline.data.data_preprocessing.set_experiment")
def test_main(mock_set_exp, mock_start_run, mock_load_params, mock_load_data, mock_preprocess, mock_save_data):
    df_mock = pd.DataFrame({"a": [1, 2]})
    mock_load_data.return_value = df_mock
    mock_preprocess.return_value = df_mock

    run_mock = MagicMock()
    mock_start_run.return_value.__enter__.return_value = run_mock

    data_preprocessing.main()

    mock_set_exp.assert_called_once_with("data_preprocessing")
    mock_load_data.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_save_data.assert_called_once()
