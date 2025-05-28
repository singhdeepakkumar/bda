import os
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

from src.pipeline.data import data_loading

@pytest.fixture
def sample_csv_data():
    return "customerID,gender,SeniorCitizen\n001,Male,0\n002,Female,1"

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "customerID": ["001", "002"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1]
    })

def test_load_data(sample_csv_data, sample_dataframe):
    with patch("builtins.open", mock_open(read_data=sample_csv_data)), \
         patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = sample_dataframe
        df = data_loading.load_data("dummy/path.csv")
        assert df.equals(sample_dataframe)
        mock_read_csv.assert_called_once_with("dummy/path.csv")

def test_save_data(tmp_path, sample_dataframe):
    output_path = tmp_path / "test_output.csv"
    data_loading.save_data(sample_dataframe, output_path)
    # Read back and verify
    result = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(result, sample_dataframe)

@patch("src.pipeline.data.data_loading.log_metric")
@patch("src.pipeline.data.data_loading.log_artifact")
@patch("src.pipeline.data.data_loading.start_run")
@patch("src.pipeline.data.data_loading.set_experiment")
@patch("src.pipeline.data.data_loading.load_data")
@patch("src.pipeline.data.data_loading.save_data")
def test_main(mock_save, mock_load, mock_set_exp, mock_start_run, mock_log_artifact, mock_log_metric):
    mock_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mock_load.return_value = mock_df
    os.environ["RAW_DATA_PATH"] = "fake/path.csv"

    mock_run = MagicMock()
    mock_start_run.return_value.__enter__.return_value = mock_run

    data_loading.main()

    mock_set_exp.assert_called_once_with("data_loading")
    mock_log_metric.assert_any_call("rows", 2)
    mock_log_metric.assert_any_call("columns", 2)
    mock_log_artifact.assert_called_once_with("data/raw/telecom_customer_churn_data.csv")
    mock_save.assert_called_once()
