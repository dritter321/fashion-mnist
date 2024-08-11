import unittest
from unittest.mock import patch, mock_open
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_run_details_to_file

class TestLogger(unittest.TestCase):

    @patch("utils.logger.os.path.exists")
    @patch("utils.logger.open", new_callable=mock_open())
    def test_log_run_details_to_file_creates_file_if_not_exists(self, mock_file, mock_exists):
        mock_exists.return_value = False
        log_run_details_to_file("test_experiment", "123", "456", 0.9, "/path/to/model")

        mock_file.assert_called_once_with(os.path.join("./mlruns", "run_logs.csv"), 'w')
        handle = mock_file()
        handle.write.assert_called_with("timestamp,experiment_name,experiment_id,run_id,execution_time,model_path\n")

    @patch("utils.logger.os.path.exists")
    @patch("utils.logger.open", new_callable=mock_open(
        read_data="timestamp,experiment_name,experiment_id,run_id,execution_time,model_path\nexisting_log_line\n"))
    def test_log_run_details_to_file_appends_to_existing_file(self, mock_file, mock_exists):
        mock_exists.return_value = True
        log_run_details_to_file("test_experiment", "123", "456", 0.9, "/path/to/model")

        handle = mock_file()
        calls = [unittest.mock.call("timestamp,experiment_name,experiment_id,run_id,execution_time,model_path\n"),
                 unittest.mock.call().write(unittest.mock.ANY),  # Checks if any second write occurs
                 unittest.mock.call().truncate()]
        handle.assert_has_calls(calls, any_order=False)


if __name__ == '__main__':
    unittest.main()