import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import log_run_details_to_file

class TestLogger(unittest.TestCase):

    def test_log_run_details_to_file_creates_file_if_not_exists(self, mock_file, mock_exists):
        log_run_details_to_file("test_experiment", "123", "456", 0.9, "/path/to/model")
        self.assertTrue(os.path.exists("./mlruns/run_logs.csv"))

    def test_log_run_details_to_file_appends_to_existing_file(self, mock_file, mock_exists):
        log_run_details_to_file("test_experiment", "123", "456", 0.9, "/path/to/model")
        header_expected = "timestamp,experiment_name,experiment_id,run_id,execution_time,model_path"
        with open("./mlruns/run_logs.csv", 'r') as file:
            first_line = file.readline().strip()
        self.assertEqual(first_line, header_expected)



if __name__ == '__main__':
    unittest.main()