import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.run_evaluation import OUTPUT_COLUMNS, redact_trace, validate_output


class EvaluationOutputTests(unittest.TestCase):
    def test_required_csv_schema_and_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS); writer.writeheader()
                writer.writerow({"query_id":"Q1","predicted_intent":"track_order","predicted_category":"ORDER","decision":"ANSWER"})
            self.assertEqual(validate_output(path, 1), 1)

    def test_output_rejects_invalid_decision_and_row_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            path.write_text("query_id,predicted_intent,predicted_category,decision\nQ1,x,y,UNKNOWN\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Invalid decision"):
                validate_output(path, 1)
            with self.assertRaisesRegex(RuntimeError, "Row count"):
                validate_output(path, 2)

    def test_trace_redacts_common_direct_identifiers(self):
        cleaned = redact_trace({"query":"Email me at customer@example.com; order ref ABCD12345; call 555-123-4567"})["query"]
        self.assertNotIn("customer@example.com", cleaned)
        self.assertNotIn("ABCD12345", cleaned)
        self.assertNotIn("555-123-4567", cleaned)


if __name__ == "__main__": unittest.main()
