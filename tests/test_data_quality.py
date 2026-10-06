import unittest
import pandas as pd
from app.analyzer.data_quality import DataQualityDiagnoser, QualityDiagnosisItem


class TestDataQualityDiagnoser(unittest.TestCase):

    def setUp(self):
        self.diagnoser = DataQualityDiagnoser()

    def test_clean_dataset_no_issues(self):
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": ["A", "B", "C", "D", "E"],
        })
        issues = self.diagnoser.diagnose(df)
        # No duplicates, no missing, no constant, etc.
        missing_or_dup = [i for i in issues if i.category in ("missing_values", "duplicates")]
        self.assertEqual(len(missing_or_dup), 0)

    def test_duplicate_diagnosis(self):
        df = pd.DataFrame({
            "a": [1, 2, 2, 3],
            "b": ["x", "y", "y", "z"],
        })
        issues = self.diagnoser.diagnose(df)
        dup_issue = next((i for i in issues if i.category == "duplicates"), None)
        self.assertIsNotNone(dup_issue)
        self.assertIn("1 baris", dup_issue.magnitude)
        self.assertTrue(len(dup_issue.options) > 0)
        self.assertTrue("Apa Dampaknya" not in dup_issue.impact)  # Impact text is meaningful
        self.assertIn("data leakage", dup_issue.impact.lower())

    def test_missing_values_diagnosis(self):
        df = pd.DataFrame({
            "col1": [1, None, None, 4],
            "col2": ["a", "b", "c", "d"],
        })
        issues = self.diagnoser.diagnose(df)
        miss_issue = next((i for i in issues if i.category == "missing_values"), None)
        self.assertIsNotNone(miss_issue)
        self.assertIn("col1", miss_issue.problem)
        self.assertIn("Imputasi", miss_issue.options[0])

    def test_outlier_diagnosis(self):
        df = pd.DataFrame({
            "val": [10, 11, 12, 11, 10, 12, 11, 1000],  # 1000 is extreme outlier
        })
        issues = self.diagnoser.diagnose(df)
        outlier_issue = next((i for i in issues if i.category == "outliers"), None)
        self.assertIsNotNone(outlier_issue)
        self.assertIn("val", outlier_issue.affected_columns)
        self.assertIn("RobustScaler", outlier_issue.options[0])

    def test_constant_column(self):
        df = pd.DataFrame({
            "constant_col": [99, 99, 99, 99],
            "valid_col": [1, 2, 3, 4],
        })
        issues = self.diagnoser.diagnose(df)
        const_issue = next((i for i in issues if i.category == "constant_columns"), None)
        self.assertIsNotNone(const_issue)
        self.assertIn("constant_col", const_issue.affected_columns)


if __name__ == "__main__":
    unittest.main()
