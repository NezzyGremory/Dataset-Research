import os
import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzer.data_quality import DataQualityDiagnoser
from app.analyzer.missing_values import (
    MissingValueAnalyzer,
    drop_missing_rows,
    impute_missing_values,
)
from app.storage.database import Database
from app.storage.project_repository import ProjectRepository
from app.storage.version_manager import DatasetVersionManager


class TestAnalysisPrepFlow(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.db = Database(db_path=self.db_path)
        self.repo = ProjectRepository(database=self.db)
        self.vm = DatasetVersionManager(database=self.db, data_dir=Path(self.temp_dir) / "data")
        self.project_id = self.repo.create_project(name="Test Analysis Flow")

        self.df = pd.DataFrame({
            "PassengerId": [1, 2, 3, 4, 5, 5],
            "Survived": [0, 1, 1, 0, 1, 1],
            "Age": [22.0, 38.0, None, 35.0, 54.0, 54.0],
            "Fare": [7.25, 71.28, 7.92, 53.1, 500.0, 500.0],
            "Sex": ["male", "female", "female", "female", "male", "male"],
        })

    def test_impute_missing_creates_version(self):
        # Initial version v0
        v0 = self.vm.create_initial_version(self.project_id, dataframe=self.df)
        self.assertEqual(v0.version, 0)
        self.assertEqual(self.df["Age"].isna().sum(), 1)

        # Impute missing values (median for numeric, mode for categorical)
        df_imputed = self.df.copy()
        for col in df_imputed.columns:
            if df_imputed[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_imputed[col]):
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                else:
                    mode_val = df_imputed[col].mode().iloc[0]
                    df_imputed[col] = df_imputed[col].fillna(mode_val)

        v1 = self.vm.create_version(
            project_id=self.project_id,
            dataframe=df_imputed,
            operation="impute_missing",
            parameters={"method": "median_mode"},
            impact={"missing_before": 1, "missing_after": 0},
            description="Imputed missing values",
            label="Missing Values Imputed",
        )

        self.assertEqual(v1.version, 1)
        self.assertEqual(df_imputed["Age"].isna().sum(), 0)

        # Verify raw v0 is immutable
        df_v0 = self.vm.load_dataframe(self.project_id, 0)
        self.assertEqual(df_v0["Age"].isna().sum(), 1)

    def test_imputer_cleans_mixed_types_and_drops_fully_empty_columns(self):
        dataframe = pd.DataFrame({
            "measurement": [2.0, 4.0, None],
            "category": ["x", None, "x"],
            "empty": [None, None, None],
            "flag": pd.Series([True, None, True], dtype="boolean"),
            "date": pd.to_datetime(["2024-01-01", None, "2024-01-01"]),
        })

        cleaned, report = impute_missing_values(dataframe)

        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        self.assertEqual(cleaned["measurement"].iloc[2], 3.0)
        self.assertEqual(cleaned["category"].iloc[1], "x")
        self.assertEqual(cleaned["flag"].dtype.name, "boolean")
        self.assertEqual(cleaned["date"].iloc[1], pd.Timestamp("2024-01-01"))
        self.assertNotIn("empty", cleaned.columns)
        self.assertEqual(report["missing_before"], 7)
        self.assertEqual(report["missing_after"], 0)
        self.assertEqual(report["dropped_empty_columns"], ["empty"])

    def test_imputer_promotes_nullable_integer_for_fractional_median(self):
        dataframe = pd.DataFrame({
            "count": pd.Series([1, None, 2], dtype="Int64"),
        })

        cleaned, report = impute_missing_values(dataframe)

        self.assertEqual(cleaned["count"].iloc[1], 1.5)
        self.assertEqual(cleaned["count"].dtype.name, "Float64")
        self.assertEqual(report["missing_after"], 0)

    def test_imputer_rejects_dataset_with_no_observed_values(self):
        dataframe = pd.DataFrame({
            "first": [None, None],
            "second": [None, None],
        })

        with self.assertRaisesRegex(ValueError, "Semua kolom berisi nilai kosong"):
            impute_missing_values(dataframe)

    def test_imputer_cleans_blank_excel_cells_and_common_null_tokens(self):
        values = ["padi"] * 30 + [""] * 20 + ["   "] * 15 + ["NA"] * 15 + ["null"] * 10 + ["-"] * 10
        dataframe = pd.DataFrame({"jenis": values, "hasil": range(len(values))})
        analyzer = MissingValueAnalyzer()

        before = analyzer.analyze(dataframe)
        self.assertEqual(before[0]["missing_count"], 70)

        cleaned, report = impute_missing_values(dataframe)

        self.assertEqual(report["missing_before"], 70)
        self.assertEqual(report["missing_after"], 0)
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        self.assertEqual(int(dataframe.isna().sum().sum()), 0)

    def test_drop_missing_rows_removes_every_row_with_blank_or_missing_marker(self):
        dataframe = pd.DataFrame({
            "jenis": pd.Series(
                ["padi", "   ", "?", "beras", "N/A", "\u200b"],
                dtype="category",
            ),
            "hasil": [10, 20, 30, None, 50, 60],
        })
        original = dataframe.copy(deep=True)

        cleaned, report = drop_missing_rows(dataframe)

        self.assertEqual(cleaned.to_dict(orient="records"), [{"jenis": "padi", "hasil": 10.0}])
        self.assertEqual(int(cleaned.isna().sum().sum()), 0)
        self.assertEqual(report["missing_before"], 5)
        self.assertEqual(report["missing_after"], 0)
        self.assertEqual(report["rows_with_missing"], 5)
        self.assertEqual(report["rows_dropped"], 5)
        pd.testing.assert_frame_equal(dataframe, original)

    def test_drop_missing_rows_refuses_to_replace_dataset_with_no_complete_rows(self):
        dataframe = pd.DataFrame({"first": [None, " "], "second": ["NA", None]})

        with self.assertRaisesRegex(ValueError, "Tidak ada baris yang seluruh kolomnya terisi"):
            drop_missing_rows(dataframe)

    def test_remove_duplicates_creates_version(self):
        self.vm.create_initial_version(self.project_id, dataframe=self.df)
        dup_count = int(self.df.duplicated().sum())
        self.assertEqual(dup_count, 1)

        df_cleaned = self.df.drop_duplicates()
        v1 = self.vm.create_version(
            project_id=self.project_id,
            dataframe=df_cleaned,
            operation="remove_duplicates",
            parameters={"keep": "first"},
            impact={"rows_before": len(self.df), "rows_after": len(df_cleaned)},
            description="Removed duplicates",
            label="Duplicates Removed",
        )

        self.assertEqual(v1.version, 1)
        self.assertEqual(len(df_cleaned), 5)
        self.assertEqual(df_cleaned.duplicated().sum(), 0)

        # Verify trail
        trail = self.vm.get_trail(self.project_id)
        self.assertEqual(len(trail), 1)
        self.assertEqual(trail[0].operation, "remove_duplicates")


if __name__ == "__main__":
    unittest.main()
