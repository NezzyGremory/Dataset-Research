import os
import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyzer.data_quality import DataQualityDiagnoser
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
