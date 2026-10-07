from __future__ import annotations

import unittest
from pathlib import Path

from app.dataset_search.huggingface import HuggingFaceDatasetClient, DatasetSearchResult
from app.dataset_search.kaggle import KaggleDatasetClient


class TestDatasetSearch(unittest.TestCase):

    def test_kaggle_client_initialization(self):
        client = KaggleDatasetClient()
        self.assertIsNotNone(client)
        self.assertEqual(client.BASE_URL, "https://www.kaggle.com/api/v1")

    def test_dataset_search_result_url_mapping(self):
        # Kaggle result
        kaggle_res = DatasetSearchResult(
            dataset_id="kaggle:heptapod/titanic",
            author="heptapod",
            downloads=1000,
        )
        self.assertEqual(kaggle_res.source_label, "Kaggle")
        self.assertEqual(kaggle_res.url, "https://www.kaggle.com/datasets/heptapod/titanic")
        self.assertEqual(kaggle_res.title, "titanic")

        # Hugging Face result
        hf_res = DatasetSearchResult(
            dataset_id="scikit-learn/iris",
            author="scikit-learn",
            downloads=500,
        )
        self.assertEqual(hf_res.source_label, "Hugging Face")
        self.assertEqual(hf_res.url, "https://huggingface.co/datasets/scikit-learn/iris")
        self.assertEqual(hf_res.title, "iris")

    def test_kaggle_credentials_resolution(self):
        client = KaggleDatasetClient(username="test_user", key="test_key")
        self.assertTrue(client.is_authenticated)
        self.assertEqual(client.auth, ("test_user", "test_key"))


if __name__ == "__main__":
    unittest.main()

