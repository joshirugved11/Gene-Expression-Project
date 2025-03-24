import unittest
import pandas as pd
import sys

sys.path.append("..")
from preprocessing.encode_labels import encode_labels
from preprocessing.feature_selection import select_features
from preprocessing.normalisation import normalize_data

# Define necessary paths
RAW_DATA_PATH = "data/raw/dataset.csv"

class TestPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load dataset once for all tests."""
        cls.df = pd.read_csv(RAW_DATA_PATH)

    def test_encode_labels(self):
        df_encoded = encode_labels(self.df)
        self.assertFalse(df_encoded.isnull().values.any(), "Encoding should not introduce null values.")

    def test_feature_selection(self):
        df_encoded = encode_labels(self.df)
        df_selected = select_features(df_encoded)
        self.assertEqual(df_selected.shape[1], 10, "Feature selection should return exactly 10 features.")

    def test_normalization(self):
        df_encoded = encode_labels(self.df)
        df_selected = select_features(df_encoded)
        df_normalized = normalize_data(df_selected)
        self.assertAlmostEqual(df_normalized.mean().mean(), 0, delta=1e-1, msg="Normalized data mean should be close to 0.")

if __name__ == "__main__":
    unittest.main()
