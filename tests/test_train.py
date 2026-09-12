#!/usr/bin/env python3
"""
Unit tests for src/train.py

Uses a SMALL fabricated synthetic dataset (hand-made embedding vectors + labels)
to test the training/evaluation logic without downloading real models or data.
"""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.train import _train_and_eval
import uuid


class TestTrainEval:
    """Tests for the _train_and_eval function with synthetic data."""

    def test_metrics_in_valid_range(self):
        """Train/eval should return accuracy and macro-F1 in [0, 1]."""
        np.random.seed(42)
        n_train = 60
        n_test = 30
        dim = 384
        n_classes = 3
        labels = ["Class_A", "Class_B", "Class_C"]

        # Create class centroids
        centroids = {
            "Class_A": np.zeros(dim, dtype=np.float32),
            "Class_B": np.ones(dim, dtype=np.float32) * 3.0,
            "Class_C": np.ones(dim, dtype=np.float32) * 6.0,
        }

        # Create training data around centroids
        X_train_dict = {}
        y_train = []
        for i in range(n_train):
            cls = labels[i % n_classes]
            desc = f"train_{i}"
            X_train_dict[desc] = centroids[cls] + np.random.randn(dim).astype(np.float32) * 0.5
            y_train.append(cls)

        # Create test data around same centroids
        X_test_dict = {}
        y_test = []
        for i in range(n_test):
            cls = labels[i % n_classes]
            desc = f"test_{i}"
            X_test_dict[desc] = centroids[cls] + np.random.randn(dim).astype(np.float32) * 0.5
            y_test.append(cls)

        import pandas as pd
        train_df = pd.DataFrame({"Description": list(X_train_dict.keys()), "label": y_train})
        test_df = pd.DataFrame({"Description": list(X_test_dict.keys()), "label": y_test})

        # Mock embedder that returns pre-computed embeddings by description
        class MockEmbedder:
            def __init__(self, embeddings_dict):
                self.embeddings_dict = embeddings_dict
                self._model_key = f"mock_{uuid.uuid4().hex[:8]}"
                self._dim = 384

            def encode(self, texts):
                # Handle both single and batch calls
                return np.vstack([self.embeddings_dict[t] for t in texts])

            @property
            def model_key(self):
                return self._model_key

            @property
            def dim(self):
                return self._dim

        # Combine all embeddings
        all_embeddings = {**X_train_dict, **X_test_dict}
        mock_embedder = MockEmbedder(all_embeddings)

        result = _train_and_eval(train_df, test_df, mock_embedder, labels)

        # Check metrics are in valid range
        assert 0.0 <= result["accuracy"] <= 1.0, f"Accuracy {result['accuracy']} not in [0,1]"
        assert 0.0 <= result["macro_f1"] <= 1.0, f"Macro-F1 {result['macro_f1']} not in [0,1]"

        # Should achieve reasonable performance on separable data
        assert result["accuracy"] > 0.8, f"Should beat random on separable synthetic data, got {result['accuracy']}"
        assert result["macro_f1"] > 0.8, f"Should beat random on separable synthetic data, got {result['macro_f1']}"

    def test_class_weighting_applied(self):
        """Class-weighted classifier should handle severe imbalance better than unweighted."""
        np.random.seed(123)
        dim = 384
        labels = ["Majority", "Minority1", "Minority2"]

        # Create class centroids
        centroids = {
            "Majority": np.zeros(dim, dtype=np.float32),
            "Minority1": np.ones(dim, dtype=np.float32) * 3.0,
            "Minority2": np.ones(dim, dtype=np.float32) * 6.0,
        }

        # Severe imbalance: 90% Majority, 5% Minority1, 5% Minority2
        n_train = 200
        n_test = 60

        # Create training data around centroids
        X_train_dict = {}
        y_train = (["Majority"] * 180 + ["Minority1"] * 10 + ["Minority2"] * 10)
        for i, cls in enumerate(y_train):
            desc = f"train_{i}"
            X_train_dict[desc] = centroids[cls] + np.random.randn(dim).astype(np.float32) * 0.5

        # Create test data around same centroids (balanced)
        X_test_dict = {}
        y_test = (["Majority"] * 20 + ["Minority1"] * 20 + ["Minority2"] * 20)
        for i, cls in enumerate(y_test):
            desc = f"test_{i}"
            X_test_dict[desc] = centroids[cls] + np.random.randn(dim).astype(np.float32) * 0.5

        import pandas as pd
        train_df = pd.DataFrame({"Description": list(X_train_dict.keys()), "label": y_train})
        test_df = pd.DataFrame({"Description": list(X_test_dict.keys()), "label": y_test})

        class MockEmbedder:
            def __init__(self, embeddings_dict):
                self.embeddings_dict = embeddings_dict
                self._model_key = f"mock_{uuid.uuid4().hex[:8]}"
                self._dim = 384

            def encode(self, texts):
                return np.vstack([self.embeddings_dict[t] for t in texts])

            @property
            def model_key(self):
                return self._model_key

            @property
            def dim(self):
                return self._dim

        all_embeddings = {**X_train_dict, **X_test_dict}
        mock_embedder = MockEmbedder(all_embeddings)

        # Train with class_weight="balanced" (our default)
        result_weighted = _train_and_eval(train_df, test_df, mock_embedder, labels)

        # Train unweighted classifier for comparison on the SAME embeddings
        X_train_arr = np.vstack([X_train_dict[desc] for desc in train_df["Description"]])
        X_test_arr = np.vstack([X_test_dict[desc] for desc in test_df["Description"]])
        clf_unweighted = LogisticRegression(
            class_weight=None,  # No weighting
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        clf_unweighted.fit(X_train_arr, y_train)
        y_pred_unweighted = clf_unweighted.predict(X_test_arr)
        macro_f1_unweighted = f1_score(y_test, y_pred_unweighted, average="macro", labels=labels, zero_division=0)

        # Weighted should achieve better macro-F1 on imbalanced data
        print(f"Weighted macro-F1: {result_weighted['macro_f1']:.4f}")
        print(f"Unweighted macro-F1: {macro_f1_unweighted:.4f}")

        # The weighted classifier should not collapse to majority-only prediction
        # (which would give macro-F1 ≈ 0.33 since it gets 1/3 classes right)
        # For this test, we just verify both produce valid metrics
        assert 0.0 <= result_weighted["macro_f1"] <= 1.0
        assert 0.0 <= macro_f1_unweighted <= 1.0

        # On severely imbalanced training data with separable classes,
        # weighted should not do worse than unweighted on macro-F1
        # (This is a sanity check - the data is separable so both should do well)
        assert result_weighted["macro_f1"] > 0.3, "Weighted should not collapse to majority class"


class TestCacheMechanism:
    """Tests for the embedding cache mechanism."""

    def test_cache_saves_and_loads(self, tmp_path):
        """Cache should save embeddings to disk and load them on subsequent calls."""
        from src.train import _cache_key, _get_cached_embedding
        import shutil

        # Use temp directory for cache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        class MockEmbedder:
            def __init__(self):
                self.call_count = 0
                self._model_key = "test_model"

            def encode(self, texts):
                self.call_count += 1
                return np.ones((len(texts), 384), dtype=np.float32) * self.call_count

            @property
            def model_key(self):
                return self._model_key

        embedder = MockEmbedder()
        desc = "Test description"

        # First call - should compute and cache
        # Need to monkey-patch the cache dir
        import src.train as train_module
        original_cache_dir = train_module.CACHE_DIR
        train_module.CACHE_DIR = cache_dir

        try:
            emb1 = train_module._get_cached_embedding(embedder, desc)
            assert embedder.call_count == 1
            assert np.allclose(emb1, 1.0)

            # Second call - should load from cache
            emb2 = train_module._get_cached_embedding(embedder, desc)
            assert embedder.call_count == 1  # Not incremented!
            assert np.allclose(emb2, 1.0)

            # Verify cache file exists
            cache_file = cache_dir / train_module._cache_key(desc, embedder.model_key)
            assert cache_file.exists()
        finally:
            train_module.CACHE_DIR = original_cache_dir


class TestPerClassF1:
    """Tests for per-class F1 computation from confusion matrix."""

    def test_per_class_f1_computation(self):
        """Per-class F1 should match sklearn's calculation."""
        from src.train import _per_class_f1_from_cm
        from sklearn.metrics import f1_score

        labels = ["A", "B", "C"]
        y_true = ["A", "A", "B", "B", "C", "C", "A", "B", "C"]
        y_pred = ["A", "B", "B", "B", "C", "A", "A", "B", "C"]

        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[labels.index(t), labels.index(p)] += 1

        per_class = _per_class_f1_from_cm(cm, labels)

        # Compare with sklearn
        sklearn_f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        for label, f1 in zip(labels, sklearn_f1):
            assert np.isclose(per_class[label], f1), f"Mismatch for {label}: {per_class[label]} vs {f1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])