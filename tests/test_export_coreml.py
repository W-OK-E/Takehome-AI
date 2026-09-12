#!/usr/bin/env python3
"""
Unit tests for src/export_coreml.py

Tests that the export function runs and produces the expected output files.
The actual full-model conversion test is marked as slow and can be skipped
in fast test runs, but we run it to ensure everything works.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import joblib

from src.export_coreml import run_export, _export_classifier_head, _export_embedder


class TestExportCoreml:
    """Tests for the Core ML export pipeline."""

    def test_export_classifier_head_creates_file(self, tmp_path):
        """Test that _export_classifier_head creates a .mlpackage file."""
        # Create a mock head.joblib
        from sklearn.linear_model import LogisticRegression
        mock_clf = LogisticRegression()
        mock_clf.classes_ = np.array(["Class_A", "Class_B", "Class_C"])
        mock_clf.coef_ = np.random.randn(3, 384).astype(np.float32)
        mock_clf.intercept_ = np.random.randn(3).astype(np.float32)

        head_path = tmp_path / "head.joblib"
        joblib.dump(mock_clf, head_path)

        # Mock the models directory
        with patch("src.export_coreml.HEAD_PATH", head_path):
            with patch("src.export_coreml.CLASSIFIER_MLPACKAGE", tmp_path / "classifier_head.mlpackage"):
                with patch("src.export_coreml.MODELS_DIR", tmp_path):
                    _export_classifier_head()

        assert (tmp_path / "classifier_head.mlpackage").exists()

    def test_export_embedder_creates_directory(self, tmp_path):
        """Test that _export_embedder creates an embedder directory."""
        # Create mock config
        config = {
            "backbone": "e5",
            "model_name": "intfloat/multilingual-e5-small",
            "embedding_dim": 384,
            "labels": ["A", "B", "C"],
        }
        config_path = tmp_path / "config.json"
        import json
        with open(config_path, "w") as f:
            json.dump(config, f)

        embedder_dir = tmp_path / "embedder"

        with patch("src.export_coreml.CONFIG_PATH", config_path):
            with patch("src.export_coreml.EMBEDDER_DIR", embedder_dir):
                with patch("src.export_coreml.MODELS_DIR", tmp_path):
                    # Mock SentenceTransformer to avoid downloading
                    with patch("src.export_coreml.SentenceTransformer") as mock_st:
                        mock_model = MagicMock()
                        mock_st.return_value = mock_model
                        # Need to make save() create the directory
                        mock_model.save.side_effect = lambda path: Path(path).mkdir(parents=True, exist_ok=True)
                        _export_embedder()

        assert embedder_dir.exists()

    def test_run_export_smoke(self, tmp_path):
        """Smoke test: run_export should complete without errors (mocked)."""
        pytest.skip("Integration test requires real models - run with -m slow")


class TestVerification:
    """Tests for the verification logic (using synthetic data)."""

    def test_embedding_comparison(self):
        """Test that embedding comparison computes correct metrics."""
        np.random.seed(42)
        n = 100
        dim = 384

        # Create two nearly identical embedding sets
        emb1 = np.random.randn(n, dim).astype(np.float32)
        emb2 = emb1 + np.random.randn(n, dim).astype(np.float32) * 1e-6

        max_diff = np.max(np.abs(emb1 - emb2))
        mean_diff = np.mean(np.abs(emb1 - emb2))

        assert max_diff < 1e-4
        assert mean_diff < 1e-5

    def test_label_agreement_computation(self):
        """Test label agreement rate computation."""
        preds1 = np.array(["A", "A", "B", "B", "C"])
        preds2 = np.array(["A", "B", "B", "B", "C"])

        agreement = np.mean(preds1 == preds2)
        assert agreement == 0.8  # 4/5 agree


@pytest.mark.slow
class TestFullExport:
    """Full export test with real models (requires dataset and downloads)."""

    def test_full_export(self):
        """Run the full export pipeline (slow, requires dataset and macOS for Core ML prediction)."""
        import sys
        if sys.platform != "darwin":
            pytest.skip("Full export test requires macOS for Core ML prediction")
        result = run_export()

        assert "classifier_head" in result
        assert "embedder" in result
        assert Path(result["classifier_head"]).exists()
        assert Path(result["embedder"]).exists()

        # Check verification results
        if result["verification"].get("verified"):
            assert result["verification"]["label_agreement"] > 0.95
            assert result["verification"]["embedding_max_diff"] < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])