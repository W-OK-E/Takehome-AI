#!/usr/bin/env python3
"""
Unit tests for src/embed.py

Tests cover:
- encode() returns the right shape for a batch of Portuguese and English strings mixed together
- Identical input strings produce identical (or near-identical) embeddings
- Clearly different strings produce different embeddings (cosine similarity well below 1.0)
- Only tested with MiniLM backbone to avoid downloading both large models
"""
import numpy as np
import pytest

from src.embed import Embedder, get_embedder


@pytest.fixture(scope="module")
def embedder():
    """Create a MiniLM embedder for testing (module scope to reuse model)."""
    return Embedder("minilm")


class TestEmbedderShape:
    """Tests for output shape correctness."""

    def test_single_string(self, embedder):
        """Single string should return shape (1, 384)."""
        result = embedder.encode(["Hello world"])
        assert result.shape == (1, 384)

    def test_batch_of_strings(self, embedder):
        """Batch of strings should return shape (n, 384)."""
        texts = ["Text 1", "Text 2", "Text 3"]
        result = embedder.encode(texts)
        assert result.shape == (3, 384)

    def test_mixed_portuguese_english(self, embedder):
        """Mixed Portuguese and English strings should work correctly."""
        texts = [
            "Excavação de fundação para edifício",  # Portuguese
            "Foundation excavation for building",   # English
            "Concretagem de laje de piso",          # Portuguese
            "Concrete floor slab pouring",          # English
            "Estrutura metálica do telhado",        # Portuguese
        ]
        result = embedder.encode(texts)
        assert result.shape == (5, 384)

    def test_empty_list(self, embedder):
        """Empty list should return empty array with correct dim."""
        result = embedder.encode([])
        assert result.shape == (0, 384)


class TestEmbedderConsistency:
    """Tests for embedding consistency."""

    def test_identical_inputs_produce_identical_embeddings(self, embedder):
        """Same input string should produce identical embeddings."""
        text = "Excavação de fundação"
        result1 = embedder.encode([text])
        result2 = embedder.encode([text])

        # Should be exactly identical (deterministic model)
        assert np.allclose(result1, result2, atol=1e-6)

    def test_identical_inputs_in_batch(self, embedder):
        """Same string repeated in batch should have identical rows."""
        text = "Concretagem de laje"
        result = embedder.encode([text, text, text])

        # All three rows should be identical
        assert np.allclose(result[0], result[1], atol=1e-6)
        assert np.allclose(result[1], result[2], atol=1e-6)

    def test_different_strings_produce_different_embeddings(self, embedder):
        """Clearly different strings should have low cosine similarity."""
        texts = [
            "Excavação de fundação para edifício residencial",  # Construction - excavation
            "Instalação de tubulação de água e esgoto",         # Construction - plumbing
            "Pintura de paredes internas com tinta látex",      # Construction - painting
        ]
        result = embedder.encode(texts)

        # Compute cosine similarities (embeddings are L2 normalized)
        sim_01 = np.dot(result[0], result[1])
        sim_02 = np.dot(result[0], result[2])
        sim_12 = np.dot(result[1], result[2])

        # All similarities should be well below 1.0 (not identical)
        # These are semantically different construction activities
        assert sim_01 < 0.9, f"Excavation vs plumbing too similar: {sim_01:.3f}"
        assert sim_02 < 0.9, f"Excavation vs painting too similar: {sim_02:.3f}"
        assert sim_12 < 0.9, f"Plumbing vs painting too similar: {sim_12:.3f}"


class TestEmbedderProperties:
    """Tests for embedder properties."""

    def test_dim_property(self, embedder):
        """dim property should return 384."""
        assert embedder.dim == 384

    def test_model_key_property(self, embedder):
        """model_key property should return 'minilm'."""
        assert embedder.model_key == "minilm"

    def test_model_name_property(self, embedder):
        """model_name should return the correct HF model name."""
        assert embedder.model_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def test_lazy_loading(self):
        """Model should not be loaded until encode() is called."""
        embedder = Embedder("minilm")
        # Before encode, model should be None
        assert embedder._model is None

        # After encode, model should be loaded
        embedder.encode(["test"])
        assert embedder._model is not None


class TestEmbedderFactory:
    """Tests for the factory function."""

    def test_get_embedder_minilm(self):
        """Factory should create minilm embedder by default."""
        embedder = get_embedder("minilm")
        assert isinstance(embedder, Embedder)
        assert embedder.model_key == "minilm"

    def test_get_embedder_e5(self):
        """Factory should create e5 embedder when requested."""
        embedder = get_embedder("e5")
        assert isinstance(embedder, Embedder)
        assert embedder.model_key == "e5"
        # e5 should have the query prefix configured
        assert embedder._config["prefix"] == "query: "

    def test_invalid_model_key(self):
        """Invalid model key should raise ValueError."""
        with pytest.raises(ValueError):
            get_embedder("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])