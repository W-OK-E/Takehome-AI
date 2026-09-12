#!/usr/bin/env python3
"""
Embedding backbone wrapper for multilingual sentence embeddings.

Provides a lazy-loaded Embedder class that wraps sentence-transformers models.
Supports both candidate backbones from the design spec:
- paraphrase-multilingual-MiniLM-L12-v2 (no prefix needed)
- multilingual-e5-small (requires "query: " prefix)
"""
from typing import Literal

import numpy as np
import torch

# Model configurations
MODEL_CONFIGS = {
    "minilm": {
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prefix": "",
        "dim": 384,
    },
    "e5": {
        "model_name": "intfloat/multilingual-e5-small",
        "prefix": "query: ",
        "dim": 384,
    },
}

ModelKey = Literal["minilm", "e5"]


def _get_default_device() -> str:
    """
    Determine the best available device.

    Returns "cpu" if CUDA is not available or if the GPU compute capability
    is not supported by the current PyTorch build.
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        # Check if the GPU compute capability is supported
        major, minor = torch.cuda.get_device_capability()
        # PyTorch 2.14+ with CUDA 13 supports CC >= 7.5 (sm_75)
        if major < 7 or (major == 7 and minor < 5):
            return "cpu"
    except Exception:
        # If we can't determine capability, play safe
        return "cpu"

    return "cuda"


class Embedder:
    """
    Lazy-loaded sentence embedding model wrapper.

    The underlying sentence-transformers model is only loaded on first call to encode().
    This makes importing the module cheap and allows selecting the model at runtime.
    """

    def __init__(self, model_key: ModelKey = "minilm", device: str | None = None):
        """
        Initialize the embedder with a model choice.

        Args:
            model_key: Either "minilm" for paraphrase-multilingual-MiniLM-L12-v2
                       or "e5" for intfloat/multilingual-e5-small
            device: Device to run on ("cpu", "cuda", etc.). Defaults to CPU if not specified
                    or if CUDA is not available/compatible.
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model_key: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")

        self._model_key = model_key
        self._config = MODEL_CONFIGS[model_key]
        self._model = None
        self._device = device or _get_default_device()

    def _load_model(self):
        """Load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._config["model_name"], device=self._device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of input strings (Portuguese, English, or mixed)

        Returns:
            np.ndarray of shape (n, 384) where n = len(texts)
        """
        if not texts:
            return np.empty((0, self._config["dim"]), dtype=np.float32)

        self._load_model()

        # Apply model-specific prefix if needed (e5 requires "query: ")
        prefix = self._config["prefix"]
        if prefix:
            texts = [prefix + t for t in texts]

        # Encode with sentence-transformers
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    @property
    def dim(self) -> int:
        """Return the embedding dimension (384 for both models)."""
        return self._config["dim"]

    @property
    def model_key(self) -> str:
        """Return the model key used."""
        return self._model_key

    @property
    def model_name(self) -> str:
        """Return the full HuggingFace model name."""
        return self._config["model_name"]

    @property
    def device(self) -> str:
        """Return the device the model runs on."""
        return self._device


def get_embedder(model_key: ModelKey = "minilm", device: str | None = None) -> Embedder:
    """
    Factory function to get an Embedder instance.

    Args:
        model_key: Either "minilm" or "e5"
        device: Device to run on ("cpu", "cuda", etc.)

    Returns:
        Embedder instance
    """
    return Embedder(model_key, device)