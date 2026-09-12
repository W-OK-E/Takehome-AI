#!/usr/bin/env python3
"""
Training script for BOQ trade-package classifier.

Runs 3-fold CV for both candidate backbones, picks the best by macro-F1,
retrains on all data, and saves the head + config.
"""
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import cv_splits, get_label_names, load_and_clean_all_folds
from src.embed import Embedder, MODEL_CONFIGS


CACHE_DIR = Path("models/cache")
HEAD_PATH = Path("models/head.joblib")
CONFIG_PATH = Path("models/config.json")


def _ensure_dirs():
    """Create necessary directories."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)


def _cache_key(description: str, model_key: str) -> str:
    """Generate a filesystem-safe cache key for a description + model."""
    import hashlib
    h = hashlib.sha256(f"{model_key}:{description}".encode()).hexdigest()[:16]
    return f"{model_key}_{h}.npy"


def _get_cached_embedding(embedder: Embedder, description: str) -> np.ndarray:
    """Get embedding for a single description, using cache if available."""
    cache_file = CACHE_DIR / _cache_key(description, embedder.model_key)
    if cache_file.exists():
        return np.load(cache_file)
    # Compute and cache
    emb = embedder.encode([description])[0]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, emb)
    return emb


def _get_embeddings_cached(embedder: Embedder, descriptions: list[str]) -> np.ndarray:
    """Get embeddings for a list of descriptions, using cache with batch processing."""
    # Check which descriptions are already cached
    cached_embeddings = {}
    uncached_descriptions = []
    uncached_indices = []

    for i, desc in enumerate(descriptions):
        cache_file = CACHE_DIR / _cache_key(desc, embedder.model_key)
        if cache_file.exists():
            cached_embeddings[i] = np.load(cache_file)
        else:
            uncached_descriptions.append(desc)
            uncached_indices.append(i)

    # Batch embed uncached descriptions
    if uncached_descriptions:
        new_embeddings = embedder.encode(uncached_descriptions)
        # Save to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for desc, emb in zip(uncached_descriptions, new_embeddings):
            cache_file = CACHE_DIR / _cache_key(desc, embedder.model_key)
            np.save(cache_file, emb)

    # Assemble full embedding matrix in original order
    embeddings = np.empty((len(descriptions), embedder.dim), dtype=np.float32)
    for i, desc in enumerate(descriptions):
        if i in cached_embeddings:
            embeddings[i] = cached_embeddings[i]
        else:
            # Find the index in uncached
            idx = uncached_indices.index(i)
            embeddings[i] = new_embeddings[idx]

    return embeddings


def _train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedder: Embedder,
    label_names: list[str],
) -> dict[str, Any]:
    """
    Train a class-weighted LogisticRegression and evaluate on test set.

    Returns dict with accuracy, macro_f1, and confusion_matrix.
    """
    # Get embeddings
    X_train = _get_embeddings_cached(embedder, train_df["Description"].tolist())
    X_test = _get_embeddings_cached(embedder, test_df["Description"].tolist())
    y_train = train_df["label"].tolist()
    y_test = test_df["label"].tolist()

    # Train class-weighted multinomial logistic regression
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", labels=label_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=label_names)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classifier": clf,
    }


def _run_cv_for_backbone(
    backbone_key: str,
    dataset_dir: Path,
    label_names: list[str],
) -> dict[str, Any]:
    """
    Run 3-fold CV for a single backbone.

    Returns dict with per-fold metrics and averaged metrics.
    """
    embedder = Embedder(backbone_key)
    fold_results = []

    print(f"\n{'='*60}")
    print(f"Running CV for backbone: {backbone_key} ({MODEL_CONFIGS[backbone_key]['model_name']})")
    print(f"{'='*60}")

    for fold_idx, (train_df, val_df, test_df) in enumerate(cv_splits(dataset_dir)):
        print(f"\n  Fold {fold_idx + 1}/3: train={len(train_df)}, test={len(test_df)}")
        start = time.time()
        result = _train_and_eval(train_df, test_df, embedder, label_names)
        elapsed = time.time() - start
        fold_results.append(result)
        print(f"    accuracy={result['accuracy']:.4f}, macro_f1={result['macro_f1']:.4f}, time={elapsed:.1f}s")

    # Average across folds
    mean_acc = np.mean([r["accuracy"] for r in fold_results])
    mean_macro_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_acc = np.std([r["accuracy"] for r in fold_results])
    std_macro_f1 = np.std([r["macro_f1"] for r in fold_results])

    # Average confusion matrix (sum then normalize)
    avg_cm = np.mean([r["confusion_matrix"] for r in fold_results], axis=0)

    print(f"\n  Mean accuracy:  {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean macro-F1:  {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")

    return {
        "backbone": backbone_key,
        "fold_results": fold_results,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "avg_confusion_matrix": avg_cm,
    }


def _print_comparison_table(results: list[dict[str, Any]]):
    """Print a comparison table of the backbones."""
    print(f"\n{'='*70}")
    print("BACKBONE COMPARISON (3-fold CV)")
    print(f"{'='*70}")
    print(f"{'Backbone':<30} {'Accuracy':>15} {'Macro-F1':>15}")
    print(f"{'-'*70}")
    for r in results:
        print(f"{r['backbone']:<30} {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}   {r['mean_macro_f1']:.4f} ± {r['std_macro_f1']:.4f}")
    print(f"{'='*70}")


def _print_confusion_matrix(cm: np.ndarray, label_names: list[str], title: str):
    """Print a formatted confusion matrix."""
    print(f"\n{title}")
    print("-" * 60)
    # Header
    print(f"{'':>25}", end="")
    for label in label_names:
        print(f"{label[:8]:>10}", end="")
    print()
    # Rows
    for i, label in enumerate(label_names):
        print(f"{label[:25]:>25}", end="")
        for j in range(len(label_names)):
            print(f"{cm[i, j]:>10.1f}", end="")
        print()


def _retrain_on_all_data(
    backbone_key: str,
    dataset_dir: Path,
    label_names: list[str],
) -> LogisticRegression:
    """Retrain the classifier on ALL cleaned+deduped data."""
    print(f"\n{'='*60}")
    print(f"Retraining on ALL data with {backbone_key}...")
    print(f"{'='*60}")

    # Load all cleaned data
    combined = load_and_clean_all_folds(dataset_dir)
    # Deduplicate (same logic as cv_splits uses internally)
    from src.data import _deduplicate_descriptions
    deduped = _deduplicate_descriptions(combined)

    print(f"Total unique descriptions: {len(deduped)}")
    print(f"Label distribution:\n{deduped['label'].value_counts().to_dict()}")

    embedder = Embedder(backbone_key)
    X = _get_embeddings_cached(embedder, deduped["Description"].tolist())
    y = deduped["label"].tolist()

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(X, y)
    print("Retraining complete.")
    return clf


def _save_artifacts(backbone_key: str, clf: LogisticRegression, label_names: list[str]):
    """Save head.joblib and config.json."""
    # Save head
    joblib.dump(clf, HEAD_PATH)
    print(f"Saved head to {HEAD_PATH}")

    # Save config
    config = {
        "backbone": backbone_key,
        "model_name": MODEL_CONFIGS[backbone_key]["model_name"],
        "embedding_dim": MODEL_CONFIGS[backbone_key]["dim"],
        "labels": label_names,
        "num_classes": len(label_names),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {CONFIG_PATH}")


def _per_class_f1_from_cm(cm: np.ndarray, label_names: list[str]) -> dict[str, float]:
    """Compute per-class F1 from confusion matrix."""
    per_class = {}
    for i, label in enumerate(label_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[label] = f1
    return per_class


def run_training(dataset_dir: Path = Path("../dataset")) -> dict[str, Any]:
    """
    Main training pipeline.

    Args:
        dataset_dir: Path to the dataset directory containing the 3 fold xlsx files.

    Returns:
        Dict with training results and final model info.
    """
    _ensure_dirs()
    overall_start = time.time()

    label_names = get_label_names()
    print(f"Label taxonomy ({len(label_names)} classes): {label_names}")

    # Run CV for each backbone
    results = []
    for backbone_key in MODEL_CONFIGS.keys():
        result = _run_cv_for_backbone(backbone_key, dataset_dir, label_names)
        results.append(result)

    # Print comparison
    _print_comparison_table(results)

    # Pick best backbone by mean macro-F1
    best = max(results, key=lambda r: r["mean_macro_f1"])
    best_key = best["backbone"]
    print(f"\n>>> SELECTED BACKBONE: {best_key} (mean macro-F1 = {best['mean_macro_f1']:.4f}) <<<")

    # Print per-class F1 from best backbone's avg confusion matrix
    per_class_f1 = _per_class_f1_from_cm(best["avg_confusion_matrix"], label_names)
    print(f"\nPer-class F1 (from avg confusion matrix of best backbone):")
    for label, f1 in sorted(per_class_f1.items(), key=lambda x: x[1]):
        print(f"  {label:<25} {f1:.4f}")

    # Retrain on all data with best backbone
    final_clf = _retrain_on_all_data(best_key, dataset_dir, label_names)

    # Save artifacts
    _save_artifacts(best_key, final_clf, label_names)

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE in {total_time:.1f}s")
    print(f"{'='*60}")
    print(f"Chosen backbone: {best_key}")
    print(f"Mean CV accuracy: {best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}")
    print(f"Mean CV macro-F1: {best['mean_macro_f1']:.4f} ± {best['std_macro_f1']:.4f}")
    print(f"Head saved to: {HEAD_PATH}")
    print(f"Config saved to: {CONFIG_PATH}")

    return {
        "chosen_backbone": best_key,
        "cv_results": results,
        "best_result": best,
        "per_class_f1": per_class_f1,
        "total_time": total_time,
        "head_path": str(HEAD_PATH),
        "config_path": str(CONFIG_PATH),
    }


if __name__ == "__main__":
    run_training()