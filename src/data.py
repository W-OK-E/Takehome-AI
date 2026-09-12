#!/usr/bin/env python3
"""
Data loading, cleaning, and cross-validation split generation for the
Portuguese BOQ trade-package classifier.

Implements the cleaning rules and label taxonomy from the pipeline design spec.
"""
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


MACRO_COLUMNS_PT = [
    "TRABALHOS PREPARATÓRIOS E MONTAGEM DE ESTALEIRO",
    "DEMOLIÇÕES E CONTENÇÃO DE FACHADA",
    "MOVIMENTO DE TERRAS",
    "CONTENÇÕES",
    "FUNDAÇÕES",
    "PAVIMENTO TERREO",
    "ESTRUTURA BETÃO ARMADO",
    "ESTRUTURA METÁLICA",
    "ESTRUTURA DE MADEIRA",
    "ESTRUTURAS DE ALVENARIA",
    "DIVERSOS",
    "REFORÇO DE ELEMENTOS ESTRUTURAIS EXISTENTES",
]

MACRO_TO_ENGLISH = {
    "TRABALHOS PREPARATÓRIOS E MONTAGEM DE ESTALEIRO": "Site Setup",
    "DEMOLIÇÕES E CONTENÇÃO DE FACHADA": "Demolition",
    "MOVIMENTO DE TERRAS": "Earthworks",
    "CONTENÇÕES": "Retaining Structures",
    "FUNDAÇÕES": "Foundations",
    "PAVIMENTO TERREO": "Ground Floor Slab",
    "ESTRUTURA BETÃO ARMADO": "RC Structure",
    "ESTRUTURA METÁLICA": "Steel Structure",
    "ESTRUTURA DE MADEIRA": "Timber Structure",
    "ESTRUTURAS DE ALVENARIA": "Masonry",
    "DIVERSOS": "Misc / Other",
    "REFORÇO DE ELEMENTOS ESTRUTURAIS EXISTENTES": "Structural Reinforcement",
}

DROP_COLUMNS = {"Unnamed: 3", "Throwaway", "Throwaway - Item"}

FOLD_FILES = [
    "Fold_1_Consolidated_Data.xlsx",
    "Consolidated_Fold_2_Data.xlsx",
    "Consolidated_Fold_3_Data.xlsx",
]


def _load_raw_fold(filepath: Path) -> pd.DataFrame:
    """Load a single fold xlsx file as-is."""
    return pd.read_excel(filepath)


def _normalize_bool_series(s: pd.Series) -> pd.Series:
    """Convert various truthy/falsy representations to boolean."""
    if s.dtype == bool:
        return s
    # Handle string representations
    if s.dtype == object:
        s_lower = s.astype(str).str.strip().str.lower()
        return s_lower.isin({"true", "1", "yes", "y", "t", "1.0"})
    # Handle numeric (int, float)
    return s.astype(float).fillna(0).ne(0)


def _clean_fold(df: pd.DataFrame, fold_idx: int) -> pd.DataFrame:
    """Apply cleaning rules to a single fold DataFrame."""
    df = df.copy()

    # Drop specified columns
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Ensure macro columns exist
    missing_macros = [c for c in MACRO_COLUMNS_PT if c not in df.columns]
    if missing_macros:
        raise ValueError(f"Fold {fold_idx}: missing macro columns: {missing_macros}")

    # Normalize macro columns to boolean
    for col in MACRO_COLUMNS_PT:
        df[col] = _normalize_bool_series(df[col])

    # Count macro flags per row
    macro_flags = df[MACRO_COLUMNS_PT].sum(axis=1)

    # Keep only rows with exactly 1 macro flagged
    single_macro_mask = macro_flags == 1
    df = df[single_macro_mask].copy()

    # Extract the single macro label for each row
    def get_single_macro(row: pd.Series) -> str:
        for col in MACRO_COLUMNS_PT:
            if row[col]:
                return col
        return ""

    df["macro_pt"] = df[MACRO_COLUMNS_PT].apply(get_single_macro, axis=1)
    df["label"] = df["macro_pt"].map(MACRO_TO_ENGLISH)

    # Keep only Description and label (plus fold index for tracking)
    df = df[["Description", "label"]].copy()
    df["fold"] = fold_idx

    # Strip whitespace from Description
    df["Description"] = df["Description"].astype(str).str.strip()

    # Drop empty/near-empty descriptions (<3 chars or only punctuation/whitespace)
    df = df[df["Description"].str.len() >= 3].copy()
    df = df[df["Description"].str.contains(r"[^\s\W]", regex=True)].copy()  # has at least one alphanumeric anywhere

    return df


def load_and_clean_all_folds(dataset_dir: Path) -> pd.DataFrame:
    """
    Load all three fold files, clean each, and combine into a single DataFrame.

    Returns a DataFrame with columns: Description, label, fold
    """
    all_folds = []
    for i, fname in enumerate(FOLD_FILES):
        fpath = dataset_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Fold file not found: {fpath}")
        raw = _load_raw_fold(fpath)
        cleaned = _clean_fold(raw, i)
        all_folds.append(cleaned)

    combined = pd.concat(all_folds, ignore_index=True)
    return combined


def _deduplicate_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by exact Description text across all folds combined.

    For each unique description, keep the first occurrence (by fold order, then row order).
    Returns a DataFrame with one row per unique description, with columns:
    Description, label, fold (the fold of the kept row)
    """
    df = df.copy()
    # Add a temporary column for original row order
    df["_row_order"] = range(len(df))
    # Sort by fold first (so fold 0 rows come first), then by original row order
    df = df.sort_values(["fold", "_row_order"]).reset_index(drop=True)
    # Drop duplicates, keeping first occurrence
    df = df.drop_duplicates(subset=["Description"], keep="first").reset_index(drop=True)
    # Drop the temporary column
    df = df.drop(columns=["_row_order"])
    return df


def cv_splits(
    dataset_dir: Path,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Generate 3-fold cross-validation splits with group-aware deduplication.

    Yields (train_df, val_df, test_df) for each rotation where:
    - test_df contains all unique description groups from one fold file
    - train_df contains all unique description groups from the other two folds
    - val_df is empty (reserved for future use; train/test only for now)

    Deduplication is done across ALL files combined before splitting, so
    a duplicate description never ends up split across train and test.
    """
    combined = load_and_clean_all_folds(dataset_dir)
    # Deduplicate: each unique description appears once, assigned to the fold of its first occurrence
    deduped = _deduplicate_descriptions(combined)

    # For each fold index, hold out that fold's descriptions as test
    for test_fold in range(3):
        test_mask = deduped["fold"] == test_fold
        train_mask = deduped["fold"] != test_fold

        test_df = deduped[test_mask][["Description", "label"]].reset_index(drop=True)
        train_df = deduped[train_mask][["Description", "label"]].reset_index(drop=True)
        val_df = pd.DataFrame(columns=["Description", "label"])

        yield train_df, val_df, test_df


def get_label_names() -> list[str]:
    """Return the 12 English label names in a consistent order."""
    return [MACRO_TO_ENGLISH[col] for col in MACRO_COLUMNS_PT]


if __name__ == "__main__":
    # Quick smoke test when run directly
    dataset_dir = Path(__file__).parent.parent / "dataset"
    if dataset_dir.exists():
        for i, (train, val, test) in enumerate(cv_splits(dataset_dir)):
            print(f"Fold {i}: train={len(train)}, val={len(val)}, test={len(test)}")
            print(f"  Train labels: {train['label'].value_counts().to_dict()}")
            print(f"  Test labels:  {test['label'].value_counts().to_dict()}")
    else:
        print(f"Dataset directory not found: {dataset_dir}")