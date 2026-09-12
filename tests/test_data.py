#!/usr/bin/env python3
"""
Unit tests for src/data.py

Tests cover:
- 0-macro row is dropped
- 1-macro row is kept with correct label
- 2+-macro row is dropped
- Duplicate description group never splits across train/test in any rotation
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    MACRO_COLUMNS_PT,
    MACRO_TO_ENGLISH,
    _clean_fold,
    _deduplicate_descriptions,
    cv_splits,
    get_label_names,
    load_and_clean_all_folds,
)


def _create_test_fold_data(fold_idx: int) -> pd.DataFrame:
    """Create a synthetic fold DataFrame matching the expected structure."""
    n_rows = 20
    # Base descriptions unique to each fold (except for intentional cross-fold duplicates)
    # Row index matches the description index for clarity
    base_descriptions = [
        f"Fold{fold_idx} Zero macro item 0",           # row 0: 0 macros
        f"Fold{fold_idx} Earthworks item 1",           # row 1: 1 macro (Earthworks)
        f"Fold{fold_idx} Foundations item 2",          # row 2: 1 macro (Foundations)
        f"Fold{fold_idx} Two macros item 3",           # row 3: 2 macros (Earthworks + Foundations)
        f"Fold{fold_idx} RC Structure item 4",         # row 4: 1 macro (RC Structure)
        f"Fold{fold_idx} Steel Structure item 5",      # row 5: 1 macro (Steel)
        f"Fold{fold_idx} Two macros item 6",           # row 6: 2 macros (Steel + Timber)
        f"Fold{fold_idx} Masonry item 7",              # row 7: 1 macro (Masonry)
        f"Fold{fold_idx} Site Setup item 8",           # row 8: 1 macro (Site Setup)
        f"Fold{fold_idx} Demolition item 9",           # row 9: 1 macro (Demolition)
        f"Fold{fold_idx} Retaining item 10",           # row 10: 1 macro (Retaining)
        f"Fold{fold_idx} Ground Slab item 11",         # row 11: 1 macro (Ground Floor Slab)
        f"Fold{fold_idx} Timber item 12",              # row 12: 1 macro (Timber)
        f"Fold{fold_idx} Misc item 13",                # row 13: 1 macro (Misc)
        f"Fold{fold_idx} Reinforcement item 14",       # row 14: 1 macro (Reinforcement)
        f"Fold{fold_idx} Zero macro item 15",          # row 15: 0 macros
    ]

    data = {
        "Art.": [f"{fold_idx}.{i}" for i in range(n_rows)],
        "Description": base_descriptions + [""] * 4,
        "Unit": ["m3"] * n_rows,
        "Unnamed: 3": [None] * n_rows,
    }

    # Add macro columns - all 0 initially (using int for Excel compatibility)
    for col in MACRO_COLUMNS_PT:
        data[col] = [0] * n_rows

    # Add Throwaway columns
    data["Throwaway"] = [0] * n_rows
    data["Throwaway - Item"] = [0] * n_rows

    # Set macro flags for specific rows to test different scenarios
    # Row 0: 0 macros (should be dropped) - already all 0
    # Row 1: 1 macro - Earthworks (MOVIMENTO DE TERRAS)
    data["MOVIMENTO DE TERRAS"][1] = 1
    # Row 2: 1 macro - Foundations (FUNDAÇÕES)
    data["FUNDAÇÕES"][2] = 1
    # Row 3: 2+ macros - Earthworks + Foundations (should be dropped)
    data["MOVIMENTO DE TERRAS"][3] = 1
    data["FUNDAÇÕES"][3] = 1
    # Row 4: 1 macro - RC Structure
    data["ESTRUTURA BETÃO ARMADO"][4] = 1
    # Row 5: 1 macro - Steel Structure
    data["ESTRUTURA METÁLICA"][5] = 1
    # Row 6: 2+ macros - Steel + Timber (should be dropped)
    data["ESTRUTURA METÁLICA"][6] = 1
    data["ESTRUTURA DE MADEIRA"][6] = 1
    # Row 7: 1 macro - Masonry
    data["ESTRUTURAS DE ALVENARIA"][7] = 1
    # Row 8: 1 macro - Site Setup
    data["TRABALHOS PREPARATÓRIOS E MONTAGEM DE ESTALEIRO"][8] = 1
    # Row 9: 1 macro - Demolition
    data["DEMOLIÇÕES E CONTENÇÃO DE FACHADA"][9] = 1
    # Row 10: 1 macro - Retaining Structures
    data["CONTENÇÕES"][10] = 1
    # Row 11: 1 macro - Ground Floor Slab
    data["PAVIMENTO TERREO"][11] = 1
    # Row 12: 1 macro - Timber Structure
    data["ESTRUTURA DE MADEIRA"][12] = 1
    # Row 13: 1 macro - Misc / Other
    data["DIVERSOS"][13] = 1
    # Row 14: 1 macro - Structural Reinforcement
    data["REFORÇO DE ELEMENTOS ESTRUTURAIS EXISTENTES"][14] = 1
    # Row 15: 0 macros (should be dropped) - already all 0
    # Row 16: 1 macro - Earthworks (CROSS-FOLD duplicate with fold 0 row 1)
    data["MOVIMENTO DE TERRAS"][16] = 1
    data["Description"][16] = "CrossFold Earthworks item 1"
    # Row 17: 1 macro - Foundations (CROSS-FOLD duplicate with fold 0 row 2)
    data["FUNDAÇÕES"][17] = 1
    data["Description"][17] = "CrossFold Foundations item 2"
    # Row 18: empty description (should be dropped)
    data["FUNDAÇÕES"][18] = 1
    data["Description"][18] = "  "
    # Row 19: short description (should be dropped)
    data["FUNDAÇÕES"][19] = 1
    data["Description"][19] = "ab"

    return pd.DataFrame(data)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory with three synthetic fold files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i in range(3):
            df = _create_test_fold_data(i)
            fpath = tmpdir / FOLD_FILES[i]
            df.to_excel(fpath, index=False)
        yield tmpdir


FOLD_FILES = [
    "Fold_1_Consolidated_Data.xlsx",
    "Consolidated_Fold_2_Data.xlsx",
    "Consolidated_Fold_3_Data.xlsx",
]


class TestCleanFold:
    """Tests for the _clean_fold function."""

    def test_zero_macro_dropped(self, temp_dataset_dir):
        """Rows with 0 macro categories flagged should be dropped."""
        fold_path = temp_dataset_dir / FOLD_FILES[0]
        raw = pd.read_excel(fold_path)
        cleaned = _clean_fold(raw, 0)

        # Row 0 and 15 have 0 macros, row 18 has empty desc, row 19 has short desc
        # Rows 1,2,4,5,7,8,9,10,11,12,13,14,16,17 = 14 rows with 1 macro
        # But rows 18,19 dropped for empty/short desc
        # Row 0, 15 dropped for 0 macros
        # Row 3, 6 dropped for 2+ macros
        # So 20 - 2 (0 macro) - 2 (2+ macro) - 2 (empty/short) = 14
        assert len(cleaned) == 14

    def test_one_macro_kept_with_correct_label(self, temp_dataset_dir):
        """Rows with exactly 1 macro should be kept with the correct English label."""
        fold_path = temp_dataset_dir / FOLD_FILES[0]
        raw = pd.read_excel(fold_path)
        cleaned = _clean_fold(raw, 0)

        # Check row 1 (Earthworks)
        earthworks_row = cleaned[cleaned["Description"] == "Fold0 Earthworks item 1"]
        assert len(earthworks_row) == 1
        assert earthworks_row.iloc[0]["label"] == "Earthworks"

        # Check row 2 (Foundations)
        foundations_row = cleaned[cleaned["Description"] == "Fold0 Foundations item 2"]
        assert len(foundations_row) == 1
        assert foundations_row.iloc[0]["label"] == "Foundations"

        # Check row 4 (RC Structure)
        rc_row = cleaned[cleaned["Description"] == "Fold0 RC Structure item 4"]
        assert len(rc_row) == 1
        assert rc_row.iloc[0]["label"] == "RC Structure"

    def test_two_plus_macro_dropped(self, temp_dataset_dir):
        """Rows with 2+ macro categories flagged should be dropped."""
        fold_path = temp_dataset_dir / FOLD_FILES[0]
        raw = pd.read_excel(fold_path)
        cleaned = _clean_fold(raw, 0)

        # Row 3 has Earthworks + Foundations (2 macros)
        # Row 6 has Steel + Timber (2 macros)
        # These should not appear in cleaned
        assert "Fold0 Two macros item 3" not in cleaned["Description"].values
        assert "Fold0 Two macros item 6" not in cleaned["Description"].values

    def test_drop_columns_removed(self, temp_dataset_dir):
        """Unnamed: 3, Throwaway, Throwaway - Item columns should be dropped."""
        fold_path = temp_dataset_dir / FOLD_FILES[0]
        raw = pd.read_excel(fold_path)
        cleaned = _clean_fold(raw, 0)

        assert "Unnamed: 3" not in cleaned.columns
        assert "Throwaway" not in cleaned.columns
        assert "Throwaway - Item" not in cleaned.columns

    def test_empty_and_short_descriptions_dropped(self, temp_dataset_dir):
        """Empty/whitespace and very short descriptions should be dropped."""
        fold_path = temp_dataset_dir / FOLD_FILES[0]
        raw = pd.read_excel(fold_path)
        cleaned = _clean_fold(raw, 0)

        assert "  " not in cleaned["Description"].values
        assert "ab" not in cleaned["Description"].values

    def test_description_starts_with_punctuation_kept(self, temp_dataset_dir):
        """Descriptions starting with punctuation but containing alphanumerics should be KEPT."""
        # Create a custom fold with descriptions starting with punctuation
        n_rows = 5
        data = {
            "Art.": [f"0.{i}" for i in range(n_rows)],
            "Description": [
                "- Vigas de fundação",       # starts with hyphen, has content -> KEEP
                "(logotipo) Fornecimento",   # starts with paren, has content -> KEEP
                "# Item especial",           # starts with hash, has content -> KEEP
                "---",                       # pure punctuation -> DROP
                "...",                       # pure punctuation -> DROP
            ],
            "Unit": ["m3"] * n_rows,
            "Unnamed: 3": [None] * n_rows,
        }
        for col in MACRO_COLUMNS_PT:
            data[col] = [0] * n_rows
        data["Throwaway"] = [0] * n_rows
        data["Throwaway - Item"] = [0] * n_rows
        # All have 1 macro (Earthworks)
        data["MOVIMENTO DE TERRAS"] = [1] * n_rows

        df = pd.DataFrame(data)
        cleaned = _clean_fold(df, 0)

        # First 3 should be kept (have alphanumeric content despite starting with punctuation)
        assert "- Vigas de fundação" in cleaned["Description"].values
        assert "(logotipo) Fornecimento" in cleaned["Description"].values
        assert "# Item especial" in cleaned["Description"].values
        # Last 2 should be dropped (pure punctuation)
        assert "---" not in cleaned["Description"].values
        assert "..." not in cleaned["Description"].values


class TestDeduplication:
    """Tests for group-aware deduplication."""

    def test_cross_fold_duplicates_merged(self, temp_dataset_dir):
        """Same description across folds should be merged to one row (from first fold)."""
        combined = load_and_clean_all_folds(temp_dataset_dir)
        deduped = _deduplicate_descriptions(combined)

        # "CrossFold Earthworks item 1" appears in all 3 folds (row 16)
        # After deduplication, only one row should remain (from fold 0)
        desc = "CrossFold Earthworks item 1"
        rows = deduped[deduped["Description"] == desc]
        assert len(rows) == 1
        # Should be from fold 0 (first occurrence)
        assert rows.iloc[0]["fold"] == 0

        # "CrossFold Foundations item 2" appears in all 3 folds (row 17)
        desc2 = "CrossFold Foundations item 2"
        rows2 = deduped[deduped["Description"] == desc2]
        assert len(rows2) == 1
        assert rows2.iloc[0]["fold"] == 0

    def test_unique_descriptions_preserved(self, temp_dataset_dir):
        """Unique descriptions should be preserved with their original fold."""
        combined = load_and_clean_all_folds(temp_dataset_dir)
        deduped = _deduplicate_descriptions(combined)

        # Fold-unique descriptions should all be present
        fold0_unique = [d for d in combined[combined["fold"] == 0]["Description"].unique()
                        if not d.startswith("CrossFold")]
        for desc in fold0_unique:
            rows = deduped[deduped["Description"] == desc]
            assert len(rows) == 1
            assert rows.iloc[0]["fold"] == 0


class TestCVSplits:
    """Tests for cross-validation split generation."""

    def test_no_group_leakage_across_train_test(self, temp_dataset_dir):
        """No description should appear in both train and test in any rotation."""
        for train_df, val_df, test_df in cv_splits(temp_dataset_dir):
            train_descs = set(train_df["Description"])
            test_descs = set(test_df["Description"])
            overlap = train_descs & test_descs
            assert len(overlap) == 0, f"Leakage detected: {overlap}"

    def test_all_descriptions_accounted_for(self, temp_dataset_dir):
        """All unique descriptions should appear in either train or test (not val)."""
        combined = load_and_clean_all_folds(temp_dataset_dir)
        deduped = _deduplicate_descriptions(combined)
        all_descs = set(deduped["Description"])

        for train_df, val_df, test_df in cv_splits(temp_dataset_dir):
            train_descs = set(train_df["Description"])
            test_descs = set(test_df["Description"])
            split_descs = train_descs | test_descs
            assert split_descs == all_descs

    def test_three_rotations_produced(self, temp_dataset_dir):
        """Should yield exactly 3 fold rotations."""
        splits = list(cv_splits(temp_dataset_dir))
        assert len(splits) == 3

    def test_each_fold_is_test_once(self, temp_dataset_dir):
        """Each fold's unique descriptions should be test exactly once across rotations."""
        all_test_descs = []
        for train_df, val_df, test_df in cv_splits(temp_dataset_dir):
            all_test_descs.extend(test_df["Description"].tolist())

        # Each unique description from deduplicated data should appear in test exactly once
        combined = load_and_clean_all_folds(temp_dataset_dir)
        deduped = _deduplicate_descriptions(combined)
        all_unique_descs = set(deduped["Description"])

        from collections import Counter
        desc_counts = Counter(all_test_descs)
        for desc, count in desc_counts.items():
            assert count == 1, f"Description '{desc}' appears in test {count} times"

        assert set(all_test_descs) == all_unique_descs

    def test_val_df_empty(self, temp_dataset_dir):
        """Validation DataFrame should be empty (reserved for future use)."""
        for train_df, val_df, test_df in cv_splits(temp_dataset_dir):
            assert len(val_df) == 0
            assert list(val_df.columns) == ["Description", "label"]


class TestLabelNames:
    """Tests for label name consistency."""

    def test_get_label_names_returns_12_labels(self):
        labels = get_label_names()
        assert len(labels) == 12

    def test_label_names_match_taxonomy(self):
        labels = get_label_names()
        expected = list(MACRO_TO_ENGLISH.values())
        assert labels == expected


class TestLoadAndCleanAllFolds:
    """Integration tests for the full pipeline."""

    def test_combined_shape(self, temp_dataset_dir):
        """Combined DataFrame should have expected columns and rows."""
        combined = load_and_clean_all_folds(temp_dataset_dir)
        assert list(combined.columns) == ["Description", "label", "fold"]
        # 14 rows per fold * 3 folds = 42, but duplicates across folds reduce this
        assert len(combined) > 0

    def test_all_labels_present(self, temp_dataset_dir):
        """All 12 labels should be representable (at least in the taxonomy)."""
        combined = load_and_clean_all_folds(temp_dataset_dir)
        # Our test data only covers some labels, but taxonomy has 12
        assert set(combined["label"].unique()).issubset(set(MACRO_TO_ENGLISH.values()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])