## Primary TODOs to complete the task:

- [x] Explore the repo
- [x] Understand the objective

- [x] Drive link is unavailable, look for sample datasets.
- [x] Since we have downloaded the portuguese version of the dataset, we need to look at ways of translating it or a method that understands it - can we develop a multi-lingual system? -> Decided: frozen multilingual sentence-embedding backbone, no translation step. See `docs/superpowers/specs/2026-09-12-boq-coreml-pipeline-design.md`.

## Phase 1: Python + Core ML pipeline (Swift/iOS app deferred)

- [x] Analyze dataset structure, label taxonomy, class balance, dedup/leakage risk
- [x] Write and approve design spec
- [x] `src/data.py` - load 3 fold files, clean/filter to single-macro rows, dedupe, group-aware 3-fold split
- [x] `tests/test_data.py` - unit tests for cleaning/dedup/split correctness (17 tests; caught+fixed a real bug: an anchored regex was dropping ~2/3 of legitimate rows starting with punctuation)
- [x] `src/embed.py` - multilingual embedding backbone wrapper (MiniLM-L12-v2 + multilingual-e5-small, both selectable). Verified cross-lingual property holds on real BOQ phrases (PT/EN same-meaning cosine sim 0.78-0.98 vs 0.25-0.36 for unrelated).
- [x] `src/train.py` - train classifier head, 3-fold CV, report accuracy/macro-F1/confusion matrix. Result: e5 beats MiniLM (macro-F1 0.7387 vs 0.7216); per-class F1 ranges 0.90 (Earthworks) down to 0.38 (Structural Reinforcement)/0.50 (Masonry) - tracks the rarer classes, as expected. Caught+fixed a real bug: `multi_class="multinomial"` is no longer a valid LogisticRegression param in scikit-learn 1.9.1, crashed the retrain step.
- [x] `src/export_coreml.py` - export backbone+head to `.mlpackage`, verify round-trip parity, report file size. Partial result: classifier head exports cleanly to real Core ML (0.02 MB). The e5 transformer backbone does NOT convert - coremltools 9.0 hits a tracing error on BERT-family position_ids handling regardless of torch version (tried both 2.14 and 2.7.0, same failure) - documented as a Phase 1 limitation for the next (Mac-based) phase to revisit. `.mlpackage` files also can't be *executed* at all on this Linux box (missing native `libcoremlpython`, macOS-only) - true on-device verification needs the eventual Mac regardless.
- [ ] `src/predict.py` - CLI: `load_boq_items()` -> embed -> head -> `predictions.json`
- [ ] Fill in `NOTES.md` (approach, decisions, results, limitations)