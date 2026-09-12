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
- [ ] `src/embed.py` - multilingual embedding backbone wrapper (benchmark MiniLM-L12-v2 vs multilingual-e5-small)
- [ ] `src/train.py` - train classifier head, 3-fold CV, report accuracy/macro-F1/confusion matrix
- [ ] `src/export_coreml.py` - export backbone+head to `.mlpackage`, verify round-trip parity, report file size
- [ ] `src/predict.py` - CLI: `load_boq_items()` -> embed -> head -> `predictions.json`
- [ ] Fill in `NOTES.md` (approach, decisions, results, limitations)