# BOQ Trade-Package Classifier

## Original Challenge

Assign Bill of Quantities (BOQ) line items from construction Excel files to
the correct trade package (Groundworks, Concrete, Electrical, etc.), so
procurement teams don't have to tag thousands of line items by hand. Output
`predictions.json` with `file`, `sheet`, `row`, `item`, `predicted_tag` for
each work item. Original constraints: ~4 hours, Python, OpenAI API key
provided (GPT-4o-mini/GPT-4o).

## What Was Actually Built

The original held-out BOQ files were never available (the Google Drive link
was dead), and the goal was extended: build this **without any API key**, as
a **memory-constrained, offline-capable** classifier suitable for an eventual
on-device (iOS / Core ML) deployment, not a cloud LLM call per line item.

In place of the missing files, training uses a substitute public dataset -
the [Portuguese Construction Dataset for AI](https://ec-3.org/publications/conference/paper/?id=EC32025_454)
(`../dataset/`, 3 fold files, ~70k labeled BOQ line items, Portuguese). Its
label taxonomy covers 12 structural/shell-and-core trades (no
electrical/plumbing/painting/roofing - this dataset has no ground truth for
those).

**Approach:** a frozen multilingual sentence-embedding backbone
(`multilingual-e5-small`, picked over `MiniLM-L12-v2` via 3-fold CV) feeds a
small class-weighted logistic regression head. Frozen embeddings mean the
same weights classify Portuguese or English text without a translation
step, and the head is cheap enough to retrain on a laptop CPU.

**Results:** mean 3-fold CV macro-F1 0.74, accuracy 0.81 (12 classes, ~9x
class imbalance). See `NOTES.md` for the full breakdown and error analysis.

**Core ML export:** the classifier head converts cleanly to a real
`.mlpackage` (20KB). The embedding backbone does not - `coremltools` hits a
tracing error on BERT-family `position_ids` handling, confirmed independent
of torch version. Documented as a Phase 1 limitation; revisiting it is
Phase 2, once real Mac/Xcode access exists for the Swift/iOS side.

## Repo Layout

```
src/
  data.py           # load fold xlsx -> clean/dedupe -> group-aware 3-fold CV splits
  embed.py          # multilingual sentence-embedding wrapper (MiniLM / e5)
  train.py          # 3-fold CV over both backbones, trains + saves the head
  export_coreml.py  # exports classifier head to Core ML; embedder export (partial, see above)
  predict.py        # CLI: BOQ excel -> predictions.json
models/
  head.joblib, config.json          # trained classifier + chosen backbone
  classifier_head.mlpackage         # Core ML export of the head
docs/superpowers/specs/             # full design spec and rationale
```

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Running

```bash
python -m src.train                       # re-run CV + retrain the head
python -m src.export_coreml                # re-export Core ML artifacts
python -m src.predict data/boq_files/*.xlsx  # -> predictions.json
```

## Known Limitations

- No ground-truth English BOQ file exists to validate against - real-world
  accuracy on the original take-home's target files is unverified.
- Label taxonomy is narrower than a full trade universe (no MEP/finishes).
- Embedding backbone isn't yet in Core ML format (see above).

See `NOTES.md` for the full write-up and `TODO.md` for task-by-task status.
