# BOQ Trade-Package Classifier — Core ML Pipeline Design

Date: 2026-09-12
Status: Approved (Phase 1 — Python/Core ML pipeline only; Swift/iOS app deferred to a later phase)

## Context

The take-home (`README.md`) asks for a classifier that assigns BOQ line
items to trade packages, graded against held-out English Excel files.
The Google Drive link for those files is dead, so no ground-truth BOQ
files (English or otherwise) are available.

The user's actual goal extends beyond the take-home: build this as an
on-device, memory-constrained system (target: iOS + Core ML) with **no
API key / no cloud dependency at inference time**. The Swift/iOS app
itself is out of scope for this phase — this machine has no Swift/Xcode
toolchain (verified: `swift`, `swiftc`, `xcodebuild` all absent), and
the user wants to build the app on a Mac later, hands-on. This phase
covers only the Python training pipeline and Core ML export/verification.

The only labeled data available is a substitute dataset: the
[Portuguese Construction Dataset for AI](https://ec-3.org/publications/conference/paper/?id=EC32025_454)
(`dataset/Fold_1_Consolidated_Data.xlsx`, `Consolidated_Fold_2_Data.xlsx`,
`Consolidated_Fold_3_Data.xlsx`), a multilabel BOQ classification dataset
built from Portuguese public procurement contracts with GPT-4o-mini
synthetic augmentation, per its EC3 2025 paper abstract.

## Data Analysis Findings

Verified by direct inspection (not assumed):

- Each fold file: one sheet, ~23,300–23,650 rows, 146 columns.
  Columns 0–3 are `Art.` (item code), `Description`, `Unit`, and an
  always-empty `Unnamed: 3` (drop it). Columns 4–145 are a hierarchical
  multi-hot label matrix: 12 ALL-CAPS **macro** category columns, each
  followed by numbered subcategory/leaf columns.
- The 12 macro categories are the only trade-package-shaped taxonomy in
  this data, and they cover **shell-and-core structural work only**:
  site setup, demolition, earthworks, retaining structures, foundations,
  ground floor slab, RC structure, steel structure, timber structure,
  masonry, misc/other, structural reinforcement. There is no
  electrical/plumbing/mechanical/painting/roofing/finishes category
  anywhere in this dataset, unlike the README's example tags
  (Groundworks, Mechanical, Painting, etc.).
- Two columns, `Throwaway` and `Throwaway - Item`, are excluded from the
  taxonomy: unlike every other column they are not descriptive category
  names, and their flag pattern doesn't correlate consistently with any
  other row property (unit presence, macro category, etc.). Treated as
  artifacts of the paper's synthetic-augmentation/labeling pipeline, not
  real trades.
- Label cardinality per row (over the 12 macro columns), fold 1:
  0 macros flagged: 75 rows (0.3%) — junk/header rows (e.g. a bare
  category name as the "description", no unit). 1 macro: 23,031 rows
  (98.7%) — clean single-label signal. 2+ macros: 239 rows (1.0%) —
  ambiguous recap/summary lines spanning multiple structural systems.
- Class balance (fold 1, single-macro rows): ranges from 427 (masonry)
  to 3,943 (steel structure) — ~9x imbalance. Needs class weighting,
  not resampling (dataset is already synthetically augmented).
- Cross-file duplicate `Description` text: ~1,720–1,775 exact matches
  between any two of the three files (~7-8% of each file's rows). A
  random split across combined folds would leak train/test signal.

## Label Taxonomy (final)

The 12 macro categories become the classifier's output classes,
English-labeled for readability in code/predictions but trained purely
on the embedding of the (Portuguese) description text:

| Portuguese column | English trade label |
|---|---|
| TRABALHOS PREPARATÓRIOS E MONTAGEM DE ESTALEIRO | Site Setup |
| DEMOLIÇÕES E CONTENÇÃO DE FACHADA | Demolition |
| MOVIMENTO DE TERRAS | Earthworks |
| CONTENÇÕES | Retaining Structures |
| FUNDAÇÕES | Foundations |
| PAVIMENTO TERREO | Ground Floor Slab |
| ESTRUTURA BETÃO ARMADO | RC Structure |
| ESTRUTURA METÁLICA | Steel Structure |
| ESTRUTURA DE MADEIRA | Timber Structure |
| ESTRUTURAS DE ALVENARIA | Masonry |
| DIVERSOS | Misc / Other |
| REFORÇO DE ELEMENTOS ESTRUTURAIS EXISTENTES | Structural Reinforcement |

This label set is narrower than the take-home's full trade universe.
That's a known, explicit limitation (see Non-Goals), not an oversight.

## Data Cleaning Rules

1. Drop `Unnamed: 3`, `Throwaway`, `Throwaway - Item` columns entirely.
2. Drop rows with 0 or 2+ macro categories flagged (junk / ambiguous).
3. Keep rows with exactly 1 macro category flagged → `(description, label)`.
4. Deduplicate by exact `Description` text **across all three files
   combined** before splitting: each unique description is one group,
   assigned entirely to one fold-role in a given CV rotation. This
   prevents a duplicate of a train-fold description from leaking into
   the test fold.
5. Strip whitespace; drop empty/near-empty descriptions (e.g. pure
   punctuation or <3 characters) if any survive.

## Model Architecture

**Backbone:** a frozen, pretrained multilingual sentence-embedding
model — not fine-tuned. Two candidates benchmarked head-to-head on
validation macro-F1, final choice picked by result:
- `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~470MB fp32 /
  ~120MB fp16 / ~30MB int8)
- `multilingual-e5-small` (384-dim, similar size, requires a
  `"query: "` / `"passage: "` prompt prefix convention)

Rationale for freezing: the whole point of the multilingual-embeddings
decision is that the *same* weights handle Portuguese training text and
English (or any other language) inference text without a translation
stage. Fine-tuning the backbone on Portuguese-only descriptions risks
pulling the embedding space away from that shared multilingual geometry
(catastrophic forgetting), which would defeat the purpose. A frozen
backbone also means training is CPU-only and fast (only the head
trains), and is the direct, validated version of the original
`.agent/CLAUDE.md` idea (quantized MiniLM + small head) rather than a
reinvention.

**Head:** a small classifier on top of the frozen 384-dim embeddings —
start with `sklearn.linear_model.LogisticRegression` (class-weighted,
multinomial), 12-way. If validation macro-F1 is materially better with
one hidden layer, add a small MLP (384 → 64 → 12); the head stays tiny
either way (a few thousand to ~25K parameters).

## Training & Evaluation

- Genuine 3-fold cross-validation using the files as-is (they are
  literally named Fold_1/2/3, matching the source paper's own fold
  structure): train on 2 files' cleaned+deduped rows, evaluate on the
  3rd, rotate through all 3 combinations.
- Metrics per rotation and averaged: accuracy, macro-F1 (macro-F1
  matters given the ~9x class imbalance), and a confusion matrix to
  surface systematic confusions (e.g. Foundations vs. RC Structure,
  which share "betão armado" vocabulary).
- No real English/held-out BOQ file exists to validate against — this
  is an explicit, reported limitation, not something this phase can fix.

## Core ML Export & Verification

- Export the frozen backbone + trained head as a single Core ML
  `.mlpackage` via `coremltools`, targeting FP16 (falling back from
  INT8 only if INT8 breaks accuracy materially) — `coremltools`
  conversion runs fine on Linux; no Xcode/macOS is required for this
  step, only for building an iOS app around the resulting file later.
- Verification (no iOS device/Simulator available):
  1. Run the exported `.mlpackage` through `coremltools`' own Python
     prediction API on a held-out sample of descriptions, and confirm
     its predicted labels match the pre-export model closely (allow
     small floating-point drift, not label-level disagreement).
  2. Report final on-disk file size against the <40MB target noted in
     `.agent/CLAUDE.md`.
- Explicitly out of scope for this phase: SwiftUI app, on-device Excel
  parsing (CoreXLSX), actual iPhone/Simulator inference. These resume
  once the user has Mac/Xcode access.

## Repo Layout

All work lands inside `Takehome-AI/` (keeps the README's Python
constraint satisfiable as a byproduct, not the primary goal):

```
Takehome-AI/
  src/
    data.py           # load fold xlsx -> clean/dedupe -> (description, label) rows + group-aware fold split
    embed.py          # multilingual embedding backbone wrapper (load once, embed(list[str]) -> np.ndarray)
    train.py          # trains head, runs 3-fold CV, prints/saves metrics
    export_coreml.py  # backbone+head -> models/boq_classifier.mlpackage, round-trip verification
    predict.py        # CLI: load_boq_items() (existing util) -> embed -> head -> predictions.json
  models/
    boq_classifier.mlpackage
    head.joblib
  NOTES.md            # filled in with approach, decisions, results, limitations per README
```

`predict.py` is kept as a thin CLI producing `predictions.json` in the
README's required format, since it falls out of the pipeline almost for
free and keeps the repo a valid, gradeable submission even though the
Swift app is paused. If it turns out to be dead weight once the rest of
the pipeline exists, it's cheap to drop.

## Testing

- Unit tests for `data.py`: label-extraction correctness on a handful
  of hand-picked rows (0-macro, 1-macro, 2+-macro cases), dedupe/group
  split correctness (no description group split across CV rotation
  train/test), row-count sanity after cleaning.
- `export_coreml.py` round-trip check is itself a verification step
  (see above), run as part of the export script, not a separate test.
- No integration test against real English BOQ files is possible
  (no such files exist yet) — noted as a limitation in `NOTES.md`.

## Non-Goals (this phase)

- Swift/SwiftUI app, CoreXLSX parsing, on-device iOS inference.
- Matching the README's full example trade universe (Electrical,
  Plumbing, Painting, Roofing, etc.) — this dataset has no ground truth
  for those trades, so no attempt is made to guess/hallucinate labels
  for classes we can't validate.
- Machine translation of Portuguese text — deliberately avoided per the
  multilingual-embeddings decision (adds a model/memory cost and an
  unnecessary failure point).
- Fine-tuning the embedding backbone itself (see rationale above).
