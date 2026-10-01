# Pharmacon — Roadmap

> **Note**: Plain-language plan for the team. Authoritative spec = `Pharmacon_Writer_Calibration_Protocol.pdf`. Context/handover log = `AGENTS.md`.

---

## What we are building

Pharmacon reads a doctor's handwritten prescription lines and turns them into structured data (medicine, strength, dose, frequency, route, duration). The core contribution is **writer adaptation**: learn one doctor's handwriting from 16 short calibration lines, then read their new prescriptions better than a generic model.

Safety: decision support, not auto-dispensing. Anything uncertain is flagged for a human to confirm.

---

## The research question (what we must prove)

> Can 16 calibration lines + 4 refinement lines reduce recognition error on new lines from the same writer, without memorizing medicines?

---

## The plan (6 stages, in order)

1. **Fixed-form capture** — photograph the printed form, detect boxes, crop each line.
2. **Generic recognition** — run TrOCR-small with no writer training (baseline).
3. **Writer adaptation** — LoRA rank 8 on the visual encoder, trained only on 16 enrollment lines, decoder frozen.
4. **Active refinement** — use the confusion matrix to pick 4 more lines covering weak letters.
5. **Structured extraction** — pull out medicine, strength, form, dose, frequency, route, duration.
6. **Human-safe action** — top-3 formulary matches + confirmation before any inventory/patient change.

---

## Data split per writer

| Split | Lines | Used for |
| :--- | :--- | :--- |
| Enrollment (E01–E16) | 16 | adaptation + confusion analysis |
| Active refinement (R01–R04) | 4 | 2nd adaptation step |
| Validation (V01–V04) | 4 | early stopping + thresholds |
| Held-out test (T01–T08) | 8 | final comparison (sealed) |

**Writer count:** 3 minimum for the presentation. Currently targeting 1 to get unstuck, then 2 more.

---

## Models (from the protocol)

| Model | Role |
| :--- | :--- |
| `trocr-small-handwritten` (62M) | Primary — writer adaptation |
| PP-OCRv5_mobile_rec (16 MB) | Deployment baseline |
| PP-OCRv5_server_rec (81 MB) | Stronger comparator |
| CRNN + BiLSTM + CTC | Student baseline (explainable) |
| `trocr-base-handwritten` (334M) | Accuracy reference (optional) |
| Tesseract | Weak printed-text baseline only |

**Forbidden as core**: hosted OCR APIs, ChatGPT/Gemini/Qwen-VL (comparison only).

---

## Adaptation recipe (exact)

- One adapter per writer, trained on E01–E16 only.
- LoRA rank 8, query+value projections, last two visual-encoder attention blocks, decoder frozen.
- Early stop on V01–V04. 20–40 epochs, LR 1e-4, batch 4, augmentation on enrollment crops only.
- Save base model + tiny per-writer adapter.

---

## Confusion matrix → refinement (WSCMC)

1. Levenshtein-align baseline predictions vs ground truth.
2. Count char substitutions/insertions/deletions → confusion probabilities.
3. Keep confusions seen ≥2×.
4. Pick 4 refinement lines greedily: 0.45 weak-char + 0.35 confusion-pair + 0.20 medical-token coverage.

**Name it Writer-Specific Confusion-Matrix Calibration (WSCMC)**, NOT "R22-CM".

---

## Evaluation (on sealed T01–T08)

| Condition | Training |
| :--- | :--- |
| A. Generic | none |
| B. Adapted | E01–E16 |
| C. Adapted + active | E01–E16 + 4 R lines |

Report: CER, WER (per-writer + macro), field exact match, top-1/top-3 medicine match, abstention rate, unsafe false confirmations, adapter size, train/inference time.

---

## Medicine resolution order

1. Clinic/pharmacy inventory
2. NLEM 2022
3. PMBJP list
4. National Formulary of India 2021

Top-3 ranked by OCR confidence + edit distance + strength/dosage-form compatibility. Ambiguous → abstain. Confirmation → one action updates everything.

---

## GO / NO-GO

Proceed only if adaptation improves held-out accuracy for most writers AND the system abstains safely. Otherwise redesign before building the dashboard.

---

## What to build now (no filled sheet needed)

1. Fixed-form crop script (detect printed boxes).
2. TrOCR-small baseline + LoRA adaptation code.
3. CER/WER evaluation script.
4. Confusion matrix + refinement selection.

Then, when 1 writer's sheet arrives: run A vs B vs C on sealed lines and report numbers.

---

## Immediate next actions

- [x] Build fixed-form crop script (box detection from the protocol PDF).
- [x] Build TrOCR-small + LoRA adaptation script.
- [x] Build CER/WER evaluation script.
- [x] Get 1 writer's filled sheet (Writer W01 evaluated).
- [x] Run generic vs adapted on sealed lines (Macro CER: 1.29 -> 0.80).
- [ ] Run active refinement (WSCMC on R01-R04).
- [ ] Recruit 2 more writers.
