# Project Context & Agent Handover Log

> **Note for AI Agents / LLMs**: This file serves as the central context and progress tracker for this repository. Whenever you work on this project, read this file first to understand the context, status, rules, and architectural decisions. Update this file before completing your turn.

---

## 1. Project Overview & Course Rules
- **Course**: B.E. Computer Science & Engineering (5th Semester, 2026-27)
- **Subject**: Software Engineering (UCS503) — Lab Project
- **Instructor / Lab Evaluator**: Mr. Sukhpal Singh (Research Scholar)
- **Repository Path**: `/home/a87k/Projects/SE_Lab_Project`
- **Team Size**: 4 Students (Member 1: User / Idea Owner)

### Strict Evaluation Rules (From Course PPTX `UCS503_Lab_01_2026-27.pptx`):
1. **No Thin Apps**: Idea will **NOT** be approved if it is a renamed copy, a thin CRUD app, or an AI wrapper without proper evaluation.
2. **Real Engineering Work**: Must show a real user problem, substantial engineering work (components, roles, states, integrations, deployment), quality requirements (security, reliability, performance, privacy, offline sync), and empirical testing.
3. **Team Comprehension**: Every team member must understand, trace, debug, and explain the code, architecture, control flow, state, and failure paths.
4. **AI-Use Logging**: Must maintain an **AI-Use Log** recording the task, tool used, output, changes made, and verification steps.

---

## 2. Primary Project: Pharmacon (APPROVED — WRITER-ADAPTIVE PROTOCOL)

> **IMPORTANT — SCOPE HISTORY (do not revert):**
> 1. Original idea (constraint scheduler + OpenFDA collision engine) → rejected.
> 2. First revision (doctor-adaptive recognition, broad) → conditionally approved.
> 3. **CURRENT (FINAL)**: Mr. Sukhpal Singh approved the project in this exact form via email + `Pharmacon_Writer_Calibration_Protocol.pdf`. The protocol is the authoritative spec. Do NOT revert to earlier scopes or the old `calibration-sheets.txt`.

- **Title**: Pharmacon — Writer-Adaptive Prescription OCR (India pilot: English & Roman-script)
- **Core contribution (NOT a generic scanner/dashboard)**: a system that learns a clinician's connected handwriting from a small controlled calibration set, proves improvement on unseen lines, then converts confirmed results into inventory and patient workflows.

### The precise research question (from the protocol):
> Can sixteen writer-specific calibration lines, followed by four actively-selected refinement lines, reduce character and field-recognition error on new English/Roman-script prescriptions from the same Indian clinician, without memorizing a fixed list of medicines?

### The demonstration must contain (6 stages):
1. **Fixed-form capture** — photograph the printed form, locate the writing boxes, crop each line consistently.
2. **Generic recognition** — run `trocr-small-handwritten` with no writer training (baseline).
3. **Writer adaptation** — LoRA on a small part of the visual encoder, trained only on that writer's 16 enrollment lines; decoder frozen.
4. **Active refinement** — use alignment errors (confusion matrix) to select 4 more lines covering the writer's weak letters/confusion pairs.
5. **Structured extraction** — extract medicine, strength, form, dose, frequency, route, duration, instructions.
6. **Human-safe action** — show top-3 formulary matches + confidence; require confirmation before inventory/patient changes.

### SAFETY BOUNDARY (non-negotiable):
- Decision support, NOT autonomous dispensing.
- Low-confidence word/strength/dosage MUST be highlighted and withheld until a clinician/pharmacist confirms.
- NEVER use an LLM to silently rewrite an uncertain medicine name.

---

## 3. The Protocol (authoritative spec)

Full detail in `Pharmacon_Writer_Calibration_Protocol.pdf` (13 pages). Summary below.

### Data split per writer (minimum):
| Split | Lines | Used for | Never for |
| :--- | :--- | :--- | :--- |
| Enrollment (E01–E16) | 16 | adaptation + confusion analysis | final performance |
| Active refinement (R01–R04) | 4 | 2nd adaptation step after error analysis | choosing final test cases |
| Validation (V01–V04) | 4 | early stopping + confidence thresholds | training/prompt selection |
| Held-out test (T01–T08) | 8 | final generic vs adapted comparison | any training/tuning |

- First presentation: **at least 3 writers**. (Team is currently targeting 1 to get unstuck; final needs 3+.)
- Final project target: 8–12 consenting clinical writers if access permits.
- Report per-writer AND macro-average. Never mix a writer's lines across training and test.

### Capture rules:
1. Print at 100% scale on plain A4. No shrink/fit.
2. Printed prompt stays ABOVE the blank box, outside the crop. Store prompt as ground truth.
3. Writer writes naturally (normal speed, normal pen). No block capitals unless natural.
4. Capture full page AND cropped line. Keep ruler/page corners visible in full photo.
5. One normal-lit image per page for the controlled comparison (angle/low-light = separate test).
6. Preserve punctuation, decimals, slashes, hyphens, units, casing exactly.
7. Separate sessions for enrollment vs test "when possible" (team may skip; note as limitation).

### Models to implement and compare (from protocol):
| Model | Role | Notes |
| :--- | :--- | :--- |
| `microsoft/trocr-small-handwritten` (62M) | **Primary writer-adaptation model** | LoRA rank 8 on query/value of last two visual-encoder attention blocks, decoder frozen. Fallback: fine-tune final encoder block only. |
| PP-OCRv5_mobile_rec (16 MB) | Deployment baseline | Low-resource, on-device/clinic-edge |
| PP-OCRv5_server_rec (81 MB) | Stronger PaddleOCR comparator | Check if mobile loses info |
| CRNN + BiLSTM + CTC | Student-designed baseline | Via PaddleOCR or MMOCR; explainable/modifiable |
| `microsoft/trocr-base-handwritten` (334M) | Accuracy reference only | Only if GPU permits |
| Tesseract | Weak printed-text baseline only | Not core |

**Do NOT use as final system**: hosted OCR APIs, ChatGPT, Gemini, Qwen-VL, other VLMs. They may be shown as comparison only.

### Adaptation recipe (from protocol):
- One adapter per writer. Train only on E01–E16.
- LoRA rank 8, query+value projections, last two visual-encoder attention blocks. Decoder frozen.
- Early stopping on V01–V04. 20–40 epochs, LR 1e-4 (LoRA), batch 4, strong augmentation (rotation, perspective, brightness, blur, scale) on enrollment crops only.
- Save writer-independent base model + tiny per-writer adapter. Load adapter at sign-in.

### Confusion matrix + active refinement (WSCMC):
- Levenshtein-align baseline predictions vs ground truth.
- Count substitution/insertion/deletion. Normalize rows to confusion probabilities.
- Retain confusions seen ≥2x (else "provisional").
- Select 4 refinement prompts greedily by weighted coverage:
  - 0.45 weak-character coverage
  - 0.35 confusion-pair coverage (m/n, u/v, 0/O, 1/l)
  - 0.20 medical-token coverage (uncovered forms/units/frequencies/routes/number formats)
- **Naming**: call this component **Writer-Specific Confusion-Matrix Calibration (WSCMC)**. Do NOT call it "R22-CM" (not a real published model).

### Evaluation conditions (all on sealed T01–T08):
- A. Generic (no writer samples)
- B. Adapted (E01–E16)
- C. Adapted + active (E01–E16 + 4 selected R lines)

### Metrics to report:
- OCR: CER, WER (per-writer + macro-average)
- Structured fields: exact match for medicine, strength, form, frequency, route, duration
- Formulary: top-1/top-3 medicine match (split seen vs unseen in enrollment)
- Safety: coverage at threshold, abstention rate, unsafe false confirmations
- Efficiency: calibration minutes, adapter train time, adapter size, inference latency

### Medicine resolution order (versioned local formulary):
1. Partner clinic/pharmacy inventory
2. NLEM 2022
3. PMBJP product list
4. National Formulary of India 2021

- Rank top-3 by OCR confidence, edit distance, strength compatibility, dosage-form compatibility.
- Ambiguous medicine/strength/route → abstain, ask for confirmation.
- After confirmation: ONE action updates patient dashboard + inventory transaction + reminders + audit trail.

### GO / NO-GO rule (from protocol):
> Proceed to full product IF writer adaptation gives consistent held-out improvement for most writers AND the system can abstain on uncertain medicine/dosage. If it improves only memorized medicines or only the enrollment sheet → redesign adaptation before building the dashboard.

---

## 4. Roadmap & Current Status

### Current phase: build the AI PoC pipeline (before the filled sheet arrives)

**What to build NOW (no sheet needed):**
1. Fixed-form crop script (detect printed boxes from the protocol PDF).
2. TrOCR-small baseline + LoRA adaptation code.
3. CER/WER evaluation script.
4. Confusion-matrix + refinement-selection code.

**Then (when 1 writer's sheet is filled):**
- Run generic vs adapted vs adapted+refinement on sealed T01–T08.
- Report the numbers.

**Then (before presentation):**
- Get 2 more writers (3 total minimum).

**Blocked on:** 1 filled calibration sheet (in progress — user is getting 1 new writer).

### Status checklist:
- [x] Protocol received and understood (email + PDF).
- [x] Old scope/docs preserved as history (calibration-sheets.txt, Florence-2 notebook, finetune_feasibility.py).
- [x] Build fixed-form crop script (box detection) — `pharmacon_ai/` package, tested.
- [x] Build TrOCR-small + LoRA adaptation script (smoke test passed: 24,576 params, 0 decoder params).
- [x] Build CER/WER evaluation script (`pharmacon_ai/evaluate.py`).
- [x] Get 1 writer's filled sheet (Writer W01 photographed and cropped).
- [x] Run generic vs adapted on sealed lines (Writer W01: Macro CER dropped 1.2898 → 0.8050, -48.47% error reduction).
- [ ] Active refinement (WSCMC) on R01–R04.
- [ ] Recruit 2 more writers (total 3).

### Prior work (history — kept for GitHub trail, NOT current approach):
- `calibration-sheets.txt` — old self-made sheet, superseded by sir's protocol PDF.
- `Florence-2(Amitesh's).ipynb` + `finetune_feasibility.py` — Florence-2 experiments. Findings: cross-writer 4.49% exact match (useless); full-page OCR found 0 regions; per-writer fine-tune loss dropped 7.98→3.44 (learns) but held-out eval never completed. Florence-2 is now REPLACED by TrOCR-small per sir's protocol.

---

## 5. Software Engineering Deliverables & Milestones

| Deliverable | Description | Status |
| :--- | :--- | :--- |
| Lab 2: Software Grid | 4 ideas, ranked, 7-section breakdown | Done |
| Commitment pitch | Revised scope approval | Done (approved) |
| Planning Presentation | Live website + Planning v1 + PoC evidence | In progress |
| PoC | Generic vs adapted on sealed lines (Writer W01 complete) | Passed (W01) |
| SRS Document | IEEE 830 | Phase 2 |
| System Design (UML) | Use Case, Class, Sequence, Component, DFD | Phase 3 |
| Working Application | Full codebase + tests | Phase 4 |
| Final Lab Report | Report + deck | End of semester |

---

## 6. Team Website Architectural Requirements
- Public frontend + dedicated pages per deliverable version (old versions stay online).
- Admin portal & API (login, drag-drop upload, metadata, publishing, permissions).
- File storage for decks/PDFs/images.
- Audit & version history (what/when/who + git commit).

---

## 7. Conversation Transcript & Log Location
- **Compact**: `/home/a87k/.gemini/antigravity-ide/brain/48117ca1-f875-4d81-b6f5-2ab35903b1f3/.system_generated/logs/transcript.jsonl`
- **Full**: `/home/a87k/.gemini/antigravity-ide/brain/48117ca1-f875-4d81-b6f5-2ab35903b1f3/.system_generated/logs/transcript_full.jsonl`

---

## 8. AI-Use Log (Course Compliance)

| Date | Task | Tool / Model | Output & Decisions | Verification |
| :--- | :--- | :--- | :--- | :--- |
| 2026-08-09 | Analyzed syllabus PPTX | Antigravity AI (Gemini) | Extracted rules, Software Grid sections, website specs | Slides 1–25 |
| 2026-08-09 | Defined "Pharmacon" | Antigravity AI (Claude + Gemini) | Pivoted to scheduler idea | Slide 19 check |
| 2026-08-09/10 | Analyzed revised scope email; tested handwriting OCR feasibility | Command Code + Antigravity | Doctor-specific calibration as tractable path | Fact-checked TrOCR/LoRA literature |
| 2026-08-10 | Finalized roadmap + AGENTS.md | Command Code | 6-phase roadmap, hard gates, tech stack | Cross-verified |
| 2026-08-27 | Florence-2 + LoRA feasibility test; crop pipeline stress-test | Command Code | CV crop wrong path; Florence-2 raw reads 0 regions; wrote finetune_feasibility.py | Loss 7.98→3.44; held-out blocked |
| 2026-09-13 | Read + decoded final protocol (email + PDF) | Command Code | Updated AGENTS.md/ROADMAP.md to TrOCR-small protocol, WSCMC, 16/4/4/8 split | Cross-checked email vs PDF |
| 2026-09-13/14 | TrOCR-small LoRA smoke test & Colab pipeline | Antigravity AI (Gemini) | Verified model load (bypassing TokenizersBackend & torchao conflicts); proved exact 24,576 trainable params (0.0399%) on blocks 10/11 query+value, decoder frozen | Output log: 8 tensors, 0 decoder params |
| 2026-09-15 | End-to-end empirical PoC on Writer W01 sheets | Antigravity AI (Gemini) | Executed complete pipeline: fixed-form crop with pre-cropped fallback, baseline inference, LoRA adaptation (E01-E16) with V01-V04 early stopping, beam search decoding. Proved Macro CER drop 1.2898 → 0.8050 (-48.47% absolute) on sealed T01-T08 lines. | `PHARMACON_POC_REPORT.md` + Colab run logs |


---

## 9. Guidelines for AI Agents Updating This File
1. Read AGENTS.md first.
2. Update AI-Use Log (Section 8) when AI is used for ideas/code/tests/docs.
3. Never remove the evaluator's rules.
4. Never revert scope: the writer-adaptive protocol (TrOCR-small + WSCMC) is final unless the evaluator changes it.
5. Do not reintroduce the scheduler idea, Florence-2, or the old calibration-sheets.txt as current approach.
