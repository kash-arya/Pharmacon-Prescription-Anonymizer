# Pharmacon — Project Roadmap

## What we are building (in plain words)

Pharmacon reads a doctor's handwritten prescription and turns it into clean, structured digital data. The twist: it learns **each doctor's own handwriting** first, so it reads their prescriptions far more accurately than a generic one-size-fits-all reader.

Here's the full story in one go:

1. A medical practitioner writes a short **calibration sheet** once (characters, abbreviations, common medicines).
2. The system learns their handwriting from those sheets.
3. When that practitioner writes a prescription, staff take **one photo**.
4. The system reads the handwriting and extracts the fields: medicine name, strength, frequency, timing.
5. It shows the photo beside what it read, and **highlights anything it is unsure about**.
6. Staff confirm once. No manual re-typing.
7. The medicine is matched to the pharmacy's **stock**, which updates automatically.
8. The confirmed prescription appears in the **patient's app** with reminders.

### What Pharmacon will NOT do (important)

- No diagnosing conditions
- No suggesting a different or cheaper medicine
- No changing a dosage
- No recommending supplements or new medicines

We only *read* what the practitioner wrote and route it through confirmation. Nothing else.

---

## Why this is hard (and worth doing)

The hard part is not the app. It is **doctor-adaptive handwriting recognition**. Every practitioner writes differently, and general readers fail on handwriting. We solve it by:

- Giving each practitioner a controlled calibration sheet
- Building a model adapter tuned to their specific handwriting
- Measuring how much more accurate it gets as we add more calibration samples

This is the technical core of the project. Everything else wraps around it.

---

## The rules we must not break

These come directly from the evaluator. Breaking any of them means the project is rejected.

1. Recognition must be **doctor-adapted** and improve from corrections. Printed OCR + reminders alone = automatic rejection.
2. **No clinical decisions** anywhere. No substitution, no diagnosis, no dosage change, no recommendation.
3. Human-in-the-loop: the original image is shown beside extracted fields, and **uncertain text is clearly marked**.
4. Inventory is exposed through a **documented API/adapter**, not a closed dashboard.
5. Full **audit trail** for every prediction, correction, and inventory action.
6. Evaluation on **held-out samples from every writer**, reporting per-doctor improvement (not just overall accuracy).

---

## Tech stack (our working plan)

| Area | Choice | Note |
| :--- | :--- | :--- |
| Handwriting model | PaddleOCR (PP-OCRv5/v6) | Primary. Free, open-source, handwriting support, light recognition head (easy per-writer fine-tuning) |
| Alternative model | Microsoft TrOCR | Compare as a fallback for pure handwriting accuracy |
| Efficient tuning | Fine-tune recognition head / LoRA | PaddleOCR head is small enough to fine-tune directly; LoRA only if needed |
| Compute | Free GPU (Colab / Kaggle) | Fine-tuning won't run on a laptop |
| Accuracy metrics | jiwer | Standard CER/WER library |
| Backend | Python/FastAPI (or Node/Express) | Python favours ML inference |
| Login / auth | Existing library (Auth.js / Passport.js / Supabase) | Never build our own auth |
| Database | PostgreSQL (SQLite for PoC first) | `tenant_id` column for multi-tenancy |
| Frontend | React | B2B platform + patient PWA |
| Deployment | Cloudflare Pages + hosted API | Site must stay live |
| Medicine list | India's NLEM + common brands | Constrained drug dictionary |

---

## The 3 calibration sheets (what practitioners write)

Each practitioner writes three sheets, staged from easy to hard. This also lets us answer "does more data help?"

**Sheet 1 — Characters**
- A–Z (upper and lower), digits 0–9

**Sheet 2 — Medical shorthand**
- `OD`, `BD`, `TDS`, `HS`, `PC`, `AC`
- `Tab`, `Cap`, `Syp`, `Inj`
- `mg`, `ml`, `mcg`, `tsp`
- Dose numbers: `500`, `250`, `5`, `10`, `2.5`, `1-0-1`, `0-1-1`

**Sheet 3 — Full prescription lines**
- 30–40 real lines like:
  - `Tab. Metformin 500mg 1-0-1 PC`
  - `Cap. Amlodipine 5mg 1-0-0 HS`
  - `Tab. Telmisartan 40mg OD AC`

Plus a separate **held-out set** (~20-30 new lines never used for training *or* calibration) to test whether the model actually learned the handwriting, not just memorised the sheets.

---

## The roadmap (6 phases)

### Phase 0 — Get ready and recruit (start now, this is the critical path)

The single biggest risk is not code. It is getting real medical practitioners to write samples.

**Do:**
- Recruit 5–8 medical practitioners (friends' relatives who are practitioners work fine)
- Download public handwritten-prescription data to build and test the **generic baseline**
- Set up the **evaluation harness now** (CER/WER measurement script) so we learn it before real data arrives
- Prepare and **version** the calibration sheet content (from NLEM)
- Print the calibration sheets and collect **paper** samples (paper is what the real system reads)

**Exit:** generic baseline runs end-to-end on downloaded data; CER/WER measured on downloaded data; practitioners committed; first real samples in hand.

---

### Phase 1 — Proof of concept (the evaluator's checkpoint)

Prove that doctor-adapted beats generic on held-out samples.

**Do:**
- Fine-tune PaddleOCR per writer on their calibration sheets
- Extract **medicine name + strength** only (the stated minimum)
- Compare generic vs adapted on held-out **paper** samples, per writer
- Report the learning curve: error rate at 1 sheet, 2 sheets, 3 sheets
- Map to a sample inventory and show the confirmation screen

**Exit:** adaptation clearly beats generic. If it doesn't, we report the measured result honestly and fix the approach.

---

### Phase 2 — Full extraction + correction loop

**Do:**
- Extract all fields: name, strength/form, frequency/route, duration, instructions
- Use confidence scores (from the model's own output)
- Add an abstention rule: anything below a confidence threshold goes to a human, always
- Let staff corrections feed back into the practitioner's adapter (continual learning)

---

### Phase 3 — Inventory + API adapter

**Do:**
- Sample inventory (SKU, brand, generic, strength, form, pack size, stock)
- Map recognised medicines to SKUs (brand ↔ generic resolution)
- Expose a documented REST API: `find`, `reduce`, `restock`
- Update stock after staff confirmation

**Exit:** a separate mock inventory client can call our API successfully.

---

### Phase 4 — Multi-tenant clinic platform (B2B)

**Do:**
- Hosted web app with login and roles (practitioner / staff / admin)
- One scan workflow: upload → extract → review → confirm → inventory update
- `tenant_id` scoping so clinics can't see each other's data
- Append-only audit log

**Exit:** two tenants are fully isolated; every action has an audit row.

---

### Phase 5 — Patient app (B2C)

**Do:**
- Read-only dashboard of confirmed prescriptions and history
- Schedule, reminders, refill status + request, adherence logging
- Clinic contact, optional caregiver alerts (with consent)
- Offline support via IndexedDB cache

**Exit:** a patient logs in and sees only their own confirmed records.

---

### Phase 6 — Evaluation + hardening

**Do:**
- Run the full metric suite on held-out samples:
  - CER, WER
  - exact medicine-name accuracy
  - exact strength/dosage accuracy
  - SKU-mapping accuracy
  - low-confidence abstention rate
  - correction time
  - per-doctor improvement
- Security: role-based access, patient data privacy, consent handling
- Keep an AI-use log and documentation so every member can explain the code

**Final demo (the thing the evaluator actually watches):**
1. Practitioner calibrates (3 paper sheets photographed)
2. Staff photograph a handwritten prescription
3. The system extracts fields and highlights uncertain text
4. Staff correct + confirm once
5. Stock updates via the API
6. Patient sees the confirmed prescription with a reminder

This end-to-end flow, working live, is the definition of done.

---

## The one thing to never lose sight of

The pass/fail hinge is **Phase 0 → Phase 1**: getting enough real, paired writer data to show adaptation beats generic on held-out samples. Internet data helps us build the baseline, but the *paired calibration + held-out structure* must come from real practitioners. Recruiting is the critical path and cannot be sped up by writing more code.
