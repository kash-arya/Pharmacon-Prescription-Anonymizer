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

## 2. Primary Project: Pharmacon (REVISED SCOPE — FINAL)

> **IMPORTANT**: The original "constraint-based intake scheduler + OpenFDA/DrugBank collision engine" idea was **rejected/conditionally revised** by Mr. Sukhpal Singh. The project has pivoted to the scope below. Do NOT revert to the old scheduler idea.

- **Title**: Pharmacon — Handwritten Prescription Digitisation & Doctor-Adaptive Recognition
- **Core Problem**: Reliable digitisation of handwritten prescriptions at a clinic/pharmacy, connected to the clinic's inventory system and the patient's personal medicine dashboard.

### The revised scope (from Mr. Sukhpal Singh's email — this is the fixed, authoritative direction):

1. **Doctor-specific handwriting adaptation**
   - Each participating doctor completes a controlled calibration sheet (paper or digital writing interface) with characters, numerals, medical abbreviations, dosage patterns, and formulary medicines.
   - The system builds a doctor-specific handwriting profile/model adapter.
   - The adapter must also learn from later corrections by authorised clinic/pharmacy staff.
   - Must measure how recognition performance changes with different numbers of calibration samples. Do NOT assume one page is enough.

2. **Prescription scanning & structured extraction**
   - A clinic/pharmacy employee photographs/uploads a handwritten prescription.
   - Extract structured fields: medicine name, strength & dosage form, frequency & route, duration & instructions, patient & prescriber identifiers (where available).
   - UI shows the original image beside extracted fields, clearly marking uncertain text.

3. **Formulary & inventory integration**
   - Recognised medicine matched against clinic/pharmacy formulary and actual inventory records.
   - Handle brand/generic names, strength, dosage form, pack size, inventory SKU mapping.
   - Matching is ONLY for interpreting what the doctor wrote. **NO symptom-based recommendation, NO substitution, NO clinical decision.**
   - After authorised staff confirm, create the digital prescription record and reserve/update inventory.
   - Workflow: one photograph → one confirmation screen → complete, no manual re-entry.

4. **B2B clinic & pharmacy platform**
   - Hosted, multi-tenant web platform with: doctor enrolment & calibration, prescription scan queue & confidence review, patient/prescription records, inventory mapping & stock/refill, correction history & continual-learning records, role-based access (doctors/pharmacists/staff), complete audit trail.
   - Inventory component exposed via a documented API/adapter (NOT a closed dashboard).

5. **B2C patient application**
   - Every confirmed prescription appears in the patient's dashboard (mobile app or PWA).
   - Patient side: prescription/medication history, schedule & reminders, refill status & requests, adherence logging, clinic/pharmacy contact, optional caregiver alerts (with consent).
   - **NO diagnosis, NO dosage changes, NO substitutions, NO medicine recommendations.**

6. **Evaluation**
   - Compare: generic handwriting model, generic + formulary constraints, doctor-adapted, doctor-adapted + formulary constraints + correction-based learning.
   - Report: CER, WER, exact medicine-name accuracy, exact strength/dosage accuracy, SKU-mapping accuracy, low-confidence abstention, correction time, per-doctor improvement.
   - Use held-out samples from every participating writer. Overall accuracy alone is insufficient (medicine/dosage errors matter more than ordinary text errors).

### Hard gates (non-negotiable, from the email):
- Recognition must be **doctor-adapted** and improve from corrections. Printed OCR + reminders alone = automatic rejection.
- **No clinical decisions** anywhere.
- Human-in-the-loop confirmation with uncertain text marked and original image shown beside fields.
- Inventory via documented API/adapter, not a closed dashboard.
- Full audit trail for every prediction, correction, and inventory action.
- Evaluation on held-out samples per writer, reporting per-doctor improvement.
- If the team excludes handwritten recognition, cannot obtain appropriate writers/samples, or reduces to printed OCR + reminders → **Pharmacon will not remain approved** (3 new ideas required).

---

## 3. Roadmap

The full plan — what we build, the rules, the tech stack, the 6 phases, the calibration sheets, and the critical risks — lives in [`ROADMAP.md`](./ROADMAP.md). That file is the single source of truth for the roadmap, written in plain language so the whole team can read it. This file (AGENTS.md) holds scope, context, status, and handover only.

Key decisions locked in (details in ROADMAP.md):
- **Model**: PaddleOCR (PP-OCRv5/v6) primary, TrOCR as fallback.
- **Adaptation**: LoRA per-writer adapters, trained on 3+ calibration sheets per doctor.
- **Writer pool**: 5–8 medical practitioners from the team's personal network.
- **Critical path**: Phase 0 → Phase 1 (recruiting real writers and proving adaptation beats generic on held-out samples).

---

## 4. Software Engineering Deliverables & Milestones

| Deliverable | Description / Requirements | Target Date / Status |
| :--- | :--- | :--- |
| **Lab 2: Software Grid** | 4 original project ideas presented & rationally ranked. Detailed 7-section breakdown for Pharmacon. | **Due Next Lab** |
| **Planning Presentation v1** | Live website upload demo + Planning v1 page published + repository deployment setup. | **In 2 Weeks** |
| **PoC (Week 2)** | Generic vs doctor-adapted recognition on held-out handwriting; extract medicine + strength; map to sample inventory; show confirmation workflow. | **Next Lab session** |
| **SRS Document** | Full IEEE 830 compliant Software Requirement Specification. | Phase 2 |
| **System Design (UML)** | Use Case, Class, Sequence, Component, DFD diagrams. | Phase 3 |
| **Working Application** | Complete codebase with unit & integration test suites. | Phase 4 |
| **Final Lab Report** | Complete SE project report and presentation deck. | End of Semester |

---

## 5. Team Website Architectural Requirements
- **Public Frontend**: Team homepage (members, roles, status, links) + dedicated pages for every presentation/deliverable version. Old versions must remain permanently online.
- **Admin Portal & API**: Authenticated login, drag-and-drop file/folder uploader, metadata entry, publishing control, permissions.
- **File Storage**: Decks, PDFs, images, and document storage integrated with frontend.
- **Audit & Version History**: Show *what changed*, *when it changed*, *who published it*, and related git commit/deployment.

---

## 6. Conversation Transcript & Log Location for External AI Agents
For Command Code or other LLMs needing the exact execution logs and raw conversation transcript of this session:
- **Compact Transcript (JSONL)**: `/home/a87k/.gemini/antigravity-ide/brain/48117ca1-f875-4d81-b6f5-2ab35903b1f3/.system_generated/logs/transcript.jsonl`
- **Full Transcript (JSONL)**: `/home/a87k/.gemini/antigravity-ide/brain/48117ca1-f875-4d81-b6f5-2ab35903b1f3/.system_generated/logs/transcript_full.jsonl`

---

## 7. Current Status & Immediate Action Plan

### Current Phase: **Lab 2 Commitment Pitch — Revised Pharmacon Scope**

- [x] Initialized multi-agent handover file (`AGENTS.md`).
- [x] Extracted course guidelines & evaluator info (Mr. Sukhpal Singh).
- [x] Installed skills globally in `~/.agents/skills/`.
- [x] Original idea stress-tested and revised by evaluator.
- [x] Revised scope fully analysed and roadmap finalized.
- [ ] **Present the revised scope + roadmap to Mr. Sukhpal Singh (commitment pitch).**
- [ ] Start Phase 0: recruit 5–8 medical practitioners from the team's personal network.
- [ ] Download public handwritten-prescription data for baseline validation.

---

## 8. AI-Use Log (Course Compliance)

| Date | Task / Prompt Goal | Tool / Model Used | Output & Decisions Made | Verification Method |
| :--- | :--- | :--- | :--- | :--- |
| 2026-08-09 | Analyzed `UCS503_Lab_01_2026-27.pptx` syllabus and lab guidelines | Antigravity AI (Gemini 3.6 Flash) | Extracted slide requirements, mapped 7 Software Grid sections and website specs. | Verified against slides 1–25 of course presentation PPTX. |
| 2026-08-09 | Stress-tested original project idea & defined "Pharmacon" | Antigravity AI (Claude Sonnet 4.6 & Gemini 3.6 Flash) | Pivoted generic reminder app into constraint-based scheduler. | Checked against Slide 19 "No Thin CRUD / No AI Wrapper" rule. |
| 2026-08-09/10 | Analysed evaluator's revised Pharmacon scope email; stress-tested handwriting OCR feasibility | Command Code (DeepSeek) + Antigravity (Claude Sonnet 4.6) | Rejected general handwriting OCR as infeasible; identified doctor-specific calibration as the tractable path; built revised roadmap. | Fact-checked by Claude Sonnet 4.6 against TrOCR/LoRA literature and evaluator email constraints. |
| 2026-08-10 | Finalized roadmap and updated AGENTS.md with revised scope | Command Code (DeepSeek) | Locked 6-phase roadmap with hard gates, tech stack, assumptions, and risk register. | Cross-verified against evaluator email and course PPTX. |

---

## 9. Guidelines for AI Agents Updating This File
1. **Read First**: Always read `AGENTS.md` at the start of any conversation.
2. **Update AI Log**: Add an entry to Section 8 whenever AI assistance is used to generate ideas, code, tests, or documentation.
3. **Preserve Rules**: Never remove the evaluator's strict evaluation rules or formatting guidelines.
4. **Never revert scope**: The revised doctor-adaptive recognition scope is final unless the evaluator explicitly changes it. Do not reintroduce the old scheduler-only idea.
