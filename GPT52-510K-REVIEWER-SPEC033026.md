Technical Specification (Streamlit on Hugging Face Spaces)

## 1. Executive Summary

The **FDA 510(k) Agentic Review System** is an AI-driven regulatory review workspace that helps medical device professionals transform unstructured submission materials into structured, review-ready outputs. This updated design **preserves all original capabilities** (Dashboard, Agent Library, Document Review, Prompt Runner, AI Note Keeper, Hermes-style UX, multi-agent sequential execution, markdown rendering, and multi-provider readiness) while introducing a **new “WOW UI” experience**, richer status telemetry, explicit API-key onboarding, advanced agent-to-agent editing controls, and a new end-to-end **510(k) Review Report Generator** workflow.

This version is designed to be **deployed on Hugging Face Spaces using Streamlit**, and to orchestrate multiple LLM providers via **Gemini API, OpenAI API, Anthropic API, and Grok API**. Agent behavior is configured through **`agents.yaml`**, with **`SKILL.md`** acting as the system prompt backbone (referenced by or embedded into agent system prompts).

Key additions in this updated design include:

- A **new WOW UI** with:
  - **Light/Dark themes**
  - **English / Traditional Chinese UI language**
  - **20 “famous painter” styles** (selectable; includes a **Jackpot** randomizer)
- **WOW status indicators** and an **interactive dashboard** showing document ingestion progress, provider health, token usage, and agent pipeline state.
- **API key input on the webpage if not found in environment**, with strict masking rules (never reveal environment-provided keys; never echo typed keys in logs).
- A redesigned **Agent Runner** that allows:
  - Pre-run editing of **prompt**, **max-tokens** (default **12000**), and **model selection** per agent
  - Editing any agent’s **output** (Text/Markdown views) to be used as the **input to the next agent**
- An expanded **AI Note Keeper** that:
  - Accepts pasted notes or uploaded files (txt/markdown)
  - Transforms into organized Markdown with **keywords highlighted in coral**
  - Supports **6 AI Magics** (defined below), including **AI Keywords** with user-controlled keyword coloring
  - Allows continued prompting on the note with selectable models
- A new **510(k) review report workflow**:
  - Users can paste/upload a 510(k) document (txt/markdown/pdf)
  - Users can paste/upload review guidance (txt/markdown) or use a provided default guidance
  - The system generates a **comprehensive 510(k) review report** in Markdown (**2000–3000 words**) in **English or Traditional Chinese**
  - Users can edit and download the report (txt/markdown) and continue prompting or run additional agents on it
- A new **agents.yaml management feature**:
  - Upload / paste / download `agents.yaml`
  - If non-standard, the system transforms it into a standardized schema

This specification describes the updated architecture, data flows, UI/UX behavior, orchestration logic, file processing pipeline, security posture, and deployment model—**without providing or modifying code**.

---

## 2. Goals, Non-Goals, and Design Principles

### 2.1 Goals
1. **Preserve all original features** while migrating to a Streamlit-first architecture suitable for Hugging Face Spaces.
2. Provide a **high-polish WOW UI** that feels premium (Hermes-inspired base) and customizable (themes, languages, painter styles).
3. Support **regulatory-grade workflows**: ingestion, structured extraction, gap analysis, checklisting, report generation, and iterative refinement.
4. Enable **agentic, human-in-the-loop** execution with editable handoffs between agents.
5. Provide robust **configuration management** through `agents.yaml` and `SKILL.md`.

### 2.2 Non-Goals
- This system is **not** a legally authoritative FDA decision engine; it produces drafts and review aids.
- This specification does not prescribe a specific OCR library implementation; it defines interfaces and expected outputs.
- This specification does not include code.

### 2.3 Design Principles
- **Traceability**: Every output should reference its inputs, guidance source, model/provider, and agent used.
- **Human control**: Users can override prompts, parameters, and intermediate outputs at every step.
- **Safety by default**: Conservative prompting, low hallucination posture, explicit “unknown/insufficient evidence” behavior.
- **Elegant UX**: High information density, smooth interactive feedback, minimal friction.

---

## 3. System Context and Deployment (Hugging Face Spaces + Streamlit)

### 3.1 Runtime Environment
- **Hosting**: Hugging Face Spaces
- **Web Framework**: Streamlit
- **State**: `st.session_state` (primary), with structured “workspace sessions”
- **Config assets**:
  - `agents.yaml` (agent definitions)
  - `SKILL.md` (system skill framework; used as system prompt scaffolding)
- **LLM Providers**:
  - Gemini API
  - OpenAI API
  - Anthropic API
  - Grok API

### 3.2 Data Persistence Model
Hugging Face Spaces often runs with ephemeral storage. The updated design assumes:
- Default: **Session-scoped** persistence in memory (`st.session_state`)
- Optional: **User export** as downloadable bundles (markdown/txt/yaml/json zip)
- Optional future extension (not required): external persistence (S3/GCS, HF Datasets, or DB)

### 3.3 Security Boundary
- The app runs server-side (Streamlit), which allows **environment keys** to be kept off the client.  
- If users input keys, the app stores them only in **session memory** and masks them in UI.

---

## 4. Updated Module Map (Preserving Original Features)

The application retains the original conceptual modules but maps them into a Streamlit multi-page or tabbed UX.

1. **WOW Dashboard** (expanded)
2. **Agent Library + agents.yaml Manager** (expanded)
3. **Document Review Workspace** (preserved + improved ingestion)
4. **Prompt Runner (Sequential Agents)** (expanded with editable handoffs)
5. **AI Note Keeper** (expanded)
6. **510(k) Review Report Generator** (new primary workflow)

---

## 5. WOW UI/UX Specification

### 5.1 Global UI Controls
The top-level header includes:
- **Theme toggle**: Light / Dark
- **UI language**: English / Traditional Chinese
- **Painter Style selector**: 20 famous painter-inspired styles
- **Jackpot button**: randomly selects a painter style with a short animation and “style reveal”
- **Provider status cluster**: small icons for Gemini/OpenAI/Anthropic/Grok with real-time availability and last latency

These controls apply across all modules, updating the visual layer immediately while preserving the underlying workspace state.

### 5.2 Themes (Light/Dark)
The system supports two base themes, each compatible with painter styles:
- **Light**: Hermes cream backgrounds, warm browns, amber/orange accents, coral highlights
- **Dark**: Deep brown/charcoal surfaces, warm amber/gold accents, coral highlights preserved for keyword emphasis

Theme affects:
- Backgrounds, cards, borders, table styling
- Markdown rendering palette (headings, code blocks, blockquotes)
- Status indicator colors (success/warn/error remain consistent but tuned for contrast)

### 5.3 UI Language vs Output Language
- **UI Language** controls labels, tooltips, navigation, and helper text (English/Traditional Chinese).
- **Output Language** is a per-task setting (especially for 510(k) report generation) that determines the language of AI-generated deliverables.

### 5.4 Painter Styles (20 presets)
Painter styles are *visual skins* applied on top of Light/Dark. Each style defines:
- Primary accent color
- Secondary accent color
- Background tint
- Card gradient behavior
- Typography nuance (e.g., serif headings vs sans headings), while keeping readability standards

**20 styles (examples with intended feel):**
1. Leonardo da Vinci — parchment, sepia ink accents  
2. Vincent van Gogh — bold cobalt/amber, textured card borders  
3. Claude Monet — pastel wash gradients, soft separators  
4. Pablo Picasso — geometric blocks, stronger outlines  
5. Salvador Dalí — surreal highlights, animated “melting” progress bars (subtle)  
6. Rembrandt — chiaroscuro contrast, dramatic lighting  
7. Frida Kahlo — floral accents, vibrant coral emphasis  
8. Katsushika Hokusai — wave blues, crisp line separators  
9. Jackson Pollock — speckled background noise (low intensity)  
10. Gustav Klimt — gold-leaf accents, ornate dividers  
11. Georgia O’Keeffe — minimal forms, soft rose/cream  
12. Andy Warhol — pop color badges, bold status pills  
13. Edward Hopper — quiet muted tones, strong layout grid  
14. Caravaggio — deep shadows, spotlight cards  
15. Jean-Michel Basquiat — scribble motifs in headers (very subtle)  
16. Johannes Vermeer — calm blue/yellow balance  
17. Wassily Kandinsky — abstract accent shapes for widgets  
18. M.C. Escher — tessellated background watermark (very light)  
19. Yayoi Kusama — dot motifs for loading states  
20. Paul Cézanne — earthy palette, structured cards

**Jackpot behavior**:
- Randomly selects one of the 20 styles.
- Logs the selection in session activity history.
- Does not alter underlying document/agent data.

---

## 6. WOW Status Indicators & Interactive Dashboard

### 6.1 Status Indicator System (Global)
A unified status indicator model is used across ingestion and LLM actions:

- **Idle**: ready state
- **Queued**: pending action (e.g., waiting for model call)
- **Running**: in-progress with spinner + elapsed time
- **Streaming/Receiving** (optional): token streaming indicator
- **Success**: completed with output size, token estimate, and duration
- **Warning**: partial output / low OCR confidence / context truncation
- **Error**: provider error, parsing failure, guidance missing, etc.

Indicators appear:
- In the header (provider health)
- In the Dashboard (pipeline view)
- In each module’s local panel (Document ingestion, Agent execution, Report generation)

### 6.2 Dashboard (New “Awesome” Interactivity)
The dashboard becomes the operational cockpit:

**A. Workspace Snapshot**
- Current workspace name / timestamp
- Active document set (510(k) doc + guidance doc + notes)
- Selected theme + painter style
- Selected default model/provider

**B. Provider Health Panel**
For each provider (Gemini/OpenAI/Anthropic/Grok):
- Availability (Up/Down/Degraded)
- Last request latency
- Recent error count (session-scoped)
- Key source: Env / User-supplied (display only the source, never the key)

**C. Document Ingestion Metrics**
- File types loaded (txt/md/pdf)
- Extracted text length
- OCR confidence (if OCR used)
- Detected language ratio (EN/ZH heuristic)
- “Context readiness”: estimated chunk count for downstream steps

**D. Agent Pipeline View**
An interactive “agent chain” visualization:
- Nodes = agents
- Edges = handoff
- Each node shows: model, max tokens, status, output size
- Clicking a node opens its output and run parameters (read-only in Dashboard; editable in Runner)

**E. Cost/Token Telemetry (Session Scoped)**
- Estimated prompt tokens / completion tokens per run
- Aggregate estimate by provider/model
- Warning if approaching provider limits or likely context overflow

**F. Compliance Heatmap (Preserved and Expanded)**
Still shows major modules (biocompatibility, software, EMC, labeling, sterilization, clinical, SE analysis), but now:
- Can be generated from agent outputs
- Supports “confidence” and “evidence completeness” overlays

---

## 7. API Key Handling (Environment + Web Input)

### 7.1 Priority Rules
For each provider, key resolution follows:
1. **Environment/Secrets** (HF Spaces secrets or environment variables)
2. **User input on webpage** (session-only)

### 7.2 UI Rules (Strict)
- If the key is present in environment:
  - UI displays: “Key loaded from environment”
  - No input required
  - No reveal button
- If missing:
  - UI shows a masked input field
  - Stored only in `session_state`
  - Never printed in logs, never included in downloads, never included in error traces
- Provide a “Clear session keys” action to remove user-provided keys immediately.

### 7.3 Operational Safety
- Add a visible banner: “Do not paste PHI or confidential data unless you have appropriate agreements and settings with the provider.”
- Provide provider-specific toggles such as “Use enterprise/zero-retention endpoint” (configuration-only; no code here).

---

## 8. Model & Parameter Controls (Per Agent, Pre-Run)

### 8.1 Supported Models (User Selectable)
The updated design supports selecting from (at minimum):
- **OpenAI**: `gpt-4o-mini`, `gpt-4.1-mini`
- **Gemini**: `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3-flash-preview`
- **Anthropic**: “anthropic models” (present as a dynamic list in UI; may be configured)
- **Grok**: `grok-4-fast-reasoning`, `grok-3-mini`

### 8.2 Max Tokens
- Default: **12000**
- Per-agent override allowed
- System warns if selected model is likely to reject the requested token budget.

### 8.3 Prompt Editing & System Prompt Strategy
Each agent run includes:
- **User prompt** (editable)
- **Agent system prompt** (from `agents.yaml` and SKILL.md integration)
- **Context attachments** (document excerpts, guidance, previous agent output)

Users can:
- Modify the user prompt per agent
- Optionally view (and if permitted by role) the system prompt in a read-only inspector
- Add “constraints” toggles: “Use bullet checklists”, “Include tables”, “Cite sections”, “Mark unknowns explicitly”

---

## 9. Document Ingestion & Normalization (txt/md/pdf)

### 9.1 Inputs
**510(k) Document** (one or multiple):
- Paste text
- Upload file: txt, md, pdf

**510(k) Review Guidance**:
- Paste text
- Upload file: txt, md
- Or select **Default Guidance** (provided in Traditional Chinese, as supplied)

### 9.2 Extraction Pipeline Requirements
For each uploaded document, system produces a **Normalized Document Object**:

- `doc_id`
- `doc_type`: `510k_submission` | `guidance` | `review_notes` | `other`
- `source`: upload | paste | default
- `original_filename`
- `mime_type`
- `raw_bytes` (optional, session only)
- `extracted_text` (required)
- `extraction_method`: text | pdf_text | ocr
- `extraction_warnings`: list
- `language_detected`: en/zh/mixed
- `page_map` (for PDF): page → extracted text block (when feasible)
- `hash` for change detection (session scoped)

### 9.3 PDF Handling
The design supports two PDF strategies:
- **Text-based PDF extraction**: preferred when selectable text exists
- **OCR fallback**:
  - triggered when extracted text is below a threshold or mostly whitespace
  - outputs OCR confidence estimate
  - flags “OCR used” prominently in report provenance

### 9.4 Chunking for LLM Context
Because 510(k) materials can exceed context windows:
- The system builds **semantic chunks** with section headers when possible.
- Maintains a chunk index:
  - chunk id
  - source doc and page (if PDF)
  - token estimate
  - top keywords
- Agents can request:
  - “summary of all chunks”
  - “retrieve top N chunks relevant to X”
This enables a RAG-like behavior without requiring a full vector database (optional future enhancement).

---

## 10. The 510(k) Review Report Generator (New Primary Workflow)

### 10.1 Purpose
Generate a **comprehensive 510(k) review report** in Markdown (**2000–3000 words**) grounded in:
- User’s 510(k) submission document(s)
- User-provided guidance or the default guidance
- The user’s selected output language (English/Traditional Chinese)

### 10.2 UX Flow
1. **Step A — Upload/Paste 510(k) Document**
   - Accept multiple files and/or pasted content
   - Show extraction preview and warnings (e.g., “OCR confidence low”)

2. **Step B — Provide Guidance**
   - Upload/paste guidance or select default guidance
   - Show diff view if user guidance replaces default (optional)

3. **Step C — Configure Generation**
   - Output language: English / Traditional Chinese
   - Select model/provider
   - Max tokens and “verbosity target” tuned to 2000–3000 words
   - Edit prompt (pre-filled with the provided default prompt)

4. **Step D — Generate Report**
   - Show running status, elapsed time, and intermediate steps (Analyze → Template → Checklist → Tables → Entities)

5. **Step E — Review & Edit**
   - Split-view editor:
     - Text view
     - Markdown-rendered view
   - Users can edit and re-render instantly

6. **Step F — Export**
   - Download as `.md` and `.txt`
   - Optional export bundle containing: report, prompt used, doc provenance, and run metadata

7. **Step G — Continue Work**
   - “Keep prompting on report” (chat-like iterative refinement using selected model)
   - “Run agents on report” (send report into Prompt Runner chain)
   - “Attach to workspace notes” (send to AI Note Keeper)

### 10.3 Default Prompt (Baseline)
The UI provides the exact default prompt supplied by the user (STEP 1–STEP 5). The system treats it as a template and adds:
- A provenance section requirement (“what was provided, what was missing”)
- A strict instruction: “Do not invent testing outcomes; state ‘Not provided’ when absent.”

### 10.4 Output Requirements (Report Content Contract)
The report must include:

- **Structured chapters**:
  - Introduction and scope
  - Device classification
  - Product specifications
  - Safety/performance testing
  - Biocompatibility
  - Substantial equivalence analysis
  - Labeling/IFU considerations
  - Software and cybersecurity (if applicable)
  - Risk management alignment (ISO 14971 framing)
  - Conclusion and actionable recommendations

- **Comprehensive review checklist**:
  - Administrative documents
  - Technical specifications
  - Safety testing reports
  - Performance validation
  - Clinical information
  - Labeling/IFU
  - Manufacturing information
  - Sterilization/disinfection (if applicable)

Each checklist item includes:
- Item description
- Required documentation
- Applicable standards
- Acceptance criteria
- Review notes/considerations

- **5 additional tables**, such as:
  1. Predicate vs subject device comparison matrix
  2. Test requirement matrix by standard
  3. Risk control mapping table (hazard → control → verification)
  4. Standards compliance tracker (claimed vs evidenced vs missing)
  5. Review timeline / decision flow

- **20 entities with context**:
  - Definition
  - Regulatory context
  - Relevance to the device type
  - Related standards
  - Practical reviewer implications

- **Evidence labeling**:
  - For each major claim, tag as: Provided / Partially provided / Not provided

### 10.5 Guidance Selection Logic
- If user supplies guidance: it becomes primary.
- If user guidance is partial: the system may supplement with default guidance, but must label what came from where.
- If no guidance supplied: default guidance is used.

---

## 11. Prompt Runner (Sequential Agents) — Editable Handoffs

### 11.1 Core Concept (Preserved)
The Prompt Runner executes agents **one by one** in a defined sequence and maintains a live run log.

### 11.2 New Capabilities
For each agent in sequence:
- Before execution:
  - User can modify prompt
  - Select model/provider
  - Set max tokens (default 12000)
- After execution:
  - Output shown in **Text** and **Markdown** views
  - User can **edit** the output
  - Edited output becomes **input to next agent**
- Provide “lock” controls:
  - Lock output to prevent accidental editing
  - “Revert to raw output” to restore the original LLM response

### 11.3 Run Ledger (Traceability)
Each agent run produces metadata:
- Agent ID/name
- Provider/model
- Prompt version hash
- Input sources (doc chunks, guidance, previous output)
- Timing and token estimates
- Status and error details

Users can export the run ledger for audit or internal QA.

---

## 12. AI Note Keeper — Expanded Feature Set

### 12.1 Inputs
- Paste note (txt/markdown)
- Upload note file (txt/markdown)

### 12.2 Transform Output Contract
- Produces **organized Markdown** with:
  - Clear headings
  - Bullet points
  - Action items
  - Open questions
  - Traceability tags when content refers to the 510(k) report or source docs
- Highlights **key words in coral color**:
  - In Markdown, keywords are wrapped in strong emphasis (`**keyword**`)
  - Rendering layer applies coral styling to strong emphasis tokens consistently across themes/styles

### 12.3 Dual-View Editing
- Text editor (raw)
- Markdown rendered preview
- Changes in one view sync to the other (with predictable formatting rules)

### 12.4 “Keep Prompting on the Note”
- A chat-like iteration panel:
  - The note is the “working memory”
  - User selects model/provider
  - Each assistant response can be inserted into the note or kept separate

### 12.5 Six AI Magics (Updated)
The system includes six curated functions (preserving spirit of original magics while adding the required keyword feature):

1. **Auto-Structure**: reorganize into compliant, readable Markdown
2. **Executive Summary**: concise regulatory summary + top risks
3. **Risk Flagging**: identify compliance risks and missing evidence, label severity
4. **Predicate Matcher**: suggest predicate categories and comparison dimensions (non-authoritative)
5. **Table Extractor**: convert structured items into Markdown tables
6. **AI Keywords (New)**:
   - User inputs a list of keywords/phrases
   - User chooses a color per keyword group (with coral as default)
   - Output marks those keywords consistently (via Markdown emphasis and/or inline span conventions supported by renderer)
   - Includes a keyword index section at the top of the note

---

## 13. Agent Library + agents.yaml Setup Feature

### 13.1 Core Library (Preserved)
- Shows available agents
- Each agent has:
  - name, description
  - provider/model defaults
  - temperature/top-p (if used)
  - max tokens default
  - system prompt (derived from SKILL.md + agent-specific instruction)

### 13.2 agents.yaml Management (New)
Users can:
- **Upload** `agents.yaml`
- **Paste** YAML content
- **Download** the current standardized `agents.yaml`

### 13.3 Standardized agents.yaml Schema (Design Contract)
A standardized agent entry must include:

- `id`: stable identifier (string or int, normalized to string)
- `name`: display name
- `description`: purpose and expected outputs
- `provider`: `gemini` | `openai` | `anthropic` | `grok`
- `model`: default model string
- `max_tokens`: default 12000 unless specified
- `temperature`: default 0.2 for regulatory tasks unless specified
- `system_prompt`:
  - must reference SKILL.md content conceptually (either embedded or appended at runtime)
  - must include formatting requirements (Markdown, tables where needed, “do not invent data”)
- `input_contract`: expected input format (text/markdown + optional attachments)
- `output_contract`: expected output format (markdown/text), required sections, table requirements
- `ui` metadata:
  - recommended icon
  - tags (e.g., “biocompatibility”, “software”, “SE”, “labeling”)

### 13.4 Non-Standard YAML Transformation Rules
If uploaded YAML is not standardized, the system transforms it using deterministic rules:

- Map alternative keys:
  - `prompt`/`instruction` → `system_prompt`
  - `llm`/`engine` → `model`
  - `vendor` → `provider`
- Fill defaults when missing:
  - `max_tokens: 12000`
  - `temperature: 0.2`
  - `provider` inferred from model naming conventions (with user confirmation banner)
- Normalize types:
  - Ensure `id` is unique and string
  - Ensure numeric fields are numeric
- Validate safety constraints:
  - If system prompt lacks “do not fabricate” clause, append a standard clause
- Emit a transformation report:
  - What changed
  - What defaults were applied
  - Any unresolved ambiguities

### 13.5 Default agents.yaml (Bundled)
The app includes a bundled default `agents.yaml` aligned to 510(k) review tasks (gap analysis, checklist, SE comparison, biocompatibility assessor, software/IEC 62304 reviewer, EMC/60601 reviewer, labeling reviewer, final report editor). The user can download and customize it.

---

## 14. SKILL.md Integration (System Prompt Backbone)

### 14.1 Purpose
`SKILL.md` defines global reviewer behavior, such as:
- Regulatory tone and structure expectations
- Evidence discipline (mark missing info)
- Formatting rules (Markdown headings, tables)
- Safety constraints (no legal claims, no invention of test data)
- Multilingual guidance for English/Traditional Chinese output

### 14.2 Composition Strategy
For each agent execution, the runtime system prompt is composed from:
1. SKILL.md (global policy)
2. Agent system_prompt (role-specific)
3. Task constraints (output language, word count, required sections)

This preserves consistent behavior across all tasks.

---

## 15. Downloads, Export Bundles, and Workspace Packaging

### 15.1 Supported Downloads
- 510(k) review report: `.md`, `.txt`
- Notes: `.md`, `.txt`
- Agent run outputs: per-agent `.md/.txt` and combined “run book”
- `agents.yaml`: standardized output
- Optional bundle export: zip containing:
  - outputs
  - prompts used
  - run ledger metadata
  - ingestion provenance (file names, extraction method, OCR confidence)

### 15.2 Redaction Rules for Exports
- Never include API keys
- Optionally allow users to redact file names and replace with doc IDs
- Provide a “confidential mode” that removes provider latency and token info from exports

---

## 16. Quality, Safety, and Compliance Considerations

### 16.1 Hallucination and Evidence Discipline
All regulatory outputs must:
- Prefer quoting or pointing to source excerpts when available
- Use explicit labels: “Not provided”, “Insufficient evidence”, “Assumption (needs verification)”
- Avoid inventing:
  - test results
  - standard compliance claims
  - predicate clearances

### 16.2 Sensitive Data Handling
- Display warning: do not submit PHI unless authorized
- Session reset clears documents and keys
- Avoid persistent logging of raw document content (only store in session)

### 16.3 Accessibility and Readability
- Maintain WCAG-friendly contrast in both Light and Dark
- Ensure tables are readable and scrollable
- Provide font sizing controls (small/medium/large)

---

## 17. Performance and Scalability

### 17.1 Responsiveness
- Use progressive disclosure: show previews and summaries rather than full extracted text by default
- For long PDFs: show page-level extraction status and chunk counts

### 17.2 Large Document Strategy
- Chunk and summarize before full report generation
- Cache extraction and chunk indexes in session_state
- Warn users when likely to exceed context; offer “summarize-first mode”

### 17.3 Provider Fallback Behavior (Design)
When a provider fails:
- Show actionable error details (rate limit vs auth vs invalid model)
- Offer a one-click rerun with a backup model/provider chosen by the user (no automatic silent switching unless enabled)

---

## 18. End-to-End User Journeys (Updated)

### 18.1 “Generate 510(k) Review Report” Journey
1. Upload/paste 510(k) doc (pdf ok)
2. Upload/paste guidance or choose default
3. Pick output language + model + max tokens
4. Generate report (2000–3000 words)
5. Edit in Markdown/Text view
6. Download report
7. Keep prompting and/or run a chain of agents for refinement

### 18.2 “Agentic Review Chain” Journey
1. Load agents.yaml (default or custom)
2. Select N agents
3. Configure per-agent model/prompt/max tokens
4. Run sequentially
5. Edit outputs between steps
6. Export run ledger and final deliverables

### 18.3 “AI Note Keeper” Journey
1. Paste notes or upload markdown
2. Run Auto-Structure + AI Keywords
3. Edit and keep prompting
4. Export notes and incorporate into report

---

## 19. Acceptance Criteria (What “Done” Looks Like)

- Users can switch **Light/Dark**, **English/Traditional Chinese UI**, and any of **20 painter styles**; Jackpot works.
- Dashboard shows provider health, ingestion metrics, pipeline status, and token/cost estimates.
- If env key exists, UI indicates it without revealing it; if missing, user can input a key and it stays masked.
- Prompt Runner supports per-agent configuration and **editable output handoff** to next agent.
- Note Keeper transforms notes into organized Markdown with **coral keyword highlighting**, supports editing, continued prompting, and 6 magics including AI Keywords with user-defined colors.
- Users can upload/paste 510(k) doc (txt/md/pdf) and guidance (txt/md) or use default guidance, then generate **2000–3000 word** Markdown review report in English/Traditional Chinese, edit it, and download it.
- Users can keep prompting on the report or run additional agents on it.
- Users can upload/paste/download `agents.yaml`; non-standard YAML is transformed into standardized YAML with a transformation report.

---

## 20. Follow-Up Questions (Comprehensive, 20)

1. **Regulatory scope**: Should the 510(k) report generator support multiple device families beyond ultrasound (e.g., orthopedic, IVD, SaMD) via selectable default guidance packs, or remain ultrasound-first?
2. **Citation policy**: Do you want the generated report to include **inline citations** pointing to extracted page numbers/chunks from the uploaded PDFs, and if so, what citation format is preferred?
3. **Evidence grading**: Should the system implement a formal evidence rubric (e.g., Provided/Partial/Missing + confidence score) per checklist item and per standard?
4. **PDF OCR expectations**: For scanned PDFs, is best-effort OCR sufficient, or do you require a minimum OCR confidence threshold that blocks report generation until improved?
5. **Multilingual behavior**: When output language is Traditional Chinese, should standards and CFR references remain in English, or be bilingual (Chinese explanation + English standard identifiers)?
6. **Default guidance governance**: How should updates to the default guidance be controlled—manual versioning, date-stamped releases, or user-managed guidance libraries?
7. **Model governance**: Do you want an admin-level policy that restricts certain models/providers for sensitive submissions (e.g., only enterprise endpoints)?
8. **Token budgeting**: Should the system automatically compute a recommended max_tokens and chunk budget based on document length to reduce failures, or keep it fully manual?
9. **Agent library design**: Should agents be grouped by regulatory domains (EMC, biocompatibility, software, clinical, labeling), by workflow stage (ingest → analyze → draft → finalize), or both?
10. **agents.yaml schema strictness**: Should non-standard YAML transformation be fully automatic, or require user confirmation when provider/model inference is ambiguous?
11. **SKILL.md customization**: Do you want SKILL.md to be user-editable in the UI with versioning, or locked as a system artifact to ensure consistent reviewer behavior?
12. **Report length control**: Should 2000–3000 words be enforced strictly (with automatic trimming/expansion), or treated as a target range?
13. **Human-in-the-loop gates**: Should the Prompt Runner support “approval checkpoints” that require user sign-off before the next agent executes?
14. **Audit and compliance**: Do you need immutable audit trails aligned with 21 CFR Part 11 concepts (user identity, timestamps, signed outputs), even in a HF Spaces context?
15. **Document redaction tools**: Should the app offer built-in redaction (mask serial numbers, patient identifiers, proprietary names) prior to sending content to LLMs?
16. **Comparison automation**: For substantial equivalence, do you want structured comparison tables that can ingest predicate device specs from user uploads, or only from user-provided text?
17. **Interactive dashboard depth**: Should the dashboard include drill-downs to chunk-level ingestion details and per-agent token usage graphs, or keep metrics high-level?
18. **Export formats**: Beyond txt/markdown, do you need PDF/DOCX exports with branded styling matching the selected painter style (or a professional neutral template)?
19. **Collaboration**: Should the design support multi-user collaboration (shared workspaces, comments, role-based access), or remain single-user/session-based?
20. **Safety and liability messaging**: What exact disclaimers and “intended use” statements should appear in the UI and exports to align with your organization’s regulatory and legal posture?
