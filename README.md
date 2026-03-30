# Sparkz Backend

This backend contains:

- A robust XLSX checklist parser
- A clean FastAPI application foundation
- Typed environment settings
- Supabase client wiring for upcoming ingestion/retrieval flows

## Project layout

- `app/main.py`: FastAPI app factory and middleware setup
- `app/core/config.py`: typed settings (`.env` driven)
- `app/core/logging.py`: centralized log setup
- `app/api/`: API routing
- `app/db/supabase.py`: Supabase client factory
- `app/pipeline/`: PDF → redact → chunk → **embed** (document side only)
- `app/evaluation/`: vector **top-k retrieval** + **LLM final verdict**
- `app/services/analysis_runner.py`: end-to-end job after upload
- `app/checklists/`: XLSX parser and checklist domain models
- `app/core/checklist_type_keys.py`: canonical framework keys (`ifrs`, `frs102`, `frs105`)
- `scripts/preview_extraction.py`: CLI parser preview tool
- `tests/`: parser + settings tests

## Checklist types (Supabase)

The product uses **three** `checklist_types.type_key` values, which must match what the upload form sends and what ingestion writes:

| `type_key` | Typical source workbook(s) |
|------------|------------------------------|
| `ifrs` | `IFRS_DC_2025.xlsx`, `Listed_Co_DC_2025.xlsx` |
| `frs102` | `FRS1021A_DC_2025.xlsx`, `LLP_FRS102_DC_2025.xlsx`, `Private_Co_FRS102_2025.xlsx`, `Charities_FRS102_DC_2025.xlsx` |
| `frs105` | `FRS105_DC_2025.xlsx` |

`sync_checklists_to_supabase.py` maps filenames to these keys automatically. If you previously ingested under old names (e.g. `ifrs_dc_2025`), **re-run sync** with `--replace-type` so `checklist_items` / embeddings align, or migrate data in Supabase.

## What it extracts

For each requirement row, it returns:

- `framework`
- `sheet_name`
- `section_path`
- `requirement_id`
- `requirement_text`
- `reference_text`
- `embedding_text` (joined canonical text for vector DB)

## Why this parser is robust

- Works directly on XLSX XML (no external spreadsheet dependency).
- Handles multi-sheet workbooks.
- Skips top metadata rows (`Client`, `Year-end`, etc.).
- Detects requirement IDs (`1.01`, `A3.01`, etc.).
- Merges continuation rows where IDs are blank.
- Normalizes broken quotes/spaces from Excel exports.

## Usage

```python
from app.checklists import parse_workbook

items = parse_workbook("Disclosure Checklists/FRS1021A_DC_2025.xlsx")
for item in items[:3]:
    print(item.requirement_id, item.requirement_text)
```

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Local setup

1) Create `.env` from template:

```bash
copy .env.example .env
```

2) Install dependencies:

```bash
pip install -e .
```

3) Run API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or use:

```bash
python run.py
```

4) Check health:

- `GET http://localhost:8000/api/health`
- `GET http://localhost:8000/api/health/supabase`

## Preview extraction for one workbook

```bash
python scripts/preview_extraction.py "E:\Tasks\Disclosure-RAG\Disclosure Checklists\FRS1021A_DC_2025.xlsx"
```

Useful options:

```bash
python scripts/preview_extraction.py "path\to\file.xlsx" --limit 30 --sheet "Small DC"
python scripts/preview_extraction.py "path\to\file.xlsx" --json output\parsed.json --csv output\parsed.csv
```

## Extract and save checklist rows to files

```bash
python scripts/extract_checklists.py "E:\Tasks\Disclosure-RAG\Disclosure Checklists\FRS1021A_DC_2025.xlsx" --json output\frs1021a.json --csv output\frs1021a.csv --txt output\frs1021a.txt
```

## Notes

- Parser output is ready to map directly into Supabase tables.
- `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are required for DB operations.
- Ingestion and vector storage modules are the next implementation step.

## Supabase schema + checklist sync

1) Open Supabase SQL Editor and run:

- `sql/migrations/001_checklist_schema.sql`
- `sql/migrations/002_analysis_pipeline_schema.sql`
- `sql/migrations/004_analysis_results_explanation.sql` (LLM rationale column)

2) Dry-run workbook parsing and type detection:

```bash
python scripts/sync_checklists_to_supabase.py --scan-root "E:\Tasks\Disclosure-RAG" --dry-run
```

3) Sync parsed rows into Supabase:

```bash
python scripts/sync_checklists_to_supabase.py --scan-root "E:\Tasks\Disclosure-RAG" --replace-type
```

By default, this command now also generates OpenAI embeddings and upserts vectors into `checklist_item_embeddings`.

**Retrieval vectors** use **obligation-only** text (`requirement_text_leaf` / `requirement_text`), not the long stored `embedding_text` (framework headers, IDs, references). That aligns rule and PDF chunk embeddings so similarity scores reflect “this passage is about this duty,” not arbitrary thresholds. After changing this logic, **re-embed** existing data:

```bash
python scripts/sync_checklists_to_supabase.py --scan-root "path/to/Disclosure Checklists" --only-embeddings --refresh-embeddings
```

Cosine similarity is **never** “100% = compliant.” Identical strings can approach 1.0; paraphrases rarely do. In this project, cosine is used for retrieval ranking only; final status is judged by OpenAI from requirement + extracted evidence.

Optional:

- Target specific files by repeating `--workbook`
- Adjust batch size with `--batch-size 250`
- Skip embeddings with `--skip-embeddings`
- Run embeddings only with `--only-embeddings`
- Tune OpenAI batch size with `--embedding-batch-size 50`

## Analysis API flow (upload -> result)

- `POST /api/analyses` with multipart fields:
  - `companyName`
  - `framework` (e.g. `ifrs_dc_2025`, `frs1021a_dc_2025`)
  - `file` (PDF)
- `GET /api/analyses/{analysis_id}/status`
- `GET /api/analyses/{analysis_id}/result`

The processing pipeline performs:

- PDF text extraction (`pypdf`)
- Optional OCR fallback (if `ENABLE_OCR=true` and OCR deps installed)
- PII redaction (regex-based MVP)
- Chunking
- OpenAI embeddings for chunks
- **Top-k** chunk retrieval per requirement (batched reads + NumPy cosine similarity — one matrix multiply per analysis)
- Status: OpenAI returns the final `fully_met` / `partially_met` / `missing` verdict for each rule from requirement text + retrieved evidence. Evidence is always extractive from chunks.

Embedding tokens apply to uploaded PDF chunks; checklist rows were embedded at sync time. Chat completions use **low `max_tokens`** and **strict one-word** instructions; answers are normalized with tolerant parsing (plain text, JSON-shaped noise, etc.).

## OCR note

OCR is optional and disabled by default:

- Set `ENABLE_OCR=true` in `.env`
- Install OCR dependencies as needed:
  - `pip install pdf2image pytesseract`
  - Tesseract binary and poppler must be installed on the machine

