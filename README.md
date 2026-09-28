# Document KG-RAG API

Generic FastAPI service that accepts PDFs, Word documents, Excel workbooks, text, and Markdown. It returns citation-backed JSON retrieval answers and a lightweight knowledge graph. Bundled sample files are optional test data; no documents are preloaded or special-cased.

This is just a demo work to consolidate for the AEVY organisation.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: enables model-generated answers + findings; local citation retrieval works without it.
copy .env.example .env   # then set GROQ_API_KEY
uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`; operational endpoints return JSON and interactive documentation is available at `/docs`.

## Endpoints

- `GET /health` — status and configured provider (never exposes a key).
- `POST /ingest/files` — multipart upload of `.pdf`, `.docx`, `.xlsx`, `.txt`, or `.md` files. A new upload replaces the current in-memory corpus.
- `POST /ingest/sample` — ingest the bundled `sample_docs/` example corpus.
- `POST /query` — `{ "question": "...", "top_k": 5 }`, with answer and citations.
- `GET /analysis` — LLM-generated, evidence-grounded findings with citations (empty without a key).
- `GET /graph` — extracted document/entity graph.

```powershell
curl.exe -X POST http://127.0.0.1:8000/ingest/files -F "files=@C:\path\to\contract.pdf" -F "files=@C:\path\to\schedule.xlsx"
curl.exe -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d "{\"question\":\"What are the key deadlines?\"}"
```

## LLM configuration

The default provider is **Groq** (OpenAI-compatible). Set `GROQ_API_KEY` in `.env` (gitignored); `LLM_MODEL` defaults to `llama-3.3-70b-versatile`. To use another OpenAI-compatible provider, set `LLM_API_KEY` and `LLM_BASE_URL` instead. `.env.example` lists all variables. If no key is supplied, `/query` and `/analysis` stay operational: query returns cited retrieval evidence and analysis returns no findings.
