from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from .models import AnalysisResponse, GraphResponse, IngestResponse, QueryRequest, QueryResponse
from .pipeline import KGRAGPipeline

app = FastAPI(title="Document KG-RAG API", version="1.0.0", description="Generic document ingestion, retrieval, and graph API. Responses are JSON.")
pipeline = KGRAGPipeline()
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"

@app.get("/")
def start():
    return {"message from server" : "The app has been started succesfully without any issues"}

@app.get("/health")
def health():
    return {"status": "ok", "documents_loaded": len({c.document_id for c in pipeline.chunks}), "llm_provider": pipeline.llm.provider}

@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(files: list[UploadFile] = File(...)):
    if not files: raise HTTPException(422, "Provide at least one file")
    try:
        return pipeline.ingest_files([(file.filename or "upload", await file.read()) for file in files])
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        raise HTTPException(422, f"Could not read uploaded document: {type(error).__name__}") from error

@app.post("/ingest/sample", response_model=IngestResponse)
def ingest_sample():
    if not SAMPLE_DIR.is_dir(): raise HTTPException(404, "No sample_docs directory found")
    try:
        return pipeline.ingest_directory(SAMPLE_DIR)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

@app.get("/analysis", response_model=AnalysisResponse)
def analysis():
    if not pipeline.chunks: raise HTTPException(409, "Ingest documents first")
    return pipeline.analyze()

@app.get("/graph", response_model=GraphResponse)
def graph():
    return {"nodes": [{"id": n.id, "label": n.label, "type": n.type} for n in pipeline.graph.nodes.values()], "edges": pipeline.graph.edges}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not pipeline.chunks: raise HTTPException(409, "Ingest documents first")
    return pipeline.query(request.question, request.top_k)
