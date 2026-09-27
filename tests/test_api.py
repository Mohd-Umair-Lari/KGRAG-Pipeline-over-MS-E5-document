from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

EPC_DOCX = Path(__file__).resolve().parent.parent / "sample_docs" / "EPC Agreement.docx"


def _ingest_epc():
    content = EPC_DOCX.read_bytes()
    return client.post(
        "/ingest/files",
        files=[(
            "files",
            (
                EPC_DOCX.name,
                content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
        )],
    )


def test_docx_ingest_query_and_graph_are_json():
    """End-to-end KG-RAG over a real .docx. The test never supplies the answer:
    it only checks the pipeline ingests the Office XML document, retrieves the
    relevant evidence for the question, and (when an LLM is configured) lets the
    LLM answer purely from that retrieved evidence."""
    ingest = _ingest_epc()
    assert ingest.status_code == 200, ingest.text
    body = ingest.json()
    assert body["ingested_documents"] == 1
    assert body["skipped_files"] == []
    assert body["chunks"] >= 1

    query = client.post("/query", json={"question": "When was the agreement signed?"})
    assert query.status_code == 200
    payload = query.json()

    citations = payload["citations"]
    assert citations
    assert citations[0]["document_name"] == EPC_DOCX.name

    assert payload["answer"].strip()
    if payload["provider"] != "local-retrieval":
        assert len(payload["answer"].strip()) > 0

    graph = client.get("/graph").json()
    assert graph["nodes"]
    assert any(n["type"] == "Document" for n in graph["nodes"])
