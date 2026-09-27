from __future__ import annotations
import hashlib, json, math, os, re, urllib.error, urllib.request
from collections import Counter
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from .models import AnalysisResponse, Citation, Finding

load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt", ".md"}
MAX_FILE_SIZE = 25 * 1024 * 1024
MAX_CHUNK_CHARS = 1_800

@dataclass
class Chunk:
    id: str; document_id: str; document_name: str; page: int; section: str; text: str
@dataclass
class Node:
    id: str; label: str; type: str; aliases: set[str] = field(default_factory=set)

class LocalVectorStore:
    def __init__(self): self.items: list[tuple[Chunk, Counter[str]]] = []
    @staticmethod
    def embed(text: str): return Counter(re.findall(r"[a-z0-9]+", text.lower()))
    @staticmethod
    def cosine(left, right):
        denom = math.sqrt(sum(v*v for v in left.values()) * sum(v*v for v in right.values()))
        return sum(v * right.get(k, 0) for k, v in left.items()) / denom if denom else 0.0
    def add(self, chunks): self.items.extend((chunk, self.embed(chunk.text)) for chunk in chunks)
    def search(self, query, top_k=5):
        vector = self.embed(query)
        return [chunk for chunk, _ in sorted(self.items, key=lambda item: self.cosine(vector, item[1]), reverse=True)[:top_k]]

class KnowledgeGraph:
    def __init__(self): self.nodes: dict[str, Node] = {}; self.edges: list[dict[str, str]] = []
    def resolve_entity(self, label, kind):
        normalized = re.sub(r"[^a-z0-9]", "", label.lower())
        node_id = f"{kind.lower()}:{normalized or hashlib.sha1(label.encode()).hexdigest()[:10]}"
        if node_id not in self.nodes: self.nodes[node_id] = Node(node_id, label, kind, {normalized})
        return node_id
    def relate(self, source, predicate, target, citation):
        edge = {"source": source, "relationship": predicate, "target": target, "citation": citation.chunk_id}
        if edge not in self.edges: self.edges.append(edge)

class LLMClient:
    """OpenAI-compatible LLM client, defaulting to Groq.

    Configure GROQ_API_KEY (or LLM_API_KEY) in .env to enable live AI. Without a
    key the deterministic fallback keeps every endpoint working offline. Because
    Groq exposes an OpenAI-compatible API, any OpenAI-compatible provider works by
    overriding LLM_BASE_URL / LLM_MODEL.
    """
    SYSTEM_ANSWER = "Answer only from supplied evidence. Say when evidence is insufficient. Cite the document and location you used."
    SYSTEM_FINDINGS = (
        "You are a document due-diligence analyst. From the numbered evidence excerpts, identify the most "
        "important findings: cross-document conflicts, expected-but-missing documents, risky dates/deadlines, "
        "financial exposure, and clauses needing review. Ground every finding ONLY in the evidence and cite it "
        "by the integer 'ref' number(s) shown; never use a ref that is not present; never invent facts; do not "
        "assume any specific industry. If nothing meaningful is supported, return an empty list. Respond with "
        'STRICT JSON: {"findings":[{"title":str,"severity":"critical|high|medium|low|info","category":str,'
        '"summary":str,"evidence_refs":[int],"recommended_actions":[str]}]}'
    )
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        default_base = "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else "https://api.openai.com/v1"
        self.base_url = os.getenv("LLM_BASE_URL", default_base).rstrip("/")
        self.model = os.getenv("LLM_MODEL") or "llama-3.3-70b-versatile"
    @property
    def provider(self): return f"openai-compatible:{self.model}" if self.api_key else "local-retrieval"
    def _chat(self, messages, json_mode=False):
        """POST to the chat-completions endpoint. Returns content string or raises."""
        body = {"model": self.model, "temperature": 0, "messages": messages}
        if json_mode: body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), method="POST", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "User-Agent": "kgrag-api/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response: data = json.loads(response.read().decode())
        return data["choices"][0]["message"]["content"].strip()
    @staticmethod
    def _format_evidence(evidence): return "\n\n".join(f"[ref {i}] ({c.document_name}, location {c.page} / {c.section}) {c.text[:1200]}" for i, c in enumerate(evidence, 1))
    def answer(self, question, evidence):
        if not self.api_key:
            return "LLM is not configured. The most relevant retrieved evidence is returned in the citations." if evidence else "No relevant evidence was found in the ingested documents."
        try:
            return self._chat([{"role": "system", "content": self.SYSTEM_ANSWER}, {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{self._format_evidence(evidence)}"}])
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as error:
            return f"The configured LLM could not complete this request ({type(error).__name__}). Retrieved citations are included below."
    def findings(self, evidence):
        """Return a list of raw finding dicts grounded in `evidence`; [] on fallback/error."""
        if not self.api_key or not evidence: return []
        try:
            content = self._chat([{"role": "system", "content": self.SYSTEM_FINDINGS}, {"role": "user", "content": f"Evidence:\n{self._format_evidence(evidence)}"}], json_mode=True)
            parsed = json.loads(content).get("findings", [])
            return parsed if isinstance(parsed, list) else []
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError, TypeError):
            return []

class KGRAGPipeline:
    def __init__(self): self.chunks=[]; self.graph=KnowledgeGraph(); self.vectors=LocalVectorStore(); self.llm=LLMClient()
    def reset(self): self.__init__()
    def ingest_directory(self, directory: Path):
        paths = [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        return self.ingest_files([(p.name, p.read_bytes()) for p in paths])
    def ingest_files(self, files):
        self.reset(); accepted=0; skipped=[]
        for filename, content in files:
            if not content: skipped.append(f"{filename} (empty)"); continue
            if len(content) > MAX_FILE_SIZE: skipped.append(f"{filename} (exceeds {MAX_FILE_SIZE // 1024 // 1024} MB limit)"); continue
            if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS: skipped.append(f"{filename} (unsupported type)"); continue
            try:
                self._ingest_bytes(filename, content); accepted += 1
            except Exception as error:
                skipped.append(f"{filename} (error: {type(error).__name__})")
        if not accepted: raise ValueError("No supported, non-empty documents were provided. Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        self.vectors.add(self.chunks); self._build_graph()
        return {"ingested_documents": accepted, "skipped_files": skipped, "chunks": len(self.chunks), "entities": len(self.graph.nodes), "relationships": len(self.graph.edges)}
    def _ingest_bytes(self, filename, content):
        doc_id = hashlib.sha1(filename.encode()).hexdigest()[:12]
        for page, section, text in self._extract(filename, content):
            for number, part in enumerate(self._split_text(text), 1):
                chunk_id=hashlib.sha1(f"{doc_id}|{page}|{section}|{number}|{part}".encode()).hexdigest()[:12]
                self.chunks.append(Chunk(chunk_id, doc_id, filename, page, section, part))
    def _extract(self, filename, content):
        suffix=Path(filename).suffix.lower()
        if suffix == ".pdf":
            reader=PdfReader(BytesIO(content)); return [(i, f"Page {i}", page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
        if suffix == ".docx":
            from docx import Document
            document=Document(BytesIO(content)); text="\n".join(p.text for p in document.paragraphs if p.text.strip())
            for table in document.tables: text += "\n" + "\n".join(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
            return [(1, "Document", text)]
        if suffix == ".xlsx":
            import openpyxl
            workbook=openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True); output=[]
            for i, sheet in enumerate(workbook.worksheets, 1):
                rows=[" | ".join("" if c is None else str(c) for c in row) for row in sheet.iter_rows(values_only=True)]
                output.append((i, sheet.title, "\n".join(row for row in rows if row.strip(" |"))))
            return output
        return [(1, "Document", content.decode("utf-8", errors="replace"))]
    @staticmethod
    def _split_text(text):
        clean=re.sub(r"\s+", " ", text).strip()
        return [clean[i:i+MAX_CHUNK_CHARS] for i in range(0, len(clean), MAX_CHUNK_CHARS)] if clean else []
    @staticmethod
    def _entities(text):
        dates=re.findall(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b", text, re.I)
        phrases=re.findall(r"\b(?:[A-Z][A-Za-z0-9&.-]*\s+){1,4}[A-Z][A-Za-z0-9&.-]*\b", text)
        return list(dict.fromkeys([(x.strip(), "Date") for x in dates] + [(x.strip(), "NamedEntity") for x in phrases if len(x.strip()) > 3]))[:30]
    def _citation_for(self, chunk): return Citation(document_id=chunk.document_id, document_name=chunk.document_name, page=chunk.page, section=chunk.section, chunk_id=chunk.id, excerpt=chunk.text[:350])
    def _build_graph(self):
        for chunk in self.chunks:
            citation=self._citation_for(chunk); document=self.graph.resolve_entity(chunk.document_name, "Document"); ids=[]
            for label, kind in self._entities(chunk.text):
                entity=self.graph.resolve_entity(label, kind); self.graph.relate(document, "MENTIONS", entity, citation); ids.append(entity)
            for left, right in zip(ids, ids[1:]): self.graph.relate(left, "CO_OCCURS_WITH", right, citation)
    def analyze(self, context_chunks=40):
        count=len({c.document_id for c in self.chunks})
        evidence=self.chunks[:context_chunks]
        findings=[]
        for index, raw in enumerate(self.llm.findings(evidence), 1):
            refs=[r for r in raw.get("evidence_refs", []) if isinstance(r, int) and 1 <= r <= len(evidence)]
            severity=raw.get("severity", "info")
            if severity not in ("critical", "high", "medium", "low", "info"): severity="info"
            findings.append(Finding(id=raw.get("id") or f"finding-{index}", title=raw.get("title", "Untitled finding"), severity=severity, category=raw.get("category", "general"), summary=raw.get("summary", ""), evidence=[self._citation_for(evidence[r-1]) for r in refs], recommended_actions=[str(a) for a in raw.get("recommended_actions", [])]))
        summary=(f"Ingested {count} document(s), indexed {len(self.chunks)} chunk(s), extracted {len(self.graph.nodes)} graph node(s), and generated {len(findings)} finding(s)." if findings else f"Ingested {count} document(s), indexed {len(self.chunks)} chunk(s), and extracted {len(self.graph.nodes)} graph node(s). Configure an LLM API key to generate findings, or use /query for evidence-grounded answers.")
        return AnalysisResponse(documents_analyzed=count, findings=findings, summary=summary, generated_by=self.llm.provider)
    def query(self, question, top_k):
        evidence=self.vectors.search(question, top_k)
        return {"answer": self.llm.answer(question, evidence), "provider": self.llm.provider, "citations": [self._citation_for(c).model_dump() for c in evidence]}
