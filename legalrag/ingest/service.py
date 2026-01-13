from __future__ import annotations
from fastapi import UploadFile, BackgroundTasks
from legalrag.ingest.ingestor import PDFIngestor
from legalrag.ingest.orchestrator import IngestOrchestrator
from legalrag.ingest.task_queue import TaskQueue

class IngestService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.status = {}
        self.ingestor = PDFIngestor(cfg)
        self.orchestrator = IngestOrchestrator(cfg, self.status)
        self.queue = TaskQueue(name="ingest-index-queue")

    async def ingest_pdf_and_schedule(self, file: UploadFile, background_tasks: BackgroundTasks):
        result = await self.ingestor.ingest(file)
        self.status[result.doc_id] = {
            "faiss": "scheduled",
            "bm25": "scheduled",
            "colbert": "scheduled",
            "graph": "scheduled",
            "added": 0,
            "error": None,
        }
        self.queue.enqueue(self.orchestrator.faiss_job, result.jsonl_path, result.doc_id)
        self.queue.enqueue(self.orchestrator.bm25_job, result.doc_id)
        self.queue.enqueue(self.orchestrator.colbert_job, result.doc_id)
        self.queue.enqueue(self.orchestrator.graph_job, result.doc_id)
        return result

    def get_status(self, doc_id: str):
        return self.status.get(doc_id, {})
