from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .pipeline import run_pipeline_job


@dataclass
class JobState:
    id: str
    status: str = "queued"
    progress: int = 0
    progress_text: str = "Queued"
    logs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: str | None = None
    output_dir: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)


class JobManager:
    def __init__(self, runs_root: Path):
        self.runs_root = runs_root
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, JobState] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2)

    def _add_log(self, job_id: str, message: str):
        with self.lock:
            job = self.jobs[job_id]
            job.logs.append(message)
            # Prevent unlimited growth
            if len(job.logs) > 2000:
                job.logs = job.logs[-2000:]

    def _set_progress(self, job_id: str, pct: int, text: str):
        with self.lock:
            job = self.jobs[job_id]
            job.progress = int(max(0, min(100, pct)))
            job.progress_text = text

    def _run_job(self, job_id: str, csv_path: Path, model_name: str, include_latent_vis: bool):
        with self.lock:
            self.jobs[job_id].status = "running"

        out_dir = self.runs_root / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = run_pipeline_job(
                csv_path=csv_path,
                out_dir=out_dir,
                model_name=model_name,
                include_latent_vis=include_latent_vis,
                log=lambda msg: self._add_log(job_id, msg),
                progress=lambda pct, text: self._set_progress(job_id, pct, text),
            )
            with self.lock:
                job = self.jobs[job_id]
                job.status = "completed"
                job.progress = 100
                job.progress_text = "Completed"
                job.summary = result.summary
                job.artifacts = result.artifact_paths
                job.output_dir = str(out_dir)
                job.finished_at = datetime.utcnow().isoformat()
        except Exception as exc:
            with self.lock:
                job = self.jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = datetime.utcnow().isoformat()

    def create_job(self, csv_path: Path, model_name: str, include_latent_vis: bool) -> JobState:
        job_id = str(uuid.uuid4())
        state = JobState(id=job_id)
        with self.lock:
            self.jobs[job_id] = state
        self.pool.submit(self._run_job, job_id, csv_path, model_name, include_latent_vis)
        return state

    def get_job(self, job_id: str) -> JobState | None:
        with self.lock:
            return self.jobs.get(job_id)

    def latest_completed(self) -> list[JobState]:
        with self.lock:
            jobs = [j for j in self.jobs.values() if j.status == "completed"]
        return sorted(jobs, key=lambda j: j.finished_at or "", reverse=True)
