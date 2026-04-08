from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .job_manager import JobManager

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
UPLOAD_ROOT = RUNS_ROOT / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Fairness Demo Web App")
app.mount("/static", StaticFiles(directory=str(APP_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))
manager = JobManager(RUNS_ROOT)

MODEL_OPTIONS = {
    "baseline_mlp": "Baseline MLP",
    "adversarial_baseline": "Standalone adversarial baseline",
    "fair_cvae_adv_only": "Fair CVAE adv_only",
    "fair_cvae_no_adv": "Fair CVAE no_adv",
    "fair_cvae_full": "Fair CVAE full",
}


def _convert_numpy_types(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def _job_to_dict(job):
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "progress_text": job.progress_text,
        "logs": job.logs[-300:],
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "summary": _convert_numpy_types(job.summary) if job.summary else None,
        "artifacts": job.artifacts,
    }
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"request": request, "models": MODEL_OPTIONS},
    )


@app.get("/demo", response_class=HTMLResponse)
def demo(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="demo.html",
        context={"request": request, "models": MODEL_OPTIONS},
    )


@app.get("/results", response_class=HTMLResponse)
def results_page(request: Request):
    jobs = manager.latest_completed()
    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={"request": request, "jobs": jobs},
    )


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={"request": request},
    )
@app.post("/api/jobs/start")
async def start_job(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    seed: int = Form(42),
    include_latent_vis: bool = Form(False),
):
    if model_name not in MODEL_OPTIONS:
        raise HTTPException(status_code=400, detail="Invalid model_name")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    upload_path = UPLOAD_ROOT / file.filename
    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    job = manager.create_job(
        csv_path=upload_path,
        model_name=model_name,
        seed=seed,
        include_latent_vis=include_latent_vis,
    )
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(_job_to_dict(job))


@app.get("/api/jobs/{job_id}/artifact/{artifact_name}")
def get_artifact(job_id: str, artifact_name: str):
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    rel = job.artifacts.get(artifact_name)
    if not rel:
        raise HTTPException(status_code=404, detail="Artifact not found")
    base = Path(job.output_dir)
    file_path = base / rel
    if file_path.is_dir():
        raise HTTPException(status_code=400, detail="Artifact is a directory")
    return FileResponse(file_path)


@app.get("/api/jobs/{job_id}/summary")
def get_summary(job_id: str):
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.summary or {})


@app.get("/api/jobs/{job_id}/latent-images")
def get_latent_images(job_id: str):
    job = manager.get_job(job_id)
    if not job or not job.output_dir:
        raise HTTPException(status_code=404, detail="Job not found")
    latent_dir = Path(job.output_dir) / "latent_vis"
    if not latent_dir.exists():
        return []
    images = []
    for p in sorted(latent_dir.glob("*.png")):
        images.append({"name": p.name, "url": f"/runs/{job_id}/latent_vis/{p.name}"})
    return images


app.mount("/runs", StaticFiles(directory=str(RUNS_ROOT)), name="runs")
