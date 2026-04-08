# Final-Year Project Demo Web Application (Local)

This repository now includes a local **FastAPI-based academic demo website** that executes your real fairness code pipeline with background jobs.

## What is included

- Real intermediate fairness analysis from `b.py`
- Real baseline model + standalone adversarial baseline from `c.py`
- Real Fair CVAE modes from `more strict .py`
  - `adv_only`
  - `no_adv`
  - `full`
- Real latent-space visualisation from `latent_vis.py`
- `a.py` is intentionally **not part of the website workflow**

## Website pages

- `/` Home
- `/results` Results
- `/demo` Interactive Demo
- `/about` About

## Interactive Demo workflow

1. Upload CSV file
2. Run intermediate fairness analysis (`b.py`)
3. Select exactly one model:
   - Baseline MLP
   - Standalone adversarial baseline
   - Fair CVAE adv_only
   - Fair CVAE no_adv
   - Fair CVAE full
4. Backend starts a background job
5. Frontend polls and displays:
   - status (`queued/running/completed/failed`)
   - progress percentage
   - progress text
   - live logs
6. Final results shown only after job completion
7. Optional latent visualisation generation for Fair CVAE runs

## Folder structure

```text
webapp/
  main.py                # FastAPI app + API routes + page routes
  job_manager.py         # Background job queue/state manager
  pipeline.py            # Real execution pipeline wrappers for b.py/c.py/more strict .py/latent_vis.py
  templates/
    base.html
    home.html
    demo.html
    results.html
    about.html
  static/
    style.css
    demo.js
runs/
  uploads/               # uploaded CSV files
  <job_id>/              # per-job outputs (CSVs, JSON, plots)
```

## Local run instructions

### 1) Install dependencies

```bash
pip install fastapi uvicorn jinja2 python-multipart pandas numpy scikit-learn torch matplotlib
```

> Optional for SHAP analysis in `b.py`:

```bash
pip install shap
```

### 2) Run the web app

```bash
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
```

### 3) Open

- http://127.0.0.1:8000/

## Notes

- This app uses **real code execution only**.
- No synthetic fallback mode is added for Fair CVAE.
- Long-running training is executed as background jobs via `ThreadPoolExecutor`.
- Live logs are captured from real stdout/stderr and streamed through polling.
- Output artifacts are stored per run in `runs/<job_id>/`.
