from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

import b
import c
from latent_vis import run_latent_visualisation

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, str], None]


@dataclass
class PipelineResult:
    summary: dict
    artifact_paths: dict


def _load_fair_cvae_module():
    module_path = Path("more strict .py").resolve()
    spec = importlib.util.spec_from_file_location("fair_cvae_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Fair CVAE module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@contextlib.contextmanager
def _capture_stdout(log_fn: LogFn):
    class _Writer(io.StringIO):
        def write(self, s):
            super().write(s)
            if s.strip():
                for line in s.rstrip().splitlines():
                    log_fn(line)
            return len(s)

    writer = _Writer()
    with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
        yield


def run_intermediate_analysis(csv_path: Path, out_dir: Path, log: LogFn, progress: ProgressFn) -> dict:
    progress(5, "Running intermediate fairness analysis from b.py")
    df = b.load_data(str(csv_path))
    df = b.add_proxy_qualified(df)

    fairness_df = b.compute_fairness_metrics(df)
    odds_df = b.compute_odds_ratios(df)
    shap_results = b.run_shap_analysis(df)

    _save_df(fairness_df, out_dir / "fairness_metrics.csv")
    _save_df(odds_df, out_dir / "odds_ratios.csv")

    summary = {
        "rows": int(len(df)),
        "groups": sorted(df["Group"].unique().tolist()),
        "fairness_file": "fairness_metrics.csv",
        "odds_file": "odds_ratios.csv",
    }

    if shap_results:
        _save_df(shap_results["results"], out_dir / "shap_bias_analysis.csv")
        shap_results["summary"].to_csv(out_dir / "shap_feature_summary.csv")
        summary["shap_results_file"] = "shap_bias_analysis.csv"
        summary["shap_summary_file"] = "shap_feature_summary.csv"

    progress(30, "Intermediate analysis complete")
    return summary


def run_selected_model(
    csv_path: Path,
    out_dir: Path,
    model_name: str,
    include_latent_vis: bool,
    log: LogFn,
    progress: ProgressFn,
) -> PipelineResult:
    model_name = model_name.strip().lower()
    artifact_paths: dict[str, str] = {}

    with _capture_stdout(log):
        analysis_summary = run_intermediate_analysis(csv_path, out_dir, log, progress)

        progress(35, f"Preparing model run: {model_name}")

        if model_name in {"baseline_mlp", "adversarial_baseline"}:
            data = c.load_and_prepare_data(str(csv_path))
            input_dim = data["input_dim"]

            if model_name == "baseline_mlp":
                progress(45, "Training baseline MLP")
                model = c.SimpleClassifier(input_dim, hidden_dim=128)
                c.train_baseline_model(model, data["X_train"], data["y_train"], epochs=150, batch_size=256, lr=0.001)
                fairness_df, pred = c.evaluate_model(model, data["X_test"], data["df_test"], model_type="simple")
                model_label = "Baseline MLP"
            else:
                progress(45, "Training standalone adversarial baseline")
                model = c.AdversarialDebiasingGRL(input_dim, hidden_dim=64, num_groups=data["n_genders"])
                history = c.train_adversarial_model_grl(
                    model,
                    data["X_train"], data["y_train"], data["g_gen_train"],
                    data["X_val"], data["y_val"], data["g_gen_val"],
                    epochs=200,
                    verbose=True,
                )
                pd.DataFrame(history).to_csv(out_dir / "adversarial_history.csv", index=False)
                artifact_paths["adversarial_history"] = "adversarial_history.csv"
                fairness_df, pred = c.evaluate_model(model, data["X_test"], data["df_test"], model_type="adversarial")
                model_label = "Standalone Adversarial Baseline"

            progress(80, "Computing model metrics")
            y_true = data["y_test"].numpy().flatten()
            acc = float((pred == y_true).mean())
            _save_df(fairness_df, out_dir / "model_fairness.csv")
            artifact_paths["model_fairness"] = "model_fairness.csv"

            result = {
                "model": model_label,
                "accuracy": acc,
                "min_di": float(fairness_df["DI"].min()),
            }

        elif model_name in {"fair_cvae_adv_only", "fair_cvae_no_adv", "fair_cvae_full"}:
            fair = _load_fair_cvae_module()
            data = fair.load_and_prepare_data(str(csv_path))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mode_map = {
                "fair_cvae_adv_only": "adv_only",
                "fair_cvae_no_adv": "no_adv",
                "fair_cvae_full": "full",
            }
            mode = mode_map[model_name]

            progress(45, f"Training Fair CVAE mode={mode}")
            cvae_model = fair.FairCVAE_v4(
                x_dim=data["input_dim"],
                n_sensitive=data["n_genders"],
                z_dim=64,
                hidden_dim=256,
                n_sensitive_directions=3,
            ).to(device)

            history = fair.train_fair_cvae_v4(
                cvae_model,
                data,
                epochs=350,
                batch_size=256,
                lr_main=1e-3,
                lr_adv=2e-3,
                adv_steps=5,
                lambda_hsic=50.0,
                lambda_adv=2.0,
                alpha_max=8.0,
                adv_reset_every=40,
                projection_update_every=20,
                device=device,
                verbose=True,
                mode=mode,
            )
            pd.DataFrame(history).to_csv(out_dir / f"fair_cvae_{mode}_history.csv", index=False)
            artifact_paths["training_history"] = f"fair_cvae_{mode}_history.csv"

            progress(78, "Evaluating Fair CVAE")
            eval_res = fair.evaluate_model(cvae_model, data, device=device)
            _save_df(eval_res["fairness"], out_dir / "model_fairness.csv")
            artifact_paths["model_fairness"] = "model_fairness.csv"

            result = {
                "model": f"Fair CVAE ({mode})",
                "accuracy": float(eval_res["accuracy"]),
                "f1": float(eval_res["f1"]),
                "min_di": float(eval_res["fairness"]["DI"].min()),
            }

            if include_latent_vis:
                progress(88, "Generating latent-space visualisations")
                baseline = fair.SimpleClassifier(data["input_dim"]).to(device)
                fair.train_baseline(baseline, data, epochs=150, batch_size=256, lr=1e-3, device=device)
                latent_dir = out_dir / "latent_vis"
                latent_summary = run_latent_visualisation(
                    baseline_model=baseline,
                    cvae_model=cvae_model,
                    data=data,
                    device=device,
                    output_dir=str(latent_dir),
                )
                result["latent_summary"] = latent_summary
                artifact_paths["latent_vis_dir"] = "latent_vis"
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    progress(98, "Finalising output files")
    summary = {
        "analysis": analysis_summary,
        "result": result,
    }

    summary_path = out_dir / "result_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifact_paths["summary"] = "result_summary.json"
    progress(100, "Completed")
    return PipelineResult(summary=summary, artifact_paths=artifact_paths)


def run_pipeline_job(
    csv_path: Path,
    out_dir: Path,
    model_name: str,
    include_latent_vis: bool,
    log: LogFn,
    progress: ProgressFn,
):
    try:
        return run_selected_model(csv_path, out_dir, model_name, include_latent_vis, log, progress)
    except Exception as exc:
        log("Pipeline failed:\n" + traceback.format_exc())
        raise exc
