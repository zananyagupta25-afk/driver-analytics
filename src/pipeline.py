"""
Standalone training/evaluation pipeline for RaceIQ.

Runs the full metrics computation independent of the Streamlit UI — this is
the piece that should be run in CI, from a notebook, or on a cron job, not
just inside the app. It:
  1. Loads config + data
  2. Fits the skill-residual model and records its diagnostics
  3. Computes all driver metrics + bootstrap confidence intervals
  4. Selects cluster count via silhouette score
  5. Logs the run (params + metrics) to MLflow if installed, and always to a
     local JSON run log as a zero-dependency fallback

Usage:
    python -m src.pipeline
    python -m src.pipeline --data data/laps.csv --config config.yaml
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from metrics import (
    load_config, compute_all_metrics, get_skill_model_diagnostics,
    bootstrap_confidence_intervals, cluster_driver_styles, name_clusters,
)

RUN_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "runs", "run_log.json")


def _log_run(params: dict, metrics_summary: dict):
    """Logs to MLflow if available; always logs to a local JSON file too so the
    pipeline has zero required dependencies for basic reproducibility tracking."""
    os.makedirs(os.path.dirname(RUN_LOG_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics": metrics_summary,
    }
    existing = []
    if os.path.exists(RUN_LOG_PATH):
        try:
            with open(RUN_LOG_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(entry)
    with open(RUN_LOG_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    try:
        import mlflow
        mlflow.set_experiment("raceiq")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics_summary.items() if isinstance(v, (int, float))})
    except ImportError:
        print("[pipeline] mlflow not installed — logged to runs/run_log.json only. "
              "`pip install mlflow` and re-run for full experiment tracking.")


def run(data_path: str, config_path: str = None):
    cfg = load_config(config_path)
    df = pd.read_csv(data_path)

    print(f"[pipeline] loaded {len(df):,} laps, {df['driver'].nunique()} drivers")

    diagnostics = get_skill_model_diagnostics(df, cfg)
    print(f"[pipeline] skill model: RidgeCV alpha={diagnostics['chosen_alpha']:.3g}, "
          f"train R²={diagnostics['r2_train']:.3f}, n_laps={diagnostics['n_laps']}")

    metrics_df = compute_all_metrics(df, cfg)
    print(f"[pipeline] computed metrics for {len(metrics_df)} drivers")
    print(metrics_df[["overall_score"]].to_string())

    print(f"[pipeline] running bootstrap ({cfg['bootstrap']['n_resamples']} resamples)...")
    ci_df = bootstrap_confidence_intervals(df, cfg)

    clusters, sil_scores = cluster_driver_styles(metrics_df, n_clusters=None, cfg=cfg)
    labels = name_clusters(metrics_df, clusters) if len(clusters) else {}
    best_k = clusters.nunique() if len(clusters) else None
    print(f"[pipeline] cluster count selected by silhouette score: k={best_k} "
          f"(scores by k: {sil_scores})")

    params = {
        "n_laps": len(df),
        "n_drivers": int(df["driver"].nunique()),
        "ridge_alpha": diagnostics["chosen_alpha"],
        "ridge_cv_folds": cfg["skill_model"]["cv_folds"],
        "bootstrap_resamples": cfg["bootstrap"]["n_resamples"],
        "bootstrap_ci_level": cfg["bootstrap"]["ci_level"],
        "cluster_k_selected": best_k,
    }
    metrics_summary = {
        "skill_model_r2_train": diagnostics["r2_train"],
        "mean_overall_score": float(metrics_df["overall_score"].mean()),
        "best_silhouette_score": max(sil_scores.values()) if sil_scores else None,
    }
    _log_run(params, metrics_summary)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(out_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(out_dir, "latest_metrics.csv"))
    ci_df.to_csv(os.path.join(out_dir, "latest_confidence_intervals.csv"))
    with open(os.path.join(out_dir, "latest_skill_model_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"[pipeline] artifacts written to {os.path.abspath(out_dir)}/")

    return metrics_df, ci_df, diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RaceIQ metrics pipeline standalone.")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "laps.csv"))
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    args = parser.parse_args()
    run(args.data, args.config if os.path.exists(args.config) else None)
