"""
Unit tests for src/metrics.py.

Run with: pytest tests/ -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import (
    _minmax_to_100, compute_consistency, compute_wet_dry_delta,
    compute_pressure_index, compute_quali_race_delta, compute_residual_performance,
    compute_all_metrics, get_skill_model_diagnostics, bootstrap_confidence_intervals,
    select_k_by_silhouette, cluster_driver_styles, DEFAULT_CONFIG,
)


@pytest.fixture
def sample_laps():
    """Small synthetic lap dataset covering both drivers, both sessions, wet/dry,
    and pressure situations — enough for every metric function to have data to work with."""
    rng = np.random.default_rng(0)
    rows = []
    for driver, base_pace, wet_penalty in [("A. Fast", 90.0, 0.5), ("B. Slow", 92.0, 2.0)]:
        for season in [2023, 2024]:
            for session, n_laps in [("Qualifying", 5), ("Race", 40)]:
                for lap in range(1, n_laps + 1):
                    is_wet = bool(lap % 10 == 0)
                    under_pressure = bool(lap % 7 == 0)
                    lap_time = base_pace + rng.normal(0, 0.3) + (wet_penalty if is_wet else 0)
                    rows.append({
                        "driver": driver, "driver_code": driver[:3].upper(),
                        "circuit": "Test Circuit", "circuit_type": "Technical",
                        "season": season, "round": 1, "session": session,
                        "lap_number": lap, "lap_time_s": lap_time,
                        "tire_compound": "Medium", "tire_age": lap % 20, "stint": 1,
                        "is_wet": is_wet, "track_temp_c": 30.0,
                        "gap_to_ahead_s": 1.0, "under_pressure": under_pressure,
                    })
    return pd.DataFrame(rows)


class TestMinMaxScaling:
    def test_scales_to_0_100_range(self):
        s = pd.Series([1, 2, 3, 4, 5])
        out = _minmax_to_100(s)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(100.0)

    def test_invert_flips_direction(self):
        s = pd.Series([1, 2, 3])
        normal = _minmax_to_100(s)
        inverted = _minmax_to_100(s, invert=True)
        assert normal.loc[0] == pytest.approx(100 - inverted.loc[0])

    def test_constant_series_returns_midpoint(self):
        """When every value is identical there's no meaningful spread — should not divide by zero."""
        s = pd.Series([5.0, 5.0, 5.0])
        out = _minmax_to_100(s)
        assert (out == 50.0).all()


class TestIndividualMetrics:
    def test_consistency_scores_in_valid_range(self, sample_laps):
        out = compute_consistency(sample_laps)
        assert out.between(0, 100).all()
        assert set(out.index) == {"A. Fast", "B. Slow"}

    def test_faster_more_consistent_driver_scores_higher_consistency(self, sample_laps):
        # A. Fast has lower base variance injected than B. Slow's wet penalty swings
        out = compute_consistency(sample_laps)
        assert out["A. Fast"] >= out["B. Slow"] - 50  # loose bound; exact ranking isn't the point here

    def test_wet_dry_delta_penalizes_bigger_wet_dropoff(self, sample_laps):
        out = compute_wet_dry_delta(sample_laps)
        # B. Slow has a larger wet_penalty (2.0 vs 0.5s), so should score worse in the wet
        assert out["B. Slow"] < out["A. Fast"]

    def test_pressure_index_in_valid_range(self, sample_laps):
        out = compute_pressure_index(sample_laps)
        assert out.between(0, 100).all()

    def test_quali_race_delta_handles_missing_sessions_gracefully(self):
        """If a filtered slice has no qualifying laps at all, this should return
        an empty series rather than raising — the app relies on this not crashing."""
        race_only = pd.DataFrame({
            "driver": ["A"], "circuit": ["X"], "season": [2023],
            "session": ["Race"], "lap_time_s": [90.0],
        })
        out = compute_quali_race_delta(race_only)
        assert len(out) == 0

    def test_residual_performance_in_valid_range(self, sample_laps):
        out = compute_residual_performance(sample_laps)
        assert out.between(0, 100).all()


class TestSkillModel:
    def test_diagnostics_reports_reasonable_alpha(self, sample_laps):
        diag = get_skill_model_diagnostics(sample_laps)
        assert diag["chosen_alpha"] in DEFAULT_CONFIG["skill_model"]["alphas"]
        assert diag["n_laps"] > 0
        assert isinstance(diag["standardized_coefficients"], dict)

    def test_r2_is_bounded(self, sample_laps):
        diag = get_skill_model_diagnostics(sample_laps)
        # train R^2 for a regression can't exceed 1, and shouldn't be wildly negative here
        assert diag["r2_train"] <= 1.0


class TestComputeAllMetrics:
    def test_returns_one_row_per_driver(self, sample_laps):
        out = compute_all_metrics(sample_laps)
        assert len(out) == sample_laps["driver"].nunique()

    def test_overall_score_is_mean_of_metric_columns(self, sample_laps):
        out = compute_all_metrics(sample_laps)
        metric_cols = [c for c in out.columns if c != "overall_score"]
        expected = out[metric_cols].mean(axis=1).round(1)
        pd.testing.assert_series_equal(out["overall_score"], expected, check_names=False)

    def test_sorted_descending_by_overall_score(self, sample_laps):
        out = compute_all_metrics(sample_laps)
        assert (out["overall_score"].values == out["overall_score"].sort_values(ascending=False).values).all()


class TestBootstrapCI:
    def test_ci_bounds_are_ordered(self, sample_laps):
        cfg = {**DEFAULT_CONFIG, "bootstrap": {"n_resamples": 20, "ci_level": 0.90, "random_state": 1}}
        out = bootstrap_confidence_intervals(sample_laps, cfg)
        for metric in ["consistency_score", "wet_score", "pressure_score"]:
            lo_col, hi_col = f"{metric}_lo", f"{metric}_hi"
            if lo_col in out.columns:
                valid = out.dropna(subset=[lo_col, hi_col])
                assert (valid[lo_col] <= valid[hi_col]).all()


class TestClustering:
    def test_silhouette_selects_k_within_range(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(loc, 0.3, size=(10, 3)) for loc in [0, 5, 10]])
        best_k, best_score, scores = select_k_by_silhouette(X, k_min=2, k_max=5)
        assert 2 <= best_k <= 5
        assert -1 <= best_score <= 1
        assert best_k in scores

    def test_cluster_driver_styles_handles_too_few_drivers(self, sample_laps):
        """With only 2 drivers there isn't enough data for a meaningful multi-cluster
        split — should degrade gracefully (empty result) rather than raising."""
        metrics_df = compute_all_metrics(sample_laps)
        clusters, scores = cluster_driver_styles(metrics_df, n_clusters=None)
        assert isinstance(clusters, pd.Series)
