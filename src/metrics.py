"""
Core analytics engine: turns raw lap data into driver performance metrics.

All scores are normalized 0-100 (higher = better) unless noted otherwise,
so they're directly comparable/plottable on the same radar chart.

Design notes (why these choices, for anyone reviewing the code):
- The "skill" metric uses RidgeCV rather than plain LinearRegression. Track/tire/
  weather dummy variables are correlated with each other (e.g. hard tires are more
  common at high-degradation circuits), so an unregularized fit can assign unstable,
  overfit coefficients. Ridge shrinks them toward zero and cross-validation picks
  the regularization strength that generalizes best, rather than guessing one value.
- Bootstrap resampling is used for confidence intervals instead of relying on a
  closed-form standard error, because several of these metrics (consistency ratio,
  wet/dry delta) aren't simple means and don't have a clean analytic variance formula.
  Resampling laps with replacement and recomputing the metric directly measures how
  much it would plausibly move with different laps.
- Cluster count (k) is chosen by maximizing silhouette score across a small range
  rather than fixing k arbitrarily, so "3 driver archetypes" is a measured choice,
  not a guess.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

DEFAULT_CONFIG = {
    "skill_model": {"alphas": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0], "cv_folds": 5},
    "bootstrap": {"n_resamples": 500, "ci_level": 0.90, "random_state": 42},
    "clustering": {"k_min": 2, "k_max": 6, "auto_select_k": True, "random_state": 42, "n_init": 10},
    "data": {"outlier_quantile_low": 0.03, "outlier_quantile_high": 0.97},
}


def load_config(path: str = None) -> dict:
    """Loads config.yaml if present, otherwise falls back to sane defaults so
    the pipeline never hard-fails just because the config file is missing."""
    if path is None:
        return DEFAULT_CONFIG
    try:
        import yaml
        with open(path) as f:
            user_cfg = yaml.safe_load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(user_cfg or {})
        return cfg
    except Exception:
        return DEFAULT_CONFIG


def _minmax_to_100(s: pd.Series, invert: bool = False) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    scaled = (s - lo) / (hi - lo) * 100
    return 100 - scaled if invert else scaled


def compute_consistency(df: pd.DataFrame, cfg: dict = DEFAULT_CONFIG) -> pd.Series:
    """Coefficient of variation of race lap times (excluding outliers/pit laps), lower = more consistent."""
    qlo = cfg["data"]["outlier_quantile_low"]
    qhi = cfg["data"]["outlier_quantile_high"]
    race = df[df["session"] == "Race"]
    q1 = race.groupby("driver")["lap_time_s"].transform(lambda x: x.quantile(qlo))
    q99 = race.groupby("driver")["lap_time_s"].transform(lambda x: x.quantile(qhi))
    clean = race[(race["lap_time_s"] >= q1) & (race["lap_time_s"] <= q99)]
    cv = clean.groupby("driver")["lap_time_s"].agg(lambda x: x.std() / x.mean())
    return _minmax_to_100(cv, invert=True).rename("consistency_score")


def compute_wet_dry_delta(df: pd.DataFrame, cfg: dict = DEFAULT_CONFIG) -> pd.Series:
    """% pace change going from dry to wet vs field average — smaller drop-off = better wet performer."""
    race = df[df["session"] == "Race"]
    grp = race.groupby(["driver", "is_wet"])["lap_time_s"].median().unstack()
    if True not in grp.columns or False not in grp.columns:
        return pd.Series(dtype=float, name="wet_score")
    pct_change = (grp[True] - grp[False]) / grp[False] * 100
    return _minmax_to_100(pct_change, invert=True).rename("wet_score")


def compute_pressure_index(df: pd.DataFrame, cfg: dict = DEFAULT_CONFIG) -> pd.Series:
    """Pace delta when under pressure (gap <1s, closing stages) vs that driver's own baseline pace."""
    race = df[df["session"] == "Race"]
    baseline = race.groupby("driver")["lap_time_s"].median()
    pressure_laps = race[race["under_pressure"]]
    if pressure_laps.empty:
        return pd.Series(dtype=float, name="pressure_score")
    pressure_pace = pressure_laps.groupby("driver")["lap_time_s"].median()
    delta = (pressure_pace - baseline) / baseline * 100  # negative = faster under pressure
    return _minmax_to_100(delta, invert=True).rename("pressure_score")


def compute_quali_race_delta(df: pd.DataFrame, cfg: dict = DEFAULT_CONFIG) -> pd.Series:
    """How well quali pace converts to race pace (rank correlation), higher = better converter."""
    quali = df[df["session"] == "Qualifying"].groupby(["driver", "circuit", "season"])["lap_time_s"].min()
    race = df[df["session"] == "Race"].groupby(["driver", "circuit", "season"])["lap_time_s"].median()
    merged = pd.concat([quali.rename("q"), race.rename("r")], axis=1).dropna()
    merged["ratio"] = merged["r"] / merged["q"]
    conv = merged.groupby("driver")["ratio"].mean()
    return _minmax_to_100(conv, invert=True).rename("conversion_score")


def _fit_skill_model(race: pd.DataFrame, cfg: dict):
    """Fits the RidgeCV skill-residual model and returns (fitted_model, X, y, feature_cols)."""
    race = pd.get_dummies(race, columns=["circuit_type", "tire_compound"], drop_first=True)
    feature_cols = ["tire_age", "track_temp_c"] + \
        [c for c in race.columns if c.startswith("circuit_type_") or c.startswith("tire_compound_")]
    X = race[feature_cols].astype(float)
    y = race["lap_time_s"]
    alphas = cfg["skill_model"]["alphas"]
    cv = cfg["skill_model"]["cv_folds"]
    model = RidgeCV(alphas=alphas, cv=cv).fit(X, y)
    return model, X, y, feature_cols


def compute_residual_performance(df: pd.DataFrame, cfg: dict = DEFAULT_CONFIG) -> pd.Series:
    """Regress lap time on track/tire/weather context with cross-validated Ridge regression;
    residual = skill component after removing conditions. Ridge (vs. plain OLS) is used because
    the dummy-encoded context features are correlated, which makes unregularized coefficients
    unstable — RidgeCV picks the shrinkage strength via cross-validation instead of a fixed guess."""
    race = df[df["session"] == "Race"].copy()
    model, X, y, feature_cols = _fit_skill_model(race, cfg)
    race["residual"] = y.values - model.predict(X)
    driver_residual = race.groupby("driver")["residual"].mean()
    return _minmax_to_100(driver_residual, invert=True).rename("skill_score")


def get_skill_model_diagnostics(df: pd.DataFrame, cfg: dict = DEFAULT_CONFIG) -> dict:
    """Returns model diagnostics (chosen alpha, train R², standardized coefficients) for the
    skill-residual model — used for the explainability panel in the app and for run logging."""
    race = df[df["session"] == "Race"].copy()
    model, X, y, feature_cols = _fit_skill_model(race, cfg)
    # standardize for coefficient comparability (raw scales differ, e.g. tire_age vs track_temp_c)
    X_std = (X - X.mean()) / X.std().replace(0, 1)
    std_model = RidgeCV(alphas=cfg["skill_model"]["alphas"], cv=cfg["skill_model"]["cv_folds"]).fit(X_std, y)
    coefs = pd.Series(std_model.coef_, index=feature_cols).sort_values(key=abs, ascending=False)
    return {
        "chosen_alpha": float(model.alpha_),
        "r2_train": float(model.score(X, y)),
        "n_laps": int(len(X)),
        "standardized_coefficients": coefs.round(3).to_dict(),
    }


def compute_all_metrics(df: pd.DataFrame, cfg: dict = DEFAULT_CONFIG) -> pd.DataFrame:
    metrics = [
        compute_consistency(df, cfg),
        compute_wet_dry_delta(df, cfg),
        compute_pressure_index(df, cfg),
        compute_quali_race_delta(df, cfg),
        compute_residual_performance(df, cfg),
    ]
    out = pd.concat(metrics, axis=1)
    out["overall_score"] = out.mean(axis=1)
    return out.sort_values("overall_score", ascending=False).round(1)


# ---------------------------------------------------------------------------
# BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------------------------
_METRIC_FUNCS = {
    "consistency_score": compute_consistency,
    "wet_score": compute_wet_dry_delta,
    "pressure_score": compute_pressure_index,
    "conversion_score": compute_quali_race_delta,
    "skill_score": compute_residual_performance,
}


def bootstrap_confidence_intervals(df: pd.DataFrame, cfg: dict = DEFAULT_CONFIG) -> pd.DataFrame:
    """Resamples laps (with replacement, per driver/session) to estimate a confidence interval
    for every metric. This quantifies how much each score could plausibly shift given natural
    lap-to-lap variance, which matters most here since the underlying dataset is small.

    Returns a DataFrame indexed by driver with `<metric>_lo` / `<metric>_hi` columns.
    """
    n = cfg["bootstrap"]["n_resamples"]
    ci = cfg["bootstrap"]["ci_level"]
    rng = np.random.default_rng(cfg["bootstrap"]["random_state"])
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2

    drivers = df["driver"].unique()
    samples = {metric: {d: [] for d in drivers} for metric in _METRIC_FUNCS}

    for _ in range(n):
        # resample laps within each (driver, session) group independently, preserving each
        # driver's sample size so the resample doesn't just favor high-lap-count drivers
        resampled = (
            df.groupby(["driver", "session"], group_keys=False)
              .apply(lambda g: g.sample(n=len(g), replace=True, random_state=int(rng.integers(1_000_000_000))))
        )
        for metric_name, func in _METRIC_FUNCS.items():
            try:
                result = func(resampled, cfg)
            except Exception:
                continue
            for d in drivers:
                if d in result.index and pd.notna(result[d]):
                    samples[metric_name][d].append(result[d])

    rows = {}
    for metric_name, per_driver in samples.items():
        for d, vals in per_driver.items():
            if len(vals) < 10:
                continue
            rows.setdefault(d, {})
            rows[d][f"{metric_name}_lo"] = float(np.quantile(vals, lo_q))
            rows[d][f"{metric_name}_hi"] = float(np.quantile(vals, hi_q))
    return pd.DataFrame.from_dict(rows, orient="index").round(1)


# ---------------------------------------------------------------------------
# CLUSTERING
# ---------------------------------------------------------------------------
def select_k_by_silhouette(X: np.ndarray, k_min: int, k_max: int, random_state: int = 42, n_init: int = 10):
    """Fits KMeans for each k in [k_min, k_max] and returns the k with the highest silhouette
    score, plus the per-k scores — so cluster count is a measured choice, not an arbitrary default."""
    best_k, best_score, scores = k_min, -1.0, {}
    k_max = min(k_max, len(X) - 1)
    for k in range(k_min, max(k_min, k_max) + 1):
        if k < 2 or k >= len(X):
            continue
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = float(score)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score, scores


def cluster_driver_styles(metrics_df: pd.DataFrame, n_clusters: int = None, cfg: dict = DEFAULT_CONFIG):
    """Groups drivers into style archetypes based on their metric profile.
    If n_clusters is None, picks k via silhouette score instead of an arbitrary default."""
    cols = ["consistency_score", "wet_score", "pressure_score", "conversion_score", "skill_score"]
    valid = metrics_df.dropna(subset=cols)
    c = cfg["clustering"]
    scores = {}
    if n_clusters is None:
        if len(valid) < c["k_min"] + 1:
            return pd.Series(dtype=int), scores
        X = StandardScaler().fit_transform(valid[cols])
        n_clusters, _, scores = select_k_by_silhouette(
            X, c["k_min"], min(c["k_max"], len(valid) - 1), c["random_state"], c["n_init"]
        )
    if len(valid) < n_clusters:
        return pd.Series(dtype=int), scores
    X = StandardScaler().fit_transform(valid[cols])
    labels = KMeans(n_clusters=n_clusters, random_state=c["random_state"], n_init=c["n_init"]).fit_predict(X)
    return pd.Series(labels, index=valid.index, name="cluster"), scores


def name_clusters(metrics_df: pd.DataFrame, clusters: pd.Series) -> dict:
    """Assigns human-readable archetype labels to numeric cluster IDs based on their averages."""
    merged = metrics_df.join(clusters)
    names = {}
    for c in merged["cluster"].dropna().unique():
        sub = merged[merged["cluster"] == c]
        avg = sub[["consistency_score", "wet_score", "pressure_score", "conversion_score", "skill_score"]].mean()
        top = avg.idxmax()
        label_map = {
            "consistency_score": "The Metronome",
            "wet_score": "Rain Specialist",
            "pressure_score": "Clutch Performer",
            "conversion_score": "Race-Day Converter",
            "skill_score": "Raw Pace Leader",
        }
        names[c] = label_map.get(top, f"Cluster {int(c)}")
    return names
