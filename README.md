# RaceIQ — Driver Consistency & Pressure Performance Analytics

Quantifies driver performance consistency and "clutch" execution across circuits
and conditions using lap-by-lap timing data — going beyond raw results to isolate
skill from car/track/weather context.

## What it measures
- **Consistency Score** — inverse coefficient of variation of race lap times
- **Wet-Weather Score** — pace drop-off from dry to wet relative to the field
- **Pressure Index** — pace when running with <1s gap in the closing stages of a race
- **Quali→Race Conversion** — how efficiently qualifying pace translates to race pace
- **Skill Score** — residual pace left over after a regression removes tire age,
  track temperature, circuit type, and compound effects
- **Style Clusters** — KMeans grouping of drivers into archetypes (e.g. "Rain
  Specialist", "The Metronome", "Clutch Performer") based on their metric profile

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app ships with a synthetic dataset (`data/laps.csv`) so it runs immediately
with realistic numbers — useful for demos and screenshots.

## Using real data
Sandbox environments can't reach live timing servers, so generate real data on
your own machine:
```bash
pip install fastf1
python data/fetch_real_data.py
```
This overwrites `data/laps.csv` in the same schema — the app needs no changes.
Note: you'll want to fill in `circuit_type` (Street/High-Speed/Technical) and
compute `gap_to_ahead_s` / `under_pressure` from FastF1's timing data or the
`Laps` object's gap columns for full accuracy.

## ML engineering behind the scores
- **Model**: the skill score comes from a `RidgeCV` regression (5-fold CV over
  a fixed alpha grid, `config.yaml`), not plain OLS — the context features are
  correlated, so regularization keeps coefficients stable. Full rationale and
  evaluation in [`MODEL_CARD.md`](MODEL_CARD.md).
- **Uncertainty**: every score ships with a 90% bootstrap confidence interval
  (500 resamples of the underlying laps) — shown as error bars in-app rather
  than presenting a bare number as if it were exact.
- **Cluster count**: the driver-archetype clustering picks k by maximizing
  silhouette score across k=2–6 by default, instead of a hardcoded guess.
- **Pipeline vs. UI**: `src/pipeline.py` runs the full metrics computation,
  model fit, and evaluation standalone — no Streamlit required — and logs
  each run's parameters/metrics to `runs/run_log.json` (and to MLflow if
  installed). This is the piece to point to when explaining "how would you
  run this outside the app."
- **Tests**: `pytest tests/` covers the scoring functions, the Ridge model
  diagnostics, bootstrap CI ordering, and clustering edge cases (17 tests).

## Project structure
```
driver_analytics/
├── app.py                       # Streamlit dashboard (UI)
├── config.yaml                  # model/pipeline hyperparameters (not hardcoded)
├── src/
│   ├── metrics.py                # analytics engine: scoring, Ridge model, bootstrap CI, clustering
│   └── pipeline.py               # standalone train/eval runner (python -m src.pipeline)
├── tests/
│   └── test_metrics.py           # pytest unit tests
├── data/
│   ├── generate_sample_data.py   # synthetic data generator
│   ├── fetch_real_data.py        # real data puller (run locally)
│   └── laps.csv                  # bundled synthetic dataset
├── runs/                         # pipeline run artifacts (git-ignored)
├── MODEL_CARD.md                 # model documentation (data, evaluation, limitations)
├── requirements.txt              # app-only deps (lean, for Streamlit Cloud)
└── requirements-dev.txt          # + pytest, mlflow for local dev/pipeline
```

## Running the pipeline & tests
```bash
pip install -r requirements-dev.txt
python -m src.pipeline      # full pipeline outside the UI, writes runs/latest_*.csv
pytest tests/ -v            # unit test suite
```

## Ideas to extend this further
1. **Predictive layer** — train on conditions to forecast an upcoming session's
   likely pressure/wet performance ranking; frame as a forecasting problem.
2. **PDF driver report cards** — one-click export of a driver's full profile
   (I can build this with the pdf skill if you want it).
3. **Telemetry sector breakdown** — if you pull FastF1 telemetry (not just
   laps), you can localize *where* on track a driver gains/loses time.
4. **CI workflow** — a GitHub Actions job that runs `pytest` on every push,
   so the README's test badge is actually enforced.
5. **SHAP values** on the residual regression as an alternative to the
   standardized-coefficient view already in the app, for a deeper drill-down.
