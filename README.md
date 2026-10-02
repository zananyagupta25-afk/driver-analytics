<div align="center">

# 🏎️ RaceIQ

**A motorsport analytics dashboard that scores drivers, shows how confident those scores are, and groups drivers by driving profile.**

</div>

---

## 📌 Overview

Ranking racing drivers by raw results can be misleading, because results depend on the car, the circuit, and luck. **RaceIQ** uses machine learning to estimate a **skill score** for each driver and shows how **confident** that estimate is, so a small gap between two drivers is not mistaken for a real difference.

The project goes beyond a basic model:

- A **regularised regression model (RidgeCV)** estimates driver skill and avoids overfitting.
- **Bootstrap confidence intervals** give every score an uncertainty band. A narrow band means the ranking is well supported by the data. A wide band means small differences between nearby drivers should not be over-read.
- **KMeans clustering**, with the number of clusters chosen by **silhouette score**, groups drivers into driving profiles.
- **MLflow** tracks every experiment, **pytest** tests the code, and a **modular structure** keeps the pipeline clean and reproducible.
- Everything is presented in an interactive **Streamlit dashboard**, with a **model card** documenting how the model works and where it falls short.

---

## ✨ Key Features

- 🎯 **Driver skill score** with explanations of how it is calculated and how confident the model is
- 📏 **Confidence intervals** shown as error bars on the skill score chart
- 🧩 **Driver clustering** with data-driven cluster count (silhouette score)
- 📊 **Interactive Plotly charts** inside a Streamlit dashboard
- 🧪 **Experiment tracking** with MLflow to compare runs, parameters, and metrics
- ✅ **Unit tests** with pytest for a reliable pipeline
- ⚙️ **Config-driven** settings through `config.yaml`
- 📝 **Model card** (`MODEL_CARD.md`) describing the model's purpose, data, and limitations

---

## 🏗️ How It Works

```
Data → Cleaning & Features → RidgeCV Skill Model → Bootstrap Confidence Intervals
                         ↘ KMeans (silhouette-tuned) → Driver Clusters
                                       ↓
                      MLflow Tracking → Streamlit Dashboard
```

1. **Data:** driver and race data is loaded from the `data/` folder and cleaned.
2. **Features:** performance features are built for each driver.
3. **Skill model:** RidgeCV fits the model, choosing the best regularisation strength by cross-validation.
4. **Uncertainty:** bootstrap resampling produces a lower and upper bound (confidence interval) for each skill score.
5. **Clustering:** KMeans groups drivers, and silhouette score picks the best number of clusters.
6. **Tracking:** parameters and metrics for each run are logged to MLflow.
7. **Dashboard:** `app.py` shows the scores, confidence intervals, and clusters in an interactive interface.

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python |
| ML | scikit-learn (RidgeCV, KMeans), NumPy, Pandas |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Testing | pytest |
| Configuration | YAML |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```
driver-analytics/
├── app.py                 # Streamlit dashboard
├── config.yaml            # Project settings
├── data/                  # Datasets
├── src/                   # Pipeline code (features, models, clustering)
├── tests/                 # pytest unit tests
├── MODEL_CARD.md          # Model documentation and limitations
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Development and testing dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation
```bash
git clone https://github.com/zananyagupta25-afk/driver-analytics.git
cd driver-analytics
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Run the dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Run the tests
```bash
pip install -r requirements-dev.txt
pytest
```

### View experiments
```bash
mlflow ui
```
Open `http://localhost:5000`.

---

## 📖 Reading the Dashboard

- **Skill score chart:** each bar is a driver's estimated skill, and the error bars show the confidence interval.
- **Narrow error bar:** the ranking for that driver is well supported by the data.
- **Wide error bar:** the estimate is uncertain, so avoid over-reading small differences between nearby drivers.
- **Clusters:** drivers in the same cluster share a similar performance profile.

---

## 🧗 Challenges and Learnings

- **Avoiding overfitting** by using RidgeCV with cross-validation instead of plain linear regression.
- **Being honest about uncertainty** by reporting bootstrap confidence intervals instead of single numbers.
- **Choosing the number of clusters objectively** with silhouette score rather than guessing.
- **Making the work reproducible** with MLflow tracking, pytest tests, a config file, and a modular code structure.
- **Documenting limitations** in a model card so results are used responsibly.

---

## 🔮 Future Improvements

- [ ] Add more seasons and circuits
- [ ] Compare gradient boosting models against Ridge in MLflow
- [ ] Deploy the dashboard online
- [ ] Add per-race and per-lap analysis

---

## 👩‍💻 Author

**Ananya Gupta**, B.Tech AI & ML
[LinkedIn](https://www.linkedin.com/in/ananya-gupta-8b7023369) · [GitHub](https://github.com/zananyagupta25-afk)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
