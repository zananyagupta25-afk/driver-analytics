"""
Generates a realistic synthetic lap-by-lap dataset so the dashboard runs
standalone without needing live data. Swap this for fetch_fastf1_data.py
output (same schema) to use real motorsport timing data.

Schema (laps.csv):
    driver, circuit, circuit_type, season, round, session,
    lap_number, lap_time_s, tire_compound, tire_age, is_wet,
    track_temp_c, gap_to_ahead_s, position, stint
"""
import numpy as np
import pandas as pd
import os

rng = np.random.default_rng(42)

DRIVERS = [
    {"name": "N. Aster",   "code": "AST", "base_pace": 0.15, "consistency": 0.08, "wet_skill": 0.30, "clutch": 0.25},
    {"name": "R. Voss",    "code": "VOS", "base_pace": 0.05, "consistency": 0.20, "wet_skill": -0.10, "clutch": -0.15},
    {"name": "K. Halden",  "code": "HAL", "base_pace": 0.00, "consistency": 0.05, "wet_skill": 0.10, "clutch": 0.40},
    {"name": "M. Rienne",  "code": "RIE", "base_pace": -0.10, "consistency": 0.35, "wet_skill": -0.30, "clutch": -0.30},
    {"name": "T. Osei",    "code": "OSE", "base_pace": 0.10, "consistency": 0.12, "wet_skill": 0.45, "clutch": 0.10},
    {"name": "P. Lindqvist","code": "LIN","base_pace": -0.05, "consistency": 0.10, "wet_skill": -0.05, "clutch": 0.05},
    {"name": "J. Ferraro", "code": "FER", "base_pace": 0.20, "consistency": 0.15, "wet_skill": -0.20, "clutch": -0.05},
    {"name": "D. Kessler",  "code": "KES", "base_pace": -0.15, "consistency": 0.25, "wet_skill": 0.05, "clutch": -0.20},
]

CIRCUITS = [
    {"name": "Meridian Street Circuit", "type": "Street",     "base_lap": 92.0,  "deg_rate": 0.06},
    {"name": "Sable Coast Circuit",     "type": "High-Speed", "base_lap": 78.5,  "deg_rate": 0.09},
    {"name": "Karst Technical Park",    "type": "Technical",  "base_lap": 105.2, "deg_rate": 0.04},
    {"name": "Northgate Ring",          "type": "High-Speed", "base_lap": 84.0,  "deg_rate": 0.10},
    {"name": "Valdera Circuit",         "type": "Technical",  "base_lap": 98.6,  "deg_rate": 0.05},
    {"name": "Port Alden Street",       "type": "Street",     "base_lap": 89.3,  "deg_rate": 0.07},
]

COMPOUNDS = ["Soft", "Medium", "Hard"]
COMPOUND_DEG_MULT = {"Soft": 1.4, "Medium": 1.0, "Hard": 0.7}
COMPOUND_PACE = {"Soft": -0.4, "Medium": 0.0, "Hard": 0.35}

rows = []
for season in [2023, 2024, 2025]:
    for rnd, circuit in enumerate(CIRCUITS, start=1):
        is_wet_race = rng.random() < 0.22
        track_temp = rng.uniform(14, 20) if is_wet_race else rng.uniform(28, 46)

        for session in ["Qualifying", "Race"]:
            n_laps = 1 if session == "Qualifying" else rng.integers(48, 64)

            for driver in DRIVERS:
                stint = 1
                tire_age = 0
                compound = rng.choice(COMPOUNDS, p=[0.35, 0.4, 0.25])
                cum_gap = rng.uniform(0, 3)

                for lap in range(1, n_laps + 1):
                    if session == "Race" and lap > 1 and tire_age > rng.integers(15, 26):
                        stint += 1
                        tire_age = 0
                        compound = rng.choice(COMPOUNDS, p=[0.3, 0.45, 0.25])

                    deg = circuit["deg_rate"] * COMPOUND_DEG_MULT[compound] * tire_age
                    wet_effect = (-driver["wet_skill"] * 1.8) if is_wet_race else 0.0
                    pressure_context = 1 if (cum_gap < 1.0 and lap > n_laps * 0.6) else 0
                    clutch_effect = -driver["clutch"] * 0.25 if pressure_context else 0.0

                    noise = rng.normal(0, driver["consistency"])
                    lap_time = (
                        circuit["base_lap"]
                        + driver["base_pace"]
                        + COMPOUND_PACE[compound]
                        + deg
                        + wet_effect
                        + clutch_effect
                        + noise
                    )
                    lap_time = max(lap_time, circuit["base_lap"] * 0.9)

                    cum_gap = max(0.05, cum_gap + rng.normal(0, 0.4) - (0.02 if pressure_context else 0))

                    rows.append({
                        "driver": driver["name"],
                        "driver_code": driver["code"],
                        "circuit": circuit["name"],
                        "circuit_type": circuit["type"],
                        "season": season,
                        "round": rnd,
                        "session": session,
                        "lap_number": lap,
                        "lap_time_s": round(lap_time, 3),
                        "tire_compound": compound,
                        "tire_age": tire_age,
                        "stint": stint,
                        "is_wet": is_wet_race,
                        "track_temp_c": round(track_temp, 1),
                        "gap_to_ahead_s": round(cum_gap, 2),
                        "under_pressure": bool(pressure_context),
                    })
                    tire_age += 1

df = pd.DataFrame(rows)
out_dir = os.path.dirname(__file__)
df.to_csv(os.path.join(out_dir, "laps.csv"), index=False)
print(f"Generated {len(df):,} laps across {df['driver'].nunique()} drivers, "
      f"{df['circuit'].nunique()} circuits, {df['season'].nunique()} seasons")
print(f"Saved to {os.path.join(out_dir, 'laps.csv')}")
