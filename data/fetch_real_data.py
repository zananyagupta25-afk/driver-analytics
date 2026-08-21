"""
Run this LOCALLY (not in a restricted sandbox) to replace the synthetic
laps.csv with real timing data pulled via the FastF1 package.

Setup:
    pip install fastf1

Usage:
    python fetch_real_data.py

This pulls lap-by-lap timing, tire, and weather data for the seasons/rounds
listed below and writes it to data/laps.csv in the same schema the app
expects, so the dashboard works unchanged on real data.
"""
import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache(os.path.join(os.path.dirname(__file__), ".fastf1_cache"))

SEASONS = [2023, 2024]          # extend as needed
ROUNDS = range(1, 6)            # first 5 rounds per season, extend as needed
SESSIONS = ["Q", "R"]           # Qualifying, Race

rows = []
for season in SEASONS:
    for rnd in ROUNDS:
        for session_code in SESSIONS:
            try:
                session = fastf1.get_session(season, rnd, session_code)
                session.load(laps=True, weather=True, telemetry=False)
            except Exception as e:
                print(f"Skip {season} R{rnd} {session_code}: {e}")
                continue

            laps = session.laps
            weather = session.weather_data
            is_wet = weather["Rainfall"].any() if weather is not None and not weather.empty else False
            track_temp = weather["TrackTemp"].mean() if weather is not None and not weather.empty else None

            for _, lap in laps.iterrows():
                if pd.isna(lap["LapTime"]):
                    continue
                rows.append({
                    "driver": lap.get("Driver", ""),
                    "driver_code": lap.get("Driver", ""),
                    "circuit": session.event["EventName"],
                    "circuit_type": "Unknown",  # tag manually or via a lookup table
                    "season": season,
                    "round": rnd,
                    "session": "Qualifying" if session_code == "Q" else "Race",
                    "lap_number": lap.get("LapNumber", None),
                    "lap_time_s": lap["LapTime"].total_seconds(),
                    "tire_compound": lap.get("Compound", "Unknown"),
                    "tire_age": lap.get("TyreLife", 0),
                    "stint": lap.get("Stint", 1),
                    "is_wet": bool(is_wet),
                    "track_temp_c": track_temp,
                    "gap_to_ahead_s": None,   # derive from GapToLeader/timing if needed
                    "under_pressure": False,  # compute post-hoc from gap data
                })

df = pd.DataFrame(rows)
out_path = os.path.join(os.path.dirname(__file__), "laps.csv")
df.to_csv(out_path, index=False)
print(f"Saved {len(df):,} real laps to {out_path}")
