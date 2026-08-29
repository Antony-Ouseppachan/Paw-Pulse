"""
Pet Data Management Engine
Handles data storage, profile schemas, daily care logs, and historical datasets using Pandas & NumPy.
"""

import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROFILES_PATH = os.path.join(DATA_DIR, "pet_profiles.json")
RECORDS_PATH = os.path.join(DATA_DIR, "pet_records.csv")


def load_pet_profiles(filepath=PROFILES_PATH):
    """Load all configured pet profiles."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pet_profiles(profiles, filepath=PROFILES_PATH):
    """Save pet profiles to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def generate_sample_historical_records(profiles, days=60):
    """
    Generate realistic multi-day care and tracking logs for pets using NumPy & Pandas.
    Simulates weight fluctuations, food/water consumption, exercise duration, and mood.
    """
    all_rows = []
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    for pet in profiles:
        pet_id = pet["id"]
        base_weight = pet["weight_kg"]
        target_cal = pet.get("daily_target_calories_kcal", 1200)
        target_water = pet.get("daily_target_water_ml", 1500)
        target_exercise = pet.get("daily_target_exercise_min", 60)

        # Generate realistic weight progression trajectory with small random walk
        weight_noise = np.random.normal(0, 0.05, size=len(date_range))
        weight_trend = np.linspace(base_weight + 0.6, base_weight, len(date_range))
        weights = np.round(weight_trend + np.cumsum(weight_noise * 0.2), 2)

        # Food & Water intake
        food_calories = np.round(np.random.normal(target_cal, target_cal * 0.08, size=len(date_range)), 0)
        water_ml = np.round(np.random.normal(target_water, target_water * 0.12, size=len(date_range)), 0)
        
        # Exercise duration (minutes) - higher on weekends
        day_of_week = date_range.dayofweek
        weekend_boost = np.where(day_of_week >= 5, 20.0, 0.0)
        exercise_mins = np.clip(np.round(np.random.normal(target_exercise, 12, size=len(date_range)) + weekend_boost), 15, 180).astype(int)

        # Calorie expenditure estimation (Metabolic + Activity)
        # RER = 70 * (weight ^ 0.75)
        # Exercise burn approx 4-7 kcal per minute depending on weight
        burn_rate_per_min = (base_weight * 0.15) + 2.0
        calories_burned = np.round(70 * (weights ** 0.75) + (exercise_mins * burn_rate_per_min), 0)

        # Mood score (1 to 5) and Health status
        mood_scores = np.random.choice([3, 4, 5], size=len(date_range), p=[0.1, 0.4, 0.5])
        health_statuses = np.random.choice(["Healthy", "Healthy", "Healthy", "Minor Lethargy"], size=len(date_range), p=[0.90, 0.05, 0.03, 0.02])

        for i, dt in enumerate(date_range):
            all_rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "pet_id": pet_id,
                "pet_name": pet["name"],
                "weight_kg": weights[i],
                "calories_consumed_kcal": int(food_calories[i]),
                "water_intake_ml": int(water_ml[i]),
                "exercise_duration_min": int(exercise_mins[i]),
                "calories_burned_kcal": int(calories_burned[i]),
                "mood_score": int(mood_scores[i]),
                "health_status": health_statuses[i],
                "notes": "Regular daily routine" if mood_scores[i] >= 4 else "Slightly less active today"
            })

    df = pd.DataFrame(all_rows)
    return df


def load_pet_records(filepath=RECORDS_PATH, profiles=None):
    """Load historical logs from CSV, creating sample data if missing."""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
        except Exception:
            pass

    if profiles is None:
        profiles = load_pet_profiles()
    df = generate_sample_historical_records(profiles, days=60)
    save_pet_records(df, filepath)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def save_pet_records(df, filepath=RECORDS_PATH):
    """Save records DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)


def add_daily_record(new_record_dict, filepath=RECORDS_PATH):
    """Append a new daily care record into the CSV database."""
    profiles = load_pet_profiles()
    df = load_pet_records(filepath, profiles)
    
    new_df = pd.DataFrame([new_record_dict])
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
    
    # Check if entry exists for this pet on this date, update or append
    mask = (df["pet_id"] == new_record_dict["pet_id"]) & (df["date"] == new_df["date"].iloc[0])
    if mask.any():
        for col in new_record_dict.keys():
            if col in df.columns:
                df.loc[mask, col] = new_record_dict[col]
    else:
        df = pd.concat([df, new_df], ignore_index=True)
        
    save_pet_records(df, filepath)
    return df
