"""
Pet Health & Telemetry Analytics Engine
Leverages NumPy and Pandas for statistical modeling, anomaly detection, and wellness scoring.
"""

import numpy as np
import pandas as pd


def compute_vital_statistics(df, pet_profile):
    """
    Calculate summary statistics and rolling trends for key vitals.
    """
    vitals_cfg = pet_profile.get("vitals_normal", {})
    hr_min_norm = vitals_cfg.get("hr_min", 70)
    hr_max_norm = vitals_cfg.get("hr_max", 120)
    temp_min_norm = vitals_cfg.get("temp_min", 38.0)
    temp_max_norm = vitals_cfg.get("temp_max", 39.2)

    stats = {
        "heart_rate": {
            "mean": float(df["heart_rate_bpm"].mean()),
            "std": float(df["heart_rate_bpm"].std()),
            "min": float(df["heart_rate_bpm"].min()),
            "max": float(df["heart_rate_bpm"].max()),
            "median": float(df["heart_rate_bpm"].median()),
            "q25": float(df["heart_rate_bpm"].quantile(0.25)),
            "q75": float(df["heart_rate_bpm"].quantile(0.75)),
            "normal_range": [hr_min_norm, hr_max_norm]
        },
        "body_temp": {
            "mean": float(df["body_temp_c"].mean()),
            "std": float(df["body_temp_c"].std()),
            "min": float(df["body_temp_c"].min()),
            "max": float(df["body_temp_c"].max()),
            "median": float(df["body_temp_c"].median()),
            "normal_range": [temp_min_norm, temp_max_norm]
        },
        "respiratory_rate": {
            "mean": float(df["resp_rate_bpm"].mean()),
            "min": float(df["resp_rate_bpm"].min()),
            "max": float(df["resp_rate_bpm"].max())
        },
        "activity": {
            "total_steps": int(df["step_count"].sum()),
            "total_calories_burned": float(df["calories_burned_kcal"].sum()),
            "avg_activity_score": float(df["activity_score"].mean())
        },
        "nutrition": {
            "total_food_g": float(df["food_intake_g"].sum()),
            "total_water_ml": float(df["water_intake_ml"].sum())
        }
    }
    return stats


def detect_anomalies(df, pet_profile, z_thresh=2.3):
    """
    Detect physiological and behavioral anomalies using NumPy Z-Scores and clinical thresholds.
    
    Returns:
    - enriched DataFrame with anomaly flags
    - list of structured alert objects
    """
    df_analyzed = df.copy()
    vitals_cfg = pet_profile.get("vitals_normal", {})
    
    hr_min_norm = vitals_cfg.get("hr_min", 70)
    hr_max_norm = vitals_cfg.get("hr_max", 120)
    temp_max_norm = vitals_cfg.get("temp_max", 39.2)
    temp_min_norm = vitals_cfg.get("temp_min", 38.0)

    # 1. Z-SCORE COMPUTATION WITH NUMPY
    hr_mean = df_analyzed["heart_rate_bpm"].mean()
    hr_std = max(df_analyzed["heart_rate_bpm"].std(), 1e-5)
    df_analyzed["hr_zscore"] = np.abs((df_analyzed["heart_rate_bpm"] - hr_mean) / hr_std)

    temp_mean = df_analyzed["body_temp_c"].mean()
    temp_std = max(df_analyzed["body_temp_c"].std(), 1e-5)
    df_analyzed["temp_zscore"] = np.abs((df_analyzed["body_temp_c"] - temp_mean) / temp_std)

    # 2. ANOMALY FLAGS
    # Fever: High temp beyond normal threshold or statistical outlier
    df_analyzed["is_fever"] = (df_analyzed["body_temp_c"] > temp_max_norm) | (
        (df_analyzed["temp_zscore"] > z_thresh) & (df_analyzed["body_temp_c"] > temp_mean)
    )

    # Hypothermia: Low temp
    df_analyzed["is_hypothermia"] = df_analyzed["body_temp_c"] < temp_min_norm

    # Tachycardia: Elevated heart rate when activity is low (resting tachycardia / distress)
    df_analyzed["is_resting_tachycardia"] = (
        (df_analyzed["heart_rate_bpm"] > hr_max_norm) & (df_analyzed["activity_score"] < 0.3)
    )

    # Geofence breach
    df_analyzed["is_geofence_breach"] = ~df_analyzed["geofence_safe"]

    # Any anomaly flag
    df_analyzed["has_anomaly"] = (
        df_analyzed["is_fever"] | 
        df_analyzed["is_hypothermia"] | 
        df_analyzed["is_resting_tachycardia"] | 
        df_analyzed["is_geofence_breach"]
    )

    # 3. COMPILE STRUCTURED ALERTS
    alerts = []
    
    # Check Geofence breaches
    breaches = df_analyzed[df_analyzed["is_geofence_breach"]]
    if not breaches.empty:
        max_dist = breaches["distance_from_home_m"].max()
        last_breach_time = breaches["timestamp"].iloc[-1]
        alerts.append({
            "type": "CRITICAL",
            "category": "GPS Geofence",
            "title": "Geofence Boundary Breach Detected!",
            "message": f"{pet_profile['name']} exited safe zone. Maximum distance recorded: {max_dist:.1f} meters at {last_breach_time}.",
            "timestamp": str(last_breach_time)
        })

    # Check Fever
    fevers = df_analyzed[df_analyzed["is_fever"]]
    if not fevers.empty:
        max_temp = fevers["body_temp_c"].max()
        alerts.append({
            "type": "WARNING",
            "category": "Thermal Health",
            "title": "Elevated Body Temperature (Fever)",
            "message": f"Recorded peak temperature of {max_temp:.2f}°C exceeding safe limit ({temp_max_norm}°C). Monitor hydration and check for distress.",
            "timestamp": str(fevers["timestamp"].iloc[-1])
        })

    # Check Resting Tachycardia
    tachy = df_analyzed[df_analyzed["is_resting_tachycardia"]]
    if not tachy.empty:
        max_hr = tachy["heart_rate_bpm"].max()
        alerts.append({
            "type": "WARNING",
            "category": "Cardiac Vitals",
            "title": "Resting Tachycardia / High Stress",
            "message": f"Heart rate reached {max_hr:.1f} BPM while resting (normal max: {hr_max_norm} BPM). Potential pain, anxiety, or heat stress.",
            "timestamp": str(tachy["timestamp"].iloc[-1])
        })

    # Battery alert
    latest_battery = df_analyzed["battery_pct"].iloc[-1]
    if latest_battery < 20:
        alerts.append({
            "type": "CRITICAL" if latest_battery < 10 else "WARNING",
            "category": "Hardware / Device",
            "title": "Smart Collar Battery Low",
            "message": f"Collar battery level is at {latest_battery:.1f}%. Please recharge collar {pet_profile.get('collar_id')}.",
            "timestamp": str(df_analyzed["timestamp"].iloc[-1])
        })

    return df_analyzed, alerts


def calculate_wellness_score(df, pet_profile):
    """
    Calculate an overall Pet Wellness Index (0-100) using weighted sub-scores:
    - Activity Score (25%)
    - Vital Signs Stability (30%)
    - Nutrition & Hydration Compliance (25%)
    - Sleep / Rest Quality (20%)
    """
    # 1. Activity Compliance
    target_steps = pet_profile.get("target_daily_steps", 8000)
    actual_steps = df["step_count"].sum()
    activity_score = min(100.0, (actual_steps / target_steps) * 100.0)

    # 2. Vital Signs Stability (penalize out-of-bound readings)
    vitals_cfg = pet_profile.get("vitals_normal", {})
    hr_safe = ((df["heart_rate_bpm"] >= vitals_cfg.get("hr_min", 60)) & 
               (df["heart_rate_bpm"] <= vitals_cfg.get("hr_max", 120))).mean() * 100.0
    temp_safe = ((df["body_temp_c"] >= vitals_cfg.get("temp_min", 38.0)) & 
                 (df["body_temp_c"] <= vitals_cfg.get("temp_max", 39.2))).mean() * 100.0
    vitals_score = 0.5 * hr_safe + 0.5 * temp_safe

    # 3. Nutrition & Hydration
    target_water = pet_profile.get("target_water_ml", 1200)
    actual_water = df["water_intake_ml"].sum()
    water_score = min(100.0, (actual_water / target_water) * 100.0) if target_water > 0 else 90.0

    target_food = pet_profile.get("target_food_g", 300)
    actual_food = df["food_intake_g"].sum()
    food_score = min(100.0, (actual_food / target_food) * 100.0) if target_food > 0 else 90.0
    nutrition_score = 0.5 * water_score + 0.5 * food_score

    # 4. Rest & Recovery Score
    resting_samples = (df["activity_score"] < 0.2).sum()
    total_samples = len(df)
    rest_pct = (resting_samples / total_samples) * 100.0
    # Ideal rest percentage for pets is roughly 50-70% of day
    if 45 <= rest_pct <= 75:
        rest_score = 100.0
    else:
        rest_score = max(50.0, 100.0 - abs(rest_pct - 60) * 2.0)

    # Overall Weighted Score
    overall_wellness = round(
        0.25 * activity_score + 
        0.30 * vitals_score + 
        0.25 * nutrition_score + 
        0.20 * rest_score, 
        1
    )

    return {
        "overall_score": overall_wellness,
        "activity_score": round(activity_score, 1),
        "vitals_score": round(vitals_score, 1),
        "nutrition_score": round(nutrition_score, 1),
        "rest_score": round(rest_score, 1),
        "actual_steps": int(actual_steps),
        "target_steps": target_steps,
        "actual_water_ml": round(actual_water, 1),
        "target_water_ml": target_water,
        "actual_food_g": round(actual_food, 1),
        "target_food_g": target_food
    }


def classify_activity_distribution(df):
    """
    Classify telemetry records into activity regimes:
    - Deep Sleep (activity < 0.08)
    - Light Rest (0.08 <= activity < 0.25)
    - Moderate Active (0.25 <= activity < 0.60)
    - High Intensity / Play (activity >= 0.60)
    """
    conditions = [
        df["activity_score"] < 0.08,
        (df["activity_score"] >= 0.08) & (df["activity_score"] < 0.25),
        (df["activity_score"] >= 0.25) & (df["activity_score"] < 0.60),
        df["activity_score"] >= 0.60
    ]
    labels = ["Deep Sleep", "Light Rest", "Moderate Active", "High Intensity Play"]
    df_classified = df.copy()
    df_classified["activity_state"] = np.select(conditions, labels, default="Light Rest")
    
    distribution = df_classified["activity_state"].value_counts(normalize=True) * 100.0
    return df_classified, distribution
