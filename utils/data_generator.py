"""
IoT Telemetry & Pet Health Data Generator
Leverages NumPy and Pandas to simulate realistic multi-sensor smart collar telemetry.
"""

import json
import os
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Earth radius in meters for geodetic conversions
EARTH_RADIUS_M = 6371000.0


def load_pet_profiles(filepath=None):
    """Load pet profiles from JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "pet_profiles.json")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pet_profiles(profiles, filepath=None):
    """Save pet profiles to JSON file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "pet_profiles.json")
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def meters_to_lat_lon(delta_north, delta_east, base_lat, base_lon):
    """Convert metric offsets (north/east) into GPS latitude/longitude."""
    d_lat = (delta_north / EARTH_RADIUS_M) * (180.0 / np.pi)
    d_lon = (delta_east / (EARTH_RADIUS_M * np.cos(np.radians(base_lat)))) * (180.0 / np.pi)
    return base_lat + d_lat, base_lon + d_lon


def haversine_distance_np(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine distance in meters between coordinates.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def generate_telemetry(pet_profile, hours=24, interval_minutes=5, start_time=None, anomaly_mode=None):
    """
    Generate realistic multi-sensor collar telemetry for a pet.
    
    Parameters:
    - pet_profile: dict with pet information and baseline ranges
    - hours: number of hours to simulate
    - interval_minutes: sampling frequency in minutes
    - start_time: datetime for the start of simulation
    - anomaly_mode: None, 'fever', 'lethargy', 'tachycardia', 'escape'
    
    Returns:
    - pd.DataFrame of timestamped sensor readings
    """
    if start_time is None:
        # Default to ending at current time
        start_time = datetime.now() - timedelta(hours=hours)

    total_points = int((hours * 60) / interval_minutes)
    time_index = [start_time + timedelta(minutes=i * interval_minutes) for i in range(total_points)]
    
    # Time of day (fractional hour 0 - 24)
    hours_of_day = np.array([t.hour + t.minute / 60.0 for t in time_index])

    # 1. BASELINE VITAL RANGES
    vitals = pet_profile.get("vitals_normal", {
        "hr_min": 70, "hr_max": 120,
        "temp_min": 38.0, "temp_max": 39.2,
        "resp_min": 15, "resp_max": 30
    })
    
    hr_mid = (vitals["hr_min"] + vitals["hr_max"]) / 2.0
    hr_range = (vitals["hr_max"] - vitals["hr_min"]) / 2.0
    
    temp_mid = (vitals["temp_min"] + vitals["temp_max"]) / 2.0
    temp_range = (vitals["temp_max"] - vitals["temp_min"]) / 2.0

    resp_mid = (vitals["resp_min"] + vitals["resp_max"]) / 2.0
    resp_range = (vitals["resp_max"] - vitals["resp_min"]) / 2.0

    # 2. DIURNAL ACTIVITY & CIRCADIAN RHYTHM USING NUMPY
    # Morning walk peak around 7:30 AM, Evening walk peak around 6:30 PM
    morning_walk = np.exp(-0.5 * ((hours_of_day - 7.5) / 1.0) ** 2)
    evening_walk = np.exp(-0.5 * ((hours_of_day - 18.5) / 1.2) ** 2)
    afternoon_play = 0.4 * np.exp(-0.5 * ((hours_of_day - 13.0) / 1.5) ** 2)
    
    # Circadian activity profile (0 = deep sleep, 1 = intense exercise)
    activity_base = 0.15 + 0.70 * morning_walk + 0.85 * evening_walk + afternoon_play
    activity_noise = np.random.uniform(0.0, 0.15, size=total_points)
    activity_level = np.clip(activity_base + activity_noise, 0.0, 1.0)

    # 3. STEPS & ACCELEROMETER METRICS
    steps_per_interval = (activity_level * 180 + np.random.poisson(lam=5, size=total_points)).astype(int)
    # Calorie burn rate in kcal per interval
    calories_burned = np.round((activity_level * 12.0 + (pet_profile.get("weight_kg", 20) * 0.08)) * (interval_minutes / 5.0), 2)

    # 4. HEART RATE (BPM) - Correlated with activity + circadian rhythm
    hr_circadian = hr_mid + (activity_level - 0.3) * hr_range * 1.6
    hr_noise = np.random.normal(0, 3.5, size=total_points)
    heart_rate = np.clip(hr_circadian + hr_noise, vitals["hr_min"] - 15, vitals["hr_max"] + 60)

    # 5. BODY TEMPERATURE (°C)
    temp_circadian = temp_mid + (activity_level - 0.4) * (temp_range * 0.5)
    temp_noise = np.random.normal(0, 0.08, size=total_points)
    body_temp = np.round(temp_circadian + temp_noise, 2)

    # 6. RESPIRATORY RATE (breaths/min)
    resp_rate = np.clip(np.round(resp_mid + (activity_level - 0.2) * resp_range * 1.5 + np.random.normal(0, 1.5, size=total_points)), 10, 80)

    # 7. GPS SIMULATION & GEOFENCE
    base_lat = pet_profile.get("home_lat", 37.7749)
    base_lon = pet_profile.get("home_lon", -122.4194)
    geo_radius = pet_profile.get("geofence_radius_m", 100)

    # Simulating coordinate walk paths
    # Active hours have walk loops, resting hours stay inside home radius
    north_offsets = np.zeros(total_points)
    east_offsets = np.zeros(total_points)

    curr_north = 0.0
    curr_east = 0.0
    
    for i in range(total_points):
        act = activity_level[i]
        if act > 0.45:  # Walking / outdoor
            angle = (i * 0.25) + np.random.uniform(-0.5, 0.5)
            step_dist = (act * 25.0) + np.random.uniform(5, 15)
            curr_north += step_dist * np.sin(angle)
            curr_east += step_dist * np.cos(angle)
            # Tether loosely towards home unless escape mode
            curr_north *= 0.94
            curr_east *= 0.94
        else:  # At home / resting
            curr_north = np.random.normal(0, 8)
            curr_east = np.random.normal(0, 8)
        
        north_offsets[i] = curr_north
        east_offsets[i] = curr_east

    # 8. ANOMALY INJECTIONS (if requested)
    if anomaly_mode == "fever":
        # Fever spike in second half
        spike_idx = int(total_points * 0.55)
        body_temp[spike_idx:] += np.linspace(0.8, 1.8, total_points - spike_idx)
        heart_rate[spike_idx:] += np.linspace(15, 35, total_points - spike_idx)
    elif anomaly_mode == "tachycardia":
        # Sudden erratic heart rate spike
        spike_idx = int(total_points * 0.4)
        heart_rate[spike_idx:spike_idx + 24] += 55.0
    elif anomaly_mode == "lethargy":
        # Zero activity / flatlined movement
        activity_level = activity_level * 0.15
        steps_per_interval = (steps_per_interval * 0.1).astype(int)
        heart_rate = np.clip(heart_rate - 20, vitals["hr_min"] - 10, vitals["hr_max"])
    elif anomaly_mode == "escape":
        # Pet breaks geofence
        spike_idx = int(total_points * 0.6)
        north_offsets[spike_idx:] += np.linspace(50, geo_radius + 300, total_points - spike_idx)
        east_offsets[spike_idx:] += np.linspace(40, geo_radius + 250, total_points - spike_idx)

    # Convert metric offsets to GPS
    lats, lons = meters_to_lat_lon(north_offsets, east_offsets, base_lat, base_lon)
    dist_from_home = haversine_distance_np(base_lat, base_lon, lats, lons)
    in_geofence = dist_from_home <= geo_radius

    # 9. FEEDING & HYDRATION EVENTS
    food_consumed_g = np.zeros(total_points)
    water_consumed_ml = np.zeros(total_points)

    # Target food and water distributed across morning & evening meals
    target_food = pet_profile.get("target_food_g", 300)
    target_water = pet_profile.get("target_water_ml", 1200)

    for i, t in enumerate(time_index):
        # Breakfast at ~8:00 AM
        if t.hour == 8 and t.minute < interval_minutes:
            food_consumed_g[i] = round(target_food * 0.5 + np.random.uniform(-15, 15), 1)
        # Dinner at ~19:00 (7 PM)
        elif t.hour == 19 and t.minute < interval_minutes:
            food_consumed_g[i] = round(target_food * 0.5 + np.random.uniform(-15, 15), 1)
        
        # Hydration sips throughout active hours
        if activity_level[i] > 0.35 and np.random.random() < 0.25:
            water_consumed_ml[i] = round(np.random.uniform(25, 80), 1)

    # 10. COLLAR BATTERY DEPLETION (linear with slight fluctuation)
    initial_battery = pet_profile.get("battery_pct", 90)
    battery_drain = np.linspace(0, 6.0 * (hours / 24.0), total_points)
    battery_level = np.clip(np.round(initial_battery - battery_drain, 1), 0, 100)

    # 11. ASSEMBLE DATAFRAME
    df = pd.DataFrame({
        "timestamp": time_index,
        "pet_id": pet_profile.get("id"),
        "pet_name": pet_profile.get("name"),
        "collar_id": pet_profile.get("collar_id"),
        "heart_rate_bpm": np.round(heart_rate, 1),
        "body_temp_c": np.round(body_temp, 2),
        "resp_rate_bpm": resp_rate.astype(int),
        "activity_score": np.round(activity_level, 3),
        "step_count": steps_per_interval,
        "calories_burned_kcal": calories_burned,
        "latitude": lats,
        "longitude": lons,
        "distance_from_home_m": np.round(dist_from_home, 1),
        "geofence_safe": in_geofence,
        "food_intake_g": food_consumed_g,
        "water_intake_ml": water_consumed_ml,
        "battery_pct": battery_level
    })

    return df


def generate_live_sample(pet_profile, previous_state=None, inject_anomaly=None):
    """
    Generate a single telemetry packet simulating real-time collar IoT payload.
    """
    now = datetime.now()
    hour_val = now.hour + now.minute / 60.0
    
    vitals = pet_profile.get("vitals_normal", {
        "hr_min": 70, "hr_max": 120,
        "temp_min": 38.0, "temp_max": 39.2,
        "resp_min": 15, "resp_max": 30
    })

    # Basic baseline
    hr_mid = (vitals["hr_min"] + vitals["hr_max"]) / 2.0
    temp_mid = (vitals["temp_min"] + vitals["temp_max"]) / 2.0
    resp_mid = (vitals["resp_min"] + vitals["resp_max"]) / 2.0

    activity = float(np.clip(np.random.beta(2, 5) * 0.8, 0.05, 0.95))
    hr = float(np.round(hr_mid + (activity - 0.3) * 30 + np.random.normal(0, 3), 1))
    temp = float(np.round(temp_mid + (activity - 0.3) * 0.3 + np.random.normal(0, 0.05), 2))
    resp = int(np.clip(resp_mid + (activity - 0.3) * 10 + np.random.normal(0, 1.5), 10, 60))

    base_lat = pet_profile.get("home_lat", 37.7749)
    base_lon = pet_profile.get("home_lon", -122.4194)
    geo_radius = pet_profile.get("geofence_radius_m", 100)

    # GPS offset
    offset_dist = np.random.uniform(5, geo_radius * 0.7)
    angle = np.random.uniform(0, 2 * np.pi)
    
    if inject_anomaly == "fever":
        temp += 2.0
        hr += 35
    elif inject_anomaly == "tachycardia":
        hr += 60
    elif inject_anomaly == "escape":
        offset_dist = geo_radius + np.random.uniform(80, 200)

    lat, lon = meters_to_lat_lon(offset_dist * np.sin(angle), offset_dist * np.cos(angle), base_lat, base_lon)
    dist_home = float(haversine_distance_np(base_lat, base_lon, lat, lon))
    is_safe = bool(dist_home <= geo_radius)

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "pet_id": pet_profile.get("id"),
        "pet_name": pet_profile.get("name"),
        "collar_id": pet_profile.get("collar_id"),
        "heart_rate_bpm": hr,
        "body_temp_c": temp,
        "resp_rate_bpm": resp,
        "activity_score": round(activity, 2),
        "steps_last_5min": int(activity * 120 + np.random.randint(0, 15)),
        "calories_kcal": round(activity * 4.2 + 0.5, 2),
        "latitude": round(lat, 6),
        "longitude": round(lon, 6),
        "distance_from_home_m": round(dist_home, 1),
        "geofence_safe": is_safe,
        "battery_pct": pet_profile.get("battery_pct", 85)
    }
