"""
Pet Health & Nutrition Analytics Engine
Uses NumPy and Pandas for nutritional requirements, weight trajectory modeling, and compliance scoring.
"""

import numpy as np
import pandas as pd


def calculate_energy_requirements(weight_kg, species="Dog", life_stage="Adult", activity_factor=1.6):
    """
    Calculate Resting Energy Requirement (RER) and Daily Energy Requirement (DER)
    using veterinary standard formulas:
    RER = 70 * (weight_kg ** 0.75)
    DER = RER * activity_multiplier
    """
    rer = 70.0 * (weight_kg ** 0.75)
    
    # Species & Life Stage multipliers
    if species.lower() == "cat":
        multipliers = {
            "Neutered Adult": 1.2,
            "Intact Adult": 1.4,
            "Weight Loss": 0.8,
            "Kitten": 2.5
        }
    else:  # Dog
        multipliers = {
            "Neutered Adult": 1.6,
            "Intact Adult": 1.8,
            "Light Work / Active": 2.0,
            "Heavy Work": 3.0,
            "Weight Loss": 1.0,
            "Puppy": 3.0
        }

    mult = multipliers.get(life_stage, activity_factor)
    der = rer * mult

    return {
        "rer_kcal": round(rer, 1),
        "der_kcal": round(der, 1),
        "multiplier": mult,
        "daily_water_ml": round(weight_kg * 60.0, 1)  # standard ~60ml per kg per day
    }


def compute_pet_analytics(pet_df, pet_profile):
    """
    Compute aggregate health, weight regression, and compliance metrics over the records dataframe.
    """
    if pet_df.empty:
        return {}

    # Sort chronologically
    df_sorted = pet_df.sort_values("date").copy()
    
    # 1. Weight Progression & NumPy Linear Fit
    weights = df_sorted["weight_kg"].values
    days_arr = np.arange(len(weights))
    
    if len(weights) > 1:
        # Linear regression slope (kg / day)
        slope, intercept = np.polyfit(days_arr, weights, 1)
        weekly_weight_change = slope * 7.0
    else:
        slope = 0.0
        weekly_weight_change = 0.0

    target_weight = pet_profile.get("target_weight_kg", pet_profile.get("weight_kg", 20.0))
    current_weight = weights[-1]
    weight_diff_to_target = current_weight - target_weight

    # 2. Nutrition & Hydration Compliance
    target_cal = pet_profile.get("daily_target_calories_kcal", 1200)
    target_water = pet_profile.get("daily_target_water_ml", 1500)
    target_exercise = pet_profile.get("daily_target_exercise_min", 60)

    avg_calories = float(df_sorted["calories_consumed_kcal"].mean())
    avg_water = float(df_sorted["water_intake_ml"].mean())
    avg_exercise = float(df_sorted["exercise_duration_min"].mean())
    avg_burned = float(df_sorted["calories_burned_kcal"].mean())

    cal_adherence = min(100.0, max(0.0, 100.0 - abs(avg_calories - target_cal) / target_cal * 100.0))
    water_adherence = min(100.0, (avg_water / target_water) * 100.0)
    exercise_adherence = min(100.0, (avg_exercise / target_exercise) * 100.0)

    # 3. Overall Care Score (0-100)
    care_score = round(0.35 * cal_adherence + 0.35 * exercise_adherence + 0.30 * water_adherence, 1)

    # 4. Energy Balance (Intake vs Burned)
    energy_balance_net = avg_calories - avg_burned

    return {
        "current_weight": current_weight,
        "target_weight": target_weight,
        "weight_diff": round(weight_diff_to_target, 2),
        "weekly_weight_change": round(weekly_weight_change, 3),
        "avg_calories": round(avg_calories, 0),
        "target_calories": target_cal,
        "cal_adherence": round(cal_adherence, 1),
        "avg_water": round(avg_water, 0),
        "target_water": target_water,
        "water_adherence": round(water_adherence, 1),
        "avg_exercise_min": round(avg_exercise, 1),
        "target_exercise_min": target_exercise,
        "exercise_adherence": round(exercise_adherence, 1),
        "avg_calories_burned": round(avg_burned, 0),
        "energy_balance_net": round(energy_balance_net, 0),
        "care_score": care_score,
        "total_logged_days": len(df_sorted)
    }


def classify_body_condition(current_weight, target_weight):
    """
    Body Condition Score (BCS) heuristic from 1 to 9 based on target weight deviation.
    """
    ratio = current_weight / max(target_weight, 0.1)
    if ratio < 0.85:
        return "BCS 3/9 (Underweight)", "#FCC419"
    elif 0.85 <= ratio <= 1.08:
        return "BCS 5/9 (Ideal Weight)", "#51CF66"
    elif 1.08 < ratio <= 1.20:
        return "BCS 7/9 (Overweight)", "#FCC419"
    else:
        return "BCS 9/9 (Obese)", "#FF6B6B"
