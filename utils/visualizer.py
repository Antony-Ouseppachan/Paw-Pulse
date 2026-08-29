"""
Pet Health & Care Visualizer using Matplotlib
Generates clean, publication-grade analytical charts for weight tracking, nutritional balance, and activity compliance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Styling configuration
DARK_BG = "#131722"
CARD_BG = "#1E222D"
TEXT_COLOR = "#FAFAFA"
MUTED_TEXT = "#A0AAB7"
ACCENT_RED = "#FF6B6B"
ACCENT_GREEN = "#51CF66"
ACCENT_BLUE = "#339AF0"
ACCENT_YELLOW = "#FCC419"
ACCENT_PURPLE = "#845EF7"

def apply_chart_theme(fig, axes):
    """Apply unified dark aesthetic to Matplotlib figures."""
    fig.patch.set_facecolor(CARD_BG)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=MUTED_TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#2D3748")
            spine.set_linewidth(1.0)
        ax.grid(True, linestyle="--", alpha=0.18, color="#FFFFFF")


def plot_weight_trajectory(pet_df, pet_profile):
    """
    Plot historical weight progression with target baseline and linear trendline.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=110)
    apply_chart_theme(fig, ax)

    df_sorted = pet_df.sort_values("date").copy()
    dates = pd.to_datetime(df_sorted["date"])
    weights = df_sorted["weight_kg"].values
    target_wt = pet_profile.get("target_weight_kg", pet_profile.get("weight_kg", 20.0))

    # Weight history line & markers
    ax.plot(dates, weights, color=ACCENT_BLUE, linewidth=2.2, marker="o", markersize=4, label="Recorded Weight (kg)")
    ax.axhline(target_wt, color=ACCENT_GREEN, linestyle="--", linewidth=1.8, label=f"Target Goal ({target_wt} kg)")

    # Linear trendline with NumPy
    if len(weights) > 2:
        x_nums = mdates.date2num(dates)
        poly = np.polyfit(x_nums, weights, 1)
        trend_y = np.polyval(poly, x_nums)
        ax.plot(dates, trend_y, color=ACCENT_YELLOW, linestyle=":", linewidth=1.8, label="Weight Trajectory Trend")

    ax.set_title(f"Weight Progression & Management History — {pet_profile['name']}", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Weight (kg)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Date", fontsize=9.5, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.legend(loc="upper right", framealpha=0.5, facecolor=CARD_BG, edgecolor="none", labelcolor=TEXT_COLOR, fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_energy_balance(pet_df):
    """
    Plot daily caloric intake vs energy expenditure.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=110)
    apply_chart_theme(fig, ax)

    df_sorted = pet_df.sort_values("date").tail(21).copy()
    dates = pd.to_datetime(df_sorted["date"])
    x_idx = np.arange(len(dates))
    bar_width = 0.38

    ax.bar(x_idx - bar_width/2, df_sorted["calories_consumed_kcal"], width=bar_width, color=ACCENT_YELLOW, alpha=0.85, label="Calories Consumed (kcal)")
    ax.bar(x_idx + bar_width/2, df_sorted["calories_burned_kcal"], width=bar_width, color=ACCENT_PURPLE, alpha=0.85, label="Calories Burned (kcal)")

    ax.set_xticks(x_idx)
    ax.set_xticklabels([d.strftime('%b %d') for d in dates], rotation=45, ha="right")
    ax.set_ylabel("Energy (kcal)", fontsize=9.5, fontweight="bold")
    ax.set_title("Caloric Intake vs Total Daily Expenditure (Last 3 Weeks)", fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.5, facecolor=CARD_BG, edgecolor="none", labelcolor=TEXT_COLOR, fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_exercise_and_hydration(pet_df, pet_profile):
    """
    Plot exercise duration and hydration logs against targets.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=110)
    apply_chart_theme(fig, [ax1, ax2])

    df_sorted = pet_df.sort_values("date").tail(14).copy()
    dates = pd.to_datetime(df_sorted["date"])

    target_exercise = pet_profile.get("daily_target_exercise_min", 60)
    target_water = pet_profile.get("daily_target_water_ml", 1500)

    # 1. Exercise Duration
    ax1.bar(dates, df_sorted["exercise_duration_min"], color=ACCENT_GREEN, width=0.6, alpha=0.85, label="Daily Activity (min)")
    ax1.axhline(target_exercise, color="#FCC419", linestyle="--", linewidth=1.5, label=f"Target ({target_exercise} min)")
    ax1.set_title("Daily Exercise Duration", fontsize=10.5, fontweight="bold", pad=10)
    ax1.set_ylabel("Minutes", fontsize=9, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax1.legend(loc="upper left", framealpha=0.4, facecolor=CARD_BG, edgecolor="none", labelcolor=TEXT_COLOR, fontsize=8)

    # 2. Water Intake
    ax2.plot(dates, df_sorted["water_intake_ml"], color=ACCENT_BLUE, marker="s", linewidth=2.0, label="Water Intake (mL)")
    ax2.axhline(target_water, color="#339AF0", linestyle="--", linewidth=1.5, label=f"Target ({target_water} mL)")
    ax2.set_title("Daily Hydration Intake", fontsize=10.5, fontweight="bold", pad=10)
    ax2.set_ylabel("Volume (mL)", fontsize=9, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.legend(loc="upper left", framealpha=0.4, facecolor=CARD_BG, edgecolor="none", labelcolor=TEXT_COLOR, fontsize=8)

    plt.tight_layout()
    return fig


def plot_care_radar(analytics):
    """
    Plot radar chart for overall pet care adherence.
    """
    categories = ['Nutrition', 'Exercise', 'Hydration', 'Weight Compliance']
    weight_score = max(0.0, 100.0 - abs(analytics.get('weight_diff', 0.0)) * 15.0)
    scores = [
        analytics.get('cal_adherence', 90.0),
        analytics.get('exercise_adherence', 85.0),
        analytics.get('water_adherence', 90.0),
        min(100.0, weight_score)
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.8, 4.8), subplot_kw=dict(polar=True), dpi=110)
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(DARK_BG)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], categories, color=TEXT_COLOR, size=9, fontweight="bold")
    plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color=MUTED_TEXT, size=7.5)
    plt.ylim(0, 100)

    ax.plot(angles, scores, color=ACCENT_GREEN, linewidth=2, linestyle='solid')
    ax.fill(angles, scores, color=ACCENT_GREEN, alpha=0.3)
    ax.grid(color="#2D3748", linestyle="--")

    ax.set_title(f"Care Compliance Index: {analytics.get('care_score', 88)}/100", size=10.5, fontweight="bold", color=TEXT_COLOR, pad=15)
    plt.tight_layout()
    return fig


def plot_correlations(pet_df):
    """
    Plot correlation matrix across recorded care features.
    """
    cols = ["weight_kg", "calories_consumed_kcal", "water_intake_ml", "exercise_duration_min", "calories_burned_kcal", "mood_score"]
    rename_dict = {
        "weight_kg": "Weight",
        "calories_consumed_kcal": "Calories In",
        "water_intake_ml": "Water",
        "exercise_duration_min": "Exercise",
        "calories_burned_kcal": "Calories Out",
        "mood_score": "Mood Score"
    }
    sub_df = pet_df[cols].rename(columns=rename_dict)
    corr = sub_df.corr()

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=110)
    apply_chart_theme(fig, ax)

    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="left", color=TEXT_COLOR, fontsize=8.5)
    ax.set_yticklabels(corr.columns, color=TEXT_COLOR, fontsize=8.5)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="#FFFFFF" if abs(corr.iloc[i,j]) > 0.5 else "#000000", fontsize=8, fontweight="bold")

    ax.set_title("Pet Health & Activity Correlation Matrix", fontsize=10.5, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig
