"""
PawPulse — Pet Health, Nutrition & Care Management System
Built with Streamlit, NumPy, Pandas, and Matplotlib.
Version: 2.0 (Software Management Edition)
"""

import os
import sys
import json
from datetime import datetime, timedelta
import streamlit as st
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from utils.data_manager import (
    load_pet_profiles, save_pet_profiles, 
    load_pet_records, save_pet_records, add_daily_record
)
from utils.analytics import (
    calculate_energy_requirements, compute_pet_analytics, classify_body_condition
)
from utils.visualizer import (
    plot_weight_trajectory, plot_energy_balance, 
    plot_exercise_and_hydration, plot_care_radar, plot_correlations
)

# Page configuration
st.set_page_config(
    page_title="PawPulse — Pet Health & Care Management System",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern, premium appearance
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Header Card */
    .main-header {
        background: linear-gradient(135deg, #1E222D 0%, #2A2F3D 100%);
        padding: 20px 25px;
        border-radius: 16px;
        border: 1px solid #363C4E;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1E222D;
        border: 1px solid #2D3748;
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #51CF66;
    }
    .metric-title {
        color: #A0AAB7;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 6px 0;
        color: #FAFAFA;
    }
    .metric-sub {
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    /* Card Container */
    .content-card {
        background: #1E222D;
        border: 1px solid #2D3748;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-valid { background-color: #2B8A3E; color: #EBFBEE; }
    .badge-warning { background-color: #D9480F; color: #FFF4E6; }
    .badge-info { background-color: #1864AB; color: #E7F5FF; }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "profiles" not in st.session_state:
    st.session_state["profiles"] = load_pet_profiles()

if "selected_pet_id" not in st.session_state:
    st.session_state["selected_pet_id"] = st.session_state["profiles"][0]["id"] if st.session_state["profiles"] else ""

if "records_df" not in st.session_state:
    st.session_state["records_df"] = load_pet_records(profiles=st.session_state["profiles"])


# Helper to get current selected pet
def get_current_pet():
    for p in st.session_state["profiles"]:
        if p["id"] == st.session_state["selected_pet_id"]:
            return p
    return st.session_state["profiles"][0] if st.session_state["profiles"] else None


current_pet = get_current_pet()


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1548767797-d8c844163c4c?w=300&auto=format&fit=crop&q=80",
        use_container_width=True
    )
    st.title("PawPulse")
    st.caption("Pet Care & Health Management System")

    st.markdown("---")
    st.subheader("Select Pet Record")
    
    pet_names = [f"[{p.get('category_tag', p['species'].upper())}] {p['name']} ({p['breed']})" for p in st.session_state["profiles"]]
    pet_ids = [p["id"] for p in st.session_state["profiles"]]
    
    current_idx = pet_ids.index(st.session_state["selected_pet_id"]) if st.session_state["selected_pet_id"] in pet_ids else 0
    selected_option = st.selectbox("Active Profile", range(len(pet_names)), format_func=lambda x: pet_names[x], index=current_idx)
    st.session_state["selected_pet_id"] = pet_ids[selected_option]
    current_pet = get_current_pet()

    st.markdown("---")
    st.subheader("Quick Profile Summary")
    st.caption(f"Species: `{current_pet['species']}` | Gender: `{current_pet.get('gender', 'N/A')}`")
    st.caption(f"Age: `{current_pet['age_years']} Years` | Weight: `{current_pet['weight_kg']} kg`")
    st.caption(f"Microchip ID: `{current_pet.get('microchip_id', 'N/A')}`")
    st.caption(f"Diet Type: `{current_pet.get('diet_type', 'Standard')}`")

    st.markdown("---")
    if st.button("Reload Records Database", use_container_width=True):
        st.session_state["records_df"] = load_pet_records(profiles=st.session_state["profiles"])
        st.success("Database refreshed!")
        st.rerun()


# Filter pet's historical data
pet_records = st.session_state["records_df"][st.session_state["records_df"]["pet_id"] == current_pet["id"]].copy()
analytics = compute_pet_analytics(pet_records, current_pet)
energy_reqs = calculate_energy_requirements(current_pet["weight_kg"], current_pet["species"])
bcs_label, bcs_color = classify_body_condition(analytics.get("current_weight", current_pet["weight_kg"]), current_pet.get("target_weight_kg", current_pet["weight_kg"]))


# ==========================================
# MAIN DASHBOARD HEADER
# ==========================================
st.markdown(f"""
<div class="main-header">
    <div>
        <h1 style="margin:0; font-size:1.8rem; font-weight:800; color:#FAFAFA;">
            [{current_pet.get('category_tag', current_pet['species'].upper())}] {current_pet['name']} — Care & Health Dashboard
        </h1>
        <p style="margin:4px 0 0 0; color:#A0AAB7; font-size:0.95rem;">
            {current_pet['breed']} | {current_pet.get('gender', 'N/A')} | {current_pet['age_years']} Years Old | Microchip: <code>{current_pet.get('microchip_id', 'N/A')}</code>
        </p>
    </div>
    <div style="text-align:right;">
        <span class="badge badge-valid">● ACTIVE PATIENT</span>
        <span class="badge badge-info">{bcs_label}</span>
        <div style="margin-top:6px; color:#A0AAB7; font-size:0.8rem;">
            Primary Vet: {current_pet.get('vet_contacts', {}).get('primary_vet', 'N/A')}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# TOP QUICK METRIC CARDS
# ==========================================
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    curr_wt = analytics.get("current_weight", current_pet["weight_kg"])
    tgt_wt = current_pet.get("target_weight_kg", curr_wt)
    wt_diff = analytics.get("weight_diff", 0.0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current Weight</div>
        <div class="metric-value" style="color:#51CF66;">{curr_wt:.1f} <span style="font-size:0.9rem;">kg</span></div>
        <div class="metric-sub" style="color:#A0AAB7;">Target: {tgt_wt} kg ({wt_diff:+.1f} kg)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Body Condition</div>
        <div class="metric-value" style="color:{bcs_color}; font-size:1.25rem;">{bcs_label.split(' ')[0]}</div>
        <div class="metric-sub" style="color:#A0AAB7;">{bcs_label.split('(')[1].replace(')', '')}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_cal = analytics.get("avg_calories", current_pet.get("daily_target_calories_kcal", 1200))
    tgt_cal = current_pet.get("daily_target_calories_kcal", 1200)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Daily Intake</div>
        <div class="metric-value" style="color:#FCC419;">{avg_cal:.0f} <span style="font-size:0.9rem;">kcal</span></div>
        <div class="metric-sub" style="color:#A0AAB7;">Target: {tgt_cal} kcal</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_ex = analytics.get("avg_exercise_min", current_pet.get("daily_target_exercise_min", 60))
    tgt_ex = current_pet.get("daily_target_exercise_min", 60)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Daily Exercise</div>
        <div class="metric-value" style="color:#339AF0;">{avg_ex:.0f} <span style="font-size:0.9rem;">min</span></div>
        <div class="metric-sub" style="color:#A0AAB7;">Target: {tgt_ex} min/day</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    avg_wtr = analytics.get("avg_water", current_pet.get("daily_target_water_ml", 1500))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Hydration Avg</div>
        <div class="metric-value" style="color:#339AF0;">{avg_wtr:.0f} <span style="font-size:0.9rem;">mL</span></div>
        <div class="metric-sub" style="color:#A0AAB7;">Target: {current_pet.get('daily_target_water_ml', 1500)} mL</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    score = analytics.get("care_score", 90.0)
    sc_color = "#51CF66" if score >= 80 else "#FCC419" if score >= 65 else "#FF6B6B"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Care Compliance</div>
        <div class="metric-value" style="color:{sc_color};">{score:.0f} <span style="font-size:0.9rem;">/100</span></div>
        <div class="metric-sub" style="color:#A0AAB7;">Status: {'Optimal' if score >= 80 else 'Attention'}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# TABBED NAVIGATION INTERFACE
# ==========================================
tab_overview, tab_medical, tab_nutrition, tab_exercise, tab_analytics, tab_records = st.tabs([
    "Care & Health Overview",
    "Medical Records & Vaccines",
    "Nutrition & Diet Planner",
    "Activity & Exercise Logger",
    "Growth & Health Analytics",
    "Database & Profile Management"
])


# ----------------------------------------------------
# TAB 1: CARE & HEALTH OVERVIEW
# ----------------------------------------------------
with tab_overview:
    ov_c1, ov_c2 = st.columns([1.5, 1])

    with ov_c1:
        st.markdown("### Weight Progression & Target Trajectory")
        fig_wt = plot_weight_trajectory(pet_records, current_pet)
        st.pyplot(fig_wt, use_container_width=True)

    with ov_c2:
        st.markdown("### Care Compliance Radar")
        fig_rad = plot_care_radar(analytics)
        st.pyplot(fig_rad, use_container_width=True)

    st.markdown("---")
    st.markdown("### Upcoming Medical & Preventative Milestones")
    
    med_col1, med_col2, med_col3 = st.columns(3)
    
    with med_col1:
        st.markdown("""
        <div class="content-card">
            <h4 style="margin:0 0 10px 0; color:#51CF66;">[VACCINATION STATUS]</h4>
        """, unsafe_allow_html=True)
        for v in current_pet.get("vaccinations", []):
            st.markdown(f"**{v['vaccine']}**: Valid until `{v['valid_until']}`")
        st.markdown("</div>", unsafe_allow_html=True)

    with med_col2:
        st.markdown("""
        <div class="content-card">
            <h4 style="margin:0 0 10px 0; color:#339AF0;">[ACTIVE MEDICATIONS]</h4>
        """, unsafe_allow_html=True)
        for m in current_pet.get("medications", []):
            st.markdown(f"**{m['name']}**: {m['dosage']} (`Next: {m['next_due']}`)")
        st.markdown("</div>", unsafe_allow_html=True)

    with med_col3:
        st.markdown("""
        <div class="content-card">
            <h4 style="margin:0 0 10px 0; color:#FCC419;">[ALLERGIES & CARE ALERTS]</h4>
        """, unsafe_allow_html=True)
        for a in current_pet.get("allergies", ["None Recorded"]):
            st.markdown(f"- {a}")
        st.markdown(f"Clinic: `{current_pet.get('vet_contacts', {}).get('clinic', 'N/A')}`")
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 2: MEDICAL RECORDS & VACCINATION PASSPORT
# ----------------------------------------------------
with tab_medical:
    st.markdown("### Medical History, Vaccinations & Prescriptions")
    
    med_t1, med_t2 = st.columns([1.2, 1])

    with med_t1:
        st.markdown("#### Vaccination Passport")
        vac_df = pd.DataFrame(current_pet.get("vaccinations", []))
        if not vac_df.empty:
            st.dataframe(vac_df, use_container_width=True, hide_index=True)
        else:
            st.info("No vaccination records logged yet.")

        st.markdown("#### Active Prescriptions & Preventatives")
        meds_df = pd.DataFrame(current_pet.get("medications", []))
        if not meds_df.empty:
            st.dataframe(meds_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active medications.")

    with med_t2:
        st.markdown("#### Record New Vaccination / Treatment")
        with st.form("add_vaccine_form"):
            v_name = st.text_input("Vaccine / Treatment Name", "Bordetella")
            v_date = st.date_input("Date Administered", datetime.now().date())
            v_valid = st.date_input("Valid Until", datetime.now().date() + timedelta(days=365))
            v_status = st.selectbox("Status", ["Valid", "Pending Booster", "Expired"])
            
            submit_vax = st.form_submit_button("Save Vaccine Entry", use_container_width=True)
            if submit_vax:
                if "vaccinations" not in current_pet:
                    current_pet["vaccinations"] = []
                current_pet["vaccinations"].append({
                    "vaccine": v_name,
                    "date_administered": str(v_date),
                    "valid_until": str(v_valid),
                    "status": v_status
                })
                save_pet_profiles(st.session_state["profiles"])
                st.success(f"Vaccination record for {v_name} saved successfully!")
                st.rerun()

        st.markdown("#### Add Medication / Supplement")
        with st.form("add_med_form"):
            m_name = st.text_input("Medication Name", "Flea & Tick Chewable")
            m_dose = st.text_input("Dosage Instructions", "1 tablet monthly")
            m_due = st.text_input("Next Schedule", "2026-09-20")
            
            submit_med = st.form_submit_button("Save Medication Entry", use_container_width=True)
            if submit_med:
                if "medications" not in current_pet:
                    current_pet["medications"] = []
                current_pet["medications"].append({
                    "name": m_name,
                    "dosage": m_dose,
                    "next_due": m_due
                })
                save_pet_profiles(st.session_state["profiles"])
                st.success(f"Medication {m_name} logged!")
                st.rerun()


# ----------------------------------------------------
# TAB 3: NUTRITION & DIET PLANNER
# ----------------------------------------------------
with tab_nutrition:
    st.markdown("### Veterinary Energy & Caloric Requirement Calculations")
    
    nut_c1, nut_c2, nut_c3 = st.columns(3)
    with nut_c1:
        st.metric("Resting Energy Req. (RER)", f"{energy_reqs['rer_kcal']} kcal/day", help="RER = 70 * (Weight ^ 0.75)")
    with nut_c2:
        st.metric("Daily Energy Req. (DER)", f"{energy_reqs['der_kcal']} kcal/day", f"{energy_reqs['multiplier']}x RER multiplier")
    with nut_c3:
        st.metric("Target Water Hydration", f"{energy_reqs['daily_water_ml']} mL/day", "~60 mL per kg bodyweight")

    st.markdown("---")
    st.markdown("### Caloric Balance: Intake vs Expenditure (Last 3 Weeks)")
    fig_cal = plot_energy_balance(pet_records)
    st.pyplot(fig_cal, use_container_width=True)

    st.markdown("---")
    st.markdown("### Quick Log: Today's Feeding & Water Intake")
    with st.form("quick_log_nutrition"):
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        with q_c1:
            log_date = st.date_input("Date", datetime.now().date(), key="nut_date")
        with q_c2:
            log_cals = st.number_input("Calories Consumed (kcal)", value=int(current_pet.get("daily_target_calories_kcal", 1200)), step=50)
        with q_c3:
            log_wtr = st.number_input("Water Consumed (mL)", value=int(current_pet.get("daily_target_water_ml", 1500)), step=50)
        with q_c4:
            log_wt_today = st.number_input("Recorded Weight (kg)", value=float(current_pet.get("weight_kg", 20.0)), step=0.1)

        sub_nut_log = st.form_submit_button("Submit Nutrition Log", use_container_width=True)
        if sub_nut_log:
            new_rec = {
                "date": str(log_date),
                "pet_id": current_pet["id"],
                "pet_name": current_pet["name"],
                "weight_kg": log_wt_today,
                "calories_consumed_kcal": log_cals,
                "water_intake_ml": log_wtr,
                "exercise_duration_min": int(current_pet.get("daily_target_exercise_min", 60)),
                "calories_burned_kcal": int(70 * (log_wt_today ** 0.75) + 200),
                "mood_score": 5,
                "health_status": "Healthy",
                "notes": "Daily feeding log"
            }
            st.session_state["records_df"] = add_daily_record(new_rec)
            # update profile weight
            current_pet["weight_kg"] = log_wt_today
            save_pet_profiles(st.session_state["profiles"])
            st.success("Nutrition and weight log updated successfully!")
            st.rerun()


# ----------------------------------------------------
# TAB 4: ACTIVITY & EXERCISE LOGGER
# ----------------------------------------------------
with tab_exercise:
    st.markdown("### Activity & Exercise Tracking")
    
    fig_ex = plot_exercise_and_hydration(pet_records, current_pet)
    st.pyplot(fig_ex, use_container_width=True)

    st.markdown("---")
    st.markdown("### Log Activity & Exercise Session")
    
    with st.form("log_exercise_form"):
        e_c1, e_c2, e_c3 = st.columns(3)
        with e_c1:
            ex_date = st.date_input("Activity Date", datetime.now().date(), key="ex_date")
            ex_type = st.selectbox("Activity Type", ["Neighborhood Walk", "Park Fetch / Running", "Agility / Training", "Indoor Play", "Swimming"])
        with e_c2:
            ex_duration = st.number_input("Duration (Minutes)", value=45, min_value=5, max_value=300, step=5)
            intensity = st.selectbox("Intensity Level", ["Light (3 kcal/min)", "Moderate (5 kcal/min)", "Vigorous (8 kcal/min)"])
        with e_c3:
            mood_val = st.slider("Pet Mood Score (1=Tired to 5=Energetic)", min_value=1, max_value=5, value=5)
            ex_notes = st.text_input("Session Notes", "Good energy, completed routine without fatigue")

        sub_ex = st.form_submit_button("Record Exercise Session", use_container_width=True)
        if sub_ex:
            intensity_rate = 3.0 if "Light" in intensity else 5.0 if "Moderate" in intensity else 8.0
            est_burn = int(70 * (current_pet["weight_kg"] ** 0.75) + (ex_duration * intensity_rate))
            
            new_rec = {
                "date": str(ex_date),
                "pet_id": current_pet["id"],
                "pet_name": current_pet["name"],
                "weight_kg": current_pet["weight_kg"],
                "calories_consumed_kcal": int(current_pet.get("daily_target_calories_kcal", 1200)),
                "water_intake_ml": int(current_pet.get("daily_target_water_ml", 1500)),
                "exercise_duration_min": ex_duration,
                "calories_burned_kcal": est_burn,
                "mood_score": mood_val,
                "health_status": "Healthy",
                "notes": f"[{ex_type}] {ex_notes}"
            }
            st.session_state["records_df"] = add_daily_record(new_rec)
            st.success(f"Recorded {ex_duration} mins of {ex_type} ({est_burn} kcal total daily expenditure)!")
            st.rerun()


# ----------------------------------------------------
# TAB 5: GROWTH & HEALTH ANALYTICS
# ----------------------------------------------------
with tab_analytics:
    st.markdown("### Growth Trajectory & Statistical Health Analytics")
    
    an_c1, an_c2 = st.columns([1.2, 1])

    with an_c1:
        fig_corr = plot_correlations(pet_records)
        st.pyplot(fig_corr, use_container_width=True)

    with an_c2:
        st.markdown("#### NumPy Weight Progression Modeling")
        st.write(f"- **Current Recorded Weight**: `{analytics.get('current_weight', current_pet['weight_kg']):.2f} kg`")
        st.write(f"- **Target Goal Weight**: `{current_pet.get('target_weight_kg', current_pet['weight_kg']):.2f} kg`")
        st.write(f"- **Deviation from Target**: `{analytics.get('weight_diff', 0.0):+.2f} kg`")
        st.write(f"- **Estimated Weekly Trajectory Slope**: `{analytics.get('weekly_weight_change', 0.0):+.3f} kg/week`")
        
        st.markdown("#### Care & Compliance Summary")
        summary_table = {
            "Metric": ["Nutrition Adherence", "Exercise Adherence", "Hydration Adherence", "Overall Care Score"],
            "Score": [
                f"{analytics.get('cal_adherence', 0)}%",
                f"{analytics.get('exercise_adherence', 0)}%",
                f"{analytics.get('water_adherence', 0)}%",
                f"{analytics.get('care_score', 0)} / 100"
            ]
        }
        st.dataframe(pd.DataFrame(summary_table), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 30-Day Health & Mood History")
    st.dataframe(
        pet_records[["date", "weight_kg", "calories_consumed_kcal", "water_intake_ml", "exercise_duration_min", "calories_burned_kcal", "mood_score", "health_status", "notes"]]
        .sort_values("date", ascending=False)
        .head(30),
        use_container_width=True,
        hide_index=True
    )


# ----------------------------------------------------
# TAB 6: DATABASE & PROFILE MANAGEMENT
# ----------------------------------------------------
with tab_records:
    st.markdown("### Database & Profile Management")
    
    with st.form("edit_profile_form"):
        st.subheader(f"Edit Profile: {current_pet['name']}")
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            new_name = st.text_input("Pet Name", value=current_pet["name"])
            new_species = st.selectbox("Species", ["Dog", "Cat", "Bird", "Other"], index=0 if current_pet["species"] == "Dog" else 1)
            new_breed = st.text_input("Breed", value=current_pet["breed"])
            new_gender = st.selectbox("Gender", ["Male", "Female"], index=0 if current_pet.get("gender") == "Male" else 1)
        with p_c2:
            new_age = st.number_input("Age (Years)", value=float(current_pet["age_years"]), step=0.5)
            new_weight = st.number_input("Current Weight (kg)", value=float(current_pet["weight_kg"]), step=0.1)
            new_target_weight = st.number_input("Target Weight (kg)", value=float(current_pet.get("target_weight_kg", current_pet["weight_kg"])), step=0.1)
            new_microchip = st.text_input("Microchip ID", value=current_pet.get("microchip_id", ""))
        with p_c3:
            new_cals = st.number_input("Target Calories (kcal/day)", value=int(current_pet.get("daily_target_calories_kcal", 1200)), step=50)
            new_water = st.number_input("Target Water (mL/day)", value=int(current_pet.get("daily_target_water_ml", 1500)), step=50)
            new_ex = st.number_input("Target Exercise (min/day)", value=int(current_pet.get("daily_target_exercise_min", 60)), step=5)
            new_diet = st.text_input("Diet Type", value=current_pet.get("diet_type", "Standard Kibble"))

        save_p_btn = st.form_submit_button("Save Profile Changes", use_container_width=True)
        if save_p_btn:
            for p in st.session_state["profiles"]:
                if p["id"] == current_pet["id"]:
                    p["name"] = new_name
                    p["species"] = new_species
                    p["breed"] = new_breed
                    p["gender"] = new_gender
                    p["age_years"] = new_age
                    p["weight_kg"] = new_weight
                    p["target_weight_kg"] = new_target_weight
                    p["microchip_id"] = new_microchip
                    p["daily_target_calories_kcal"] = new_cals
                    p["daily_target_water_ml"] = new_water
                    p["daily_target_exercise_min"] = new_ex
                    p["diet_type"] = new_diet
                    break
            save_pet_profiles(st.session_state["profiles"])
            st.success(f"Profile for {new_name} updated successfully!")
            st.rerun()

    st.markdown("---")
    st.markdown("#### Register New Pet Profile")
    with st.expander("Add New Pet to Database"):
        with st.form("add_pet_form"):
            an_name = st.text_input("Name", "Charlie")
            an_species = st.selectbox("Species", ["Dog", "Cat", "Other"], key="add_sp")
            an_breed = st.text_input("Breed", "Labrador Retriever")
            an_gender = st.selectbox("Gender", ["Male", "Female"], key="add_gen")
            an_age = st.number_input("Age", 2.0)
            an_weight = st.number_input("Weight (kg)", 25.0)
            an_chip = st.text_input("Microchip ID", f"98514100{np.random.randint(100, 999)}")
            
            sub_new = st.form_submit_button("Register Pet", use_container_width=True)
            if sub_new:
                new_p = {
                    "id": f"PET-{len(st.session_state['profiles']) + 1:03d}",
                    "name": an_name,
                    "species": an_species,
                    "breed": an_breed,
                    "gender": an_gender,
                    "age_years": an_age,
                    "weight_kg": an_weight,
                    "target_weight_kg": an_weight,
                    "microchip_id": an_chip,
                    "category_tag": "CANINE" if an_species == "Dog" else "FELINE",
                    "diet_type": "Standard Formula",
                    "daily_target_calories_kcal": int(70 * (an_weight ** 0.75) * 1.6),
                    "daily_target_water_ml": int(an_weight * 60),
                    "daily_target_exercise_min": 60,
                    "allergies": ["None Recorded"],
                    "medications": [],
                    "vaccinations": [],
                    "vet_contacts": {"clinic": "Community Veterinary Hospital", "primary_vet": "Duty Vet, DVM", "phone": "+1-555-010-0000"}
                }
                st.session_state["profiles"].append(new_p)
                save_pet_profiles(st.session_state["profiles"])
                st.success(f"Added {an_name} successfully!")
                st.rerun()

    st.markdown("---")
    st.markdown("#### Export / Import Care Database")
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        csv_data = st.session_state["records_df"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Complete CSV Records Database",
            data=csv_data,
            file_name=f"pawpulse_pet_records_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with exp_c2:
        up_file = st.file_uploader("Import CSV Records", type=["csv"])
        if up_file is not None:
            try:
                uploaded_df = pd.read_csv(up_file)
                st.session_state["records_df"] = uploaded_df
                save_pet_records(uploaded_df)
                st.success(f"Successfully loaded {len(uploaded_df)} records!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading CSV: {e}")


# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.85rem; padding: 15px;'>"
    "<strong>PawPulse Pet Health & Care Management System</strong> | Built with Streamlit, NumPy, Pandas & Matplotlib | Software Care & Analytics Platform"
    "</div>",
    unsafe_allow_html=True
)
