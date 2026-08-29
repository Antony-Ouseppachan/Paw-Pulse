# PawPulse — Pet Health, Nutrition & Care Management System

An intelligent, software-driven pet health monitoring, nutrition planning, and care analytics platform built with **Streamlit**, **NumPy**, **Pandas**, and **Matplotlib**.

![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Statistical%20Math-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Handling-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Analytics%20Viz-11557c?style=for-the-badge)

---

## Key Modules & Features

1. **Pet Profiles & Medical Passport**:
   - Complete pet profiles (Breed, Gender, Age, Weight, Microchip ID, Allergies).
   - Vaccination passport tracking validity dates, statuses, and booster schedules.
   - Active prescription and preventative medication scheduler.

2. **Veterinary Nutrition & Diet Planning**:
   - Computes veterinary Resting Energy Requirement (RER) and Daily Energy Requirement (DER) using **NumPy** power equations ($RER = 70 \times \text{Weight}^{0.75}$).
   - Caloric balance tracking (Daily Caloric Intake vs. Total Energy Burned).
   - Daily feeding and hydration volume logging.

3. **Activity & Exercise Management**:
   - Logging walks, play sessions, and training routines.
   - Activity intensity tracking and caloric burn calculations.
   - 14-day exercise consistency and hydration trend charts.

4. **Growth & Health Analytics**:
   - Weight trajectory linear regression modeling using **NumPy** (`np.polyfit`) to track weekly weight velocity towards target goal.
   - Body Condition Score (BCS) categorization.
   - Statistical correlation matrix across nutrition, activity, hydration, and pet mood.
   - Care Compliance Index radar chart (0 - 100).

5. **Database & Data Handling**:
   - Historical daily records management with **Pandas**.
   - Export full historical care data to CSV.
   - Import external CSV records.
   - Add and update multi-pet profiles.

---

## Architecture & Technology Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend & UI** | `Streamlit` | Interactive web dashboard, metric cards, tabbed views |
| **Mathematical Engine** | `NumPy` | RER/DER formulas, linear weight regression, statistical modeling |
| **Data Handling** | `Pandas` | Historical logs, CSV database handling, data aggregation |
| **Visual Analytics** | `Matplotlib` | Weight progression charts, caloric balance, radar chart, correlation heatmap |

---

## Project Structure

```
lab10/
├── .streamlit/
│   └── config.toml             # Streamlit UI theme and server settings
├── data/
│   ├── pet_profiles.json       # Configured pet profiles, medical history & targets
│   └── pet_records.csv         # 60-day historical health, nutrition & exercise logs
├── utils/
│   ├── __init__.py
│   ├── data_manager.py         # CSV & JSON persistence, daily log updater
│   ├── analytics.py            # Veterinary RER/DER equations, linear regression & scoring
│   ├── visualizer.py           # Matplotlib charting routines & dark theme
│   └── seed_data.py            # Script to seed sample historical logs
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies for local & cloud deployment
└── README.md                   # Project documentation
```

---

## Getting Started Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Deployment to Streamlit Community Cloud

This project is pre-configured for direct zero-friction deployment to **[Streamlit Community Cloud](https://streamlit.io/cloud)**:

1. Push this repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in.
3. Select your GitHub repository, branch `main`, and set **Main file path** to `app.py`.
4. Click **"Deploy"**!
