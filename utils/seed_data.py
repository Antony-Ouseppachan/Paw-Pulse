"""
Seed generator script to create data/pet_records.csv
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_manager import load_pet_profiles, generate_sample_historical_records, save_pet_records

def main():
    profiles = load_pet_profiles()
    df = generate_sample_historical_records(profiles, days=60)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "pet_records.csv")
    save_pet_records(df, out_path)
    print(f"Generated {len(df)} care and health records across {len(profiles)} pets at {out_path}")

if __name__ == "__main__":
    main()
