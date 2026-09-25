"""
Homework 1 - Machine Learning Zoomcamp (2026 Cohort)
Intro to Machine Learning & Exploratory Data Analysis
"""

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"


def main():
    print("=" * 60)
    print("Homework 1: Intro to Machine Learning")
    print("=" * 60)

    # Q1. Pandas version
    pandas_version = pd.__version__
    print(f"\n[Q1] Pandas Version: {pandas_version}")

    # Load Dataset
    print(f"\nFetching dataset from: {DATA_URL}...")
    df = pd.read_csv(DATA_URL)
    print("Dataset successfully loaded.")

    # Q2. Records count
    records_count = len(df)
    print(f"\n[Q2] Number of records: {records_count}")

    # Q3. Fuel types
    fuel_types_count = df["fuel_type"].nunique()
    fuel_types = list(df["fuel_type"].unique())
    print(f"\n[Q3] Number of fuel types: {fuel_types_count} (Values: {fuel_types})")

    # Q4. Missing values
    cols_with_missing = (df.isnull().sum() > 0).sum()
    missing_cols_detail = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    print(f"\n[Q4] Columns with missing values: {cols_with_missing} ({missing_cols_detail})")

    # Q5. Max fuel efficiency
    max_fuel_eff_asia = df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max()
    print(f"\n[Q5] Maximum fuel efficiency for cars from Asia: {max_fuel_eff_asia}")

    # Q6. Median value of horsepower
    median_before = df["horsepower"].median()
    mode_hp = df["horsepower"].mode()[0]
    hp_filled = df["horsepower"].fillna(mode_hp)
    median_after = hp_filled.median()

    if median_after > median_before:
        change_desc = "Yes, it increased"
    elif median_after < median_before:
        change_desc = "Yes, it decreased"
    else:
        change_desc = "No"

    print(f"\n[Q6] Horsepower median before filling: {median_before}")
    print(f"     Horsepower most frequent value (mode): {mode_hp}")
    print(f"     Horsepower median after filling with mode: {median_after}")
    print(f"     Has it changed? -> {change_desc}")

    # Q7. Sum of weights
    # 1. Select all the cars from Asia
    # 2. Select only columns vehicle_weight and model_year
    # 3. Select the first 7 values
    asia_subset = df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]].iloc[:7]

    # 4. Get underlying NumPy array X
    X = asia_subset.to_numpy()

    # 5. Matrix-matrix multiplication between X.T and X
    XTX = X.T.dot(X)

    # 6. Invert XTX
    XTX_inv = np.linalg.inv(XTX)

    # 7. Create y array
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

    # 8. Multiply inverse of XTX with X.T, then with y
    w = XTX_inv.dot(X.T).dot(y)

    # 9. Sum of all elements of w
    w_sum = w.sum()
    print(f"\n[Q7] Linear regression weights w: {w}")
    print(f"     Sum of all elements of w: {w_sum:.4f} (raw: {w_sum})")

    print("\n" + "=" * 60)
    print("SUMMARY OF ANSWERS:")
    print("=" * 60)
    print(f"Q1: {pandas_version}")
    print(f"Q2: {records_count}")
    print(f"Q3: {fuel_types_count}")
    print(f"Q4: {cols_with_missing}")
    print(f"Q5: {max_fuel_eff_asia}")
    print(f"Q6: {change_desc}")
    print(f"Q7: {w_sum:.3f} (approx 0.369)")
    print("=" * 60)


if __name__ == "__main__":
    main()
