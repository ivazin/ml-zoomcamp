"""
Homework 2 - Machine Learning Zoomcamp (2026 Cohort)
Linear Regression and Regularization
"""

import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
FEATURES = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
    "fuel_efficiency_mpg",
]


def train_linear_regression(X: np.ndarray, y: np.ndarray):
    """Train linear regression without regularization."""
    ones = np.ones(X.shape[0])
    X_full = np.column_stack([ones, X])
    XTX = X_full.T.dot(X_full)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X_full.T).dot(y)
    return w[0], w[1:]


def train_linear_regression_reg(X: np.ndarray, y: np.ndarray, r: float = 0.0):
    """Train linear regression with L2 (Ridge) regularization."""
    ones = np.ones(X.shape[0])
    X_full = np.column_stack([ones, X])
    XTX = X_full.T.dot(X_full)
    reg = r * np.eye(XTX.shape[0])
    XTX_reg = XTX + reg
    XTX_inv = np.linalg.inv(XTX_reg)
    w = XTX_inv.dot(X_full.T).dot(y)
    return w[0], w[1:]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    error = y_pred - y_true
    mse = (error**2).mean()
    return float(np.sqrt(mse))


def main():
    print("=" * 60)
    print("Homework 2: Linear Regression")
    print("=" * 60)

    print(f"\nDownloading dataset from: {DATA_URL}...")
    df_raw = pd.read_csv(DATA_URL)
    df = df_raw[FEATURES].copy()
    print("Dataset loaded successfully.")

    # EDA
    print(f"\n[EDA] fuel_efficiency_mpg describe:")
    print(df["fuel_efficiency_mpg"].describe())

    # Question 1: Missing values column
    missing_counts = df.isnull().sum()
    col_missing = missing_counts[missing_counts > 0].index[0]
    print(f"\n[Q1] Column with missing values: '{col_missing}' (Count: {missing_counts[col_missing]})")

    # Question 2: Median horsepower
    median_hp = int(df["horsepower"].median())
    print(f"\n[Q2] Median (50th percentile) for 'horsepower': {median_hp}")

    # Prepare and split dataset
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(42)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].copy()
    df_val = df.iloc[idx[n_train : n_train + n_val]].copy()
    df_test = df.iloc[idx[n_train + n_val :]].copy()

    y_train = df_train["fuel_efficiency_mpg"].values
    y_val = df_val["fuel_efficiency_mpg"].values
    y_test = df_test["fuel_efficiency_mpg"].values

    del df_train["fuel_efficiency_mpg"]
    del df_val["fuel_efficiency_mpg"]
    del df_test["fuel_efficiency_mpg"]

    # Question 3: Imputation with 0 vs mean
    # Option A: fill with 0
    X_train_zero = df_train.fillna(0).values
    w0_zero, w_zero = train_linear_regression(X_train_zero, y_train)
    X_val_zero = df_val.fillna(0).values
    y_pred_zero = w0_zero + X_val_zero.dot(w_zero)
    rmse_zero = round(rmse(y_val, y_pred_zero), 3)

    # Option B: fill with mean (training set only!)
    hp_train_mean = df_train["horsepower"].mean()
    X_train_mean = df_train.fillna(hp_train_mean).values
    w0_mean, w_mean = train_linear_regression(X_train_mean, y_train)
    X_val_mean = df_val.fillna(hp_train_mean).values
    y_pred_mean = w0_mean + X_val_mean.dot(w_mean)
    rmse_mean = round(rmse(y_val, y_pred_mean), 3)

    if rmse_zero < rmse_mean:
        best_imputation = "With 0"
    elif rmse_mean < rmse_zero:
        best_imputation = "With mean"
    else:
        best_imputation = "Both are equally good"

    print(f"\n[Q3] RMSE with 0: {rmse_zero:.3f}")
    print(f"     RMSE with mean: {rmse_mean:.3f}")
    print(f"     Better option: {best_imputation}")

    # Question 4: Regularization parameter r
    r_candidates = [0, 0.01, 0.1, 1, 5, 10, 100]
    best_r = None
    best_rmse_val = float("inf")

    print(f"\n[Q4] Evaluating regularized linear regression (filling NAs with 0):")
    for r in r_candidates:
        w0_r, w_r = train_linear_regression_reg(X_train_zero, y_train, r=r)
        y_pred_r = w0_r + X_val_zero.dot(w_r)
        score = round(rmse(y_val, y_pred_r), 4)
        print(f"     r={r:>5}: RMSE = {score:.4f}")
        if score < best_rmse_val:
            best_rmse_val = score
            best_r = r

    print(f"     Best r: {best_r} (lowest RMSE: {best_rmse_val:.4f})")

    # Question 5: Seed variation and stability (std of RMSE)
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    seed_scores = []

    for seed in seeds:
        np.random.seed(seed)
        seed_idx = np.arange(n)
        np.random.shuffle(seed_idx)

        d_tr = df.iloc[seed_idx[:n_train]].copy()
        d_v = df.iloc[seed_idx[n_train : n_train + n_val]].copy()

        y_tr = d_tr["fuel_efficiency_mpg"].values
        y_v = d_v["fuel_efficiency_mpg"].values

        del d_tr["fuel_efficiency_mpg"]
        del d_v["fuel_efficiency_mpg"]

        X_tr = d_tr.fillna(0).values
        w0_s, w_s = train_linear_regression(X_tr, y_tr)

        X_v = d_v.fillna(0).values
        y_pred_s = w0_s + X_v.dot(w_s)
        seed_scores.append(rmse(y_v, y_pred_s))

    std_val = round(float(np.std(seed_scores)), 3)
    print(f"\n[Q5] Standard deviation across seeds 0-9: {std_val:.3f}")

    # Question 6: Train on full train (train + val) with seed 9 and r=0.001
    np.random.seed(9)
    idx_q6 = np.arange(n)
    np.random.shuffle(idx_q6)

    df_train_q6 = df.iloc[idx_q6[:n_train]].copy()
    df_val_q6 = df.iloc[idx_q6[n_train : n_train + n_val]].copy()
    df_test_q6 = df.iloc[idx_q6[n_train + n_val :]].copy()

    df_full_train = pd.concat([df_train_q6, df_val_q6], axis=0)
    y_full_train = df_full_train["fuel_efficiency_mpg"].values
    y_test_q6 = df_test_q6["fuel_efficiency_mpg"].values

    del df_full_train["fuel_efficiency_mpg"]
    del df_test_q6["fuel_efficiency_mpg"]

    X_full_train = df_full_train.fillna(0).values
    w0_final, w_final = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)

    X_test_q6 = df_test_q6.fillna(0).values
    y_test_pred = w0_final + X_test_q6.dot(w_final)
    test_rmse = round(rmse(y_test_q6, y_test_pred), 3)
    print(f"\n[Q6] Final model (seed 9, full train, r=0.001) Test RMSE: {test_rmse:.3f}")

    # Summary Table
    print("\n" + "=" * 60)
    print("SUMMARY OF HW-02 ANSWERS:")
    print("=" * 60)
    print(f"Q1: {col_missing}")
    print(f"Q2: {median_hp}")
    print(f"Q3: {best_imputation}")
    print(f"Q4: {best_r}")
    print(f"Q5: {std_val:.3f}")
    print(f"Q6: {test_rmse:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
