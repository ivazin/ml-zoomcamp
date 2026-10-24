"""Run the HW-04 experiments without exporting result files."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
DATA_SHA256 = "2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9"
NUMERICAL = ["lead_score", "number_of_courses_viewed", "interaction_count", "annual_income"]


def load_data(path):
    """Download the pinned release if missing, then fill missing values."""
    path = Path(path)
    if not path.exists():
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded dataset does not match the pinned release.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("Local dataset does not match the pinned release.")
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df = raw.copy()
    categorical = df.select_dtypes(include=["object", "str"]).columns.tolist()
    df[categorical] = df[categorical].fillna("NA")
    df[NUMERICAL] = df[NUMERICAL].fillna(0.0)
    return raw, df


def split_data(df):
    full_train, test = train_test_split(df, test_size=0.2, random_state=1)
    train, val = train_test_split(full_train, test_size=0.25, random_state=1)
    return full_train, train, val, test


def predict_probabilities(train, val, C=1.0):
    """Fit a fresh encoder and model; keep converted out of the features."""
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train.drop(columns="converted").to_dict(orient="records"))
    X_val = dv.transform(val.drop(columns="converted").to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
    model.fit(X_train, train.converted.to_numpy())
    return model.predict_proba(X_val)[:, 1]


def feature_auc(train):
    rows = []
    for feature in NUMERICAL:
        score = train[feature].to_numpy()
        raw_auc = roc_auc_score(train.converted, score)
        inverted = raw_auc < 0.5
        adjusted_auc = roc_auc_score(train.converted, -score if inverted else score)
        rows.append({"feature": feature, "raw_auc": raw_auc,
                     "inverted": inverted, "auc": adjusted_auc})
    return pd.DataFrame(rows).set_index("feature")


def threshold_metrics(y_true, probabilities):
    """Compute confusion counts and metrics on the requested threshold grid."""
    y_true = np.asarray(y_true)
    rows = []
    for threshold in np.arange(101) / 100:
        predicted = probabilities >= threshold
        actual = y_true == 1
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        tn = int((~predicted & ~actual).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(threshold), "tp": tp, "fp": fp,
                     "fn": fn, "tn": tn, "precision": precision,
                     "recall": recall, "f1": f1, "gap": abs(precision - recall)})
    return pd.DataFrame(rows)


def select_thresholds(metrics):
    # Ignore the trivial precision=recall=0 intersection.
    eligible = metrics.loc[(metrics.precision != 0) | (metrics.recall != 0)]
    intersection = eligible.sort_values(["gap", "threshold"]).iloc[0]
    best_f1 = metrics.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
    return float(intersection.threshold), float(best_f1.threshold)


def cross_validate(full_train, C):
    splitter = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = []
    for train_idx, val_idx in splitter.split(full_train):
        train, val = full_train.iloc[train_idx], full_train.iloc[val_idx]
        probabilities = predict_probabilities(train, val, C=C)
        scores.append(float(roc_auc_score(val.converted, probabilities)))
    return np.asarray(scores)


def tune_regularization(full_train):
    rows = []
    for C in [0.000001, 0.001, 1.0]:
        scores = cross_validate(full_train, C)
        mean, std = float(scores.mean()), float(scores.std(ddof=0))
        rows.append({"C": C, "fold_scores": scores.tolist(), "mean_auc": mean,
                     "std_auc": std, "mean_rounded": round(mean, 3),
                     "std_rounded": round(std, 3)})
    return pd.DataFrame(rows)


def main():
    raw, df = load_data(Path(__file__).resolve().parent / "data/course_lead_scoring_2026.csv")
    full_train, train, val, test = split_data(df)
    print("Missing values before preparation:\n", raw.isna().sum())
    print("Train / validation / test:", len(train), len(val), len(test))
    features = feature_auc(train)
    probabilities = predict_probabilities(train, val)
    auc = float(roc_auc_score(val.converted, probabilities))
    metrics = threshold_metrics(val.converted, probabilities)
    intersection, f1_threshold = select_thresholds(metrics)
    tuning = tune_regularization(full_train)
    baseline_cv = tuning.loc[tuning.C == 1.0].iloc[0]
    best = tuning.sort_values(["mean_rounded", "std_rounded", "C"],
                              ascending=[False, True, True]).iloc[0]
    print("\nQ1 — numerical feature AUC:\n", features)
    print("\nQ2 — validation AUC:", auc, "rounded:", round(auc, 3))
    print("\nQ3/Q4 — threshold candidates:\n",
          metrics.loc[metrics.threshold.isin([intersection, f1_threshold])])
    print("\nQ5/Q6 — cross-validation:\n", tuning.to_string(index=False))
    print("\nComputed answers:")
    print("Q1:", features.auc.idxmax())
    print("Q2:", round(auc, 3))
    print("Q3:", intersection)
    print("Q4:", f1_threshold)
    print("Q5:", round(float(baseline_cv.std_auc), 3))
    print("Q6:", float(best.C))


if __name__ == "__main__":
    main()
