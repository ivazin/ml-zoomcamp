"""Reproducible HW-03 solution. Run: uv run python solution.py."""
from pathlib import Path
import hashlib
import json
import platform

import pandas as pd
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/course_lead_scoring_2026.csv"


def prepare():
    # keep_default_na=False preserves the literal NA category;
    # empty fields are still treated as missing values.
    raw = pd.read_csv(DATA, keep_default_na=False, na_values=[""])
    categorical = raw.select_dtypes(include=["object", "str"]).columns.tolist()
    numerical = [c for c in raw.columns if c not in categorical and c != "converted"]
    df = raw.copy()
    df[categorical] = df[categorical].fillna("NA")
    df[numerical] = df[numerical].fillna(0.0)
    full_train, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(full_train, test_size=0.25, random_state=42)
    y_train, y_val = train.converted.to_numpy(), val.converted.to_numpy()
    train, val, test = [part.drop(columns="converted") for part in (train, val, test)]
    return raw, df, categorical, numerical, train, val, test, y_train, y_val


def fit_accuracy(train, val, y_train, y_val, features, C=1.0):
    # Fit the encoder only on the training set. Keep numerical features unscaled:
    # this preserves the homework setup and the effect of C on the original scale.
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train[features].to_dict(orient="records"))
    X_val = dv.transform(val[features].to_dict(orient="records"))
    model = LogisticRegression(solver="liblinear", C=C, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_val)[:, 1] >= 0.5
    return float(accuracy_score(y_val, predictions))


def solve():
    raw, df, categorical, numerical, train, val, test, y_train, y_val = prepare()
    corr = df[numerical].corr()
    pairs = [("interaction_count", "lead_score"), ("number_of_courses_viewed", "lead_score"),
             ("number_of_courses_viewed", "interaction_count"), ("annual_income", "interaction_count")]
    correlations = {f"{a} / {b}": float(corr.loc[a, b]) for a, b in pairs}
    mi = {c: float(mutual_info_score(train[c], y_train)) for c in categorical}
    mi_rounded = {c: round(score, 2) for c, score in mi.items()}
    features = train.columns.tolist()
    baseline = fit_accuracy(train, val, y_train, y_val, features)
    elimination = {}
    for feature in features:
        accuracy = fit_accuracy(train, val, y_train, y_val, [c for c in features if c != feature])
        elimination[feature] = {"accuracy": accuracy, "difference": baseline - accuracy}
    candidates = ["lead_source", "number_of_courses_viewed", "interaction_count"]
    regularization = {}
    for C in [0.000001, 0.00001, 0.0001, 0.001]:
        accuracy = fit_accuracy(train, val, y_train, y_val, features, C=C)
        regularization[str(C)] = {"accuracy": accuracy, "rounded": round(accuracy, 3)}
    best_C = min(regularization, key=lambda c: (-regularization[c]["rounded"], float(c)))
    result = {
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "shape": list(df.shape), "split_sizes": {"train": len(train), "validation": len(val), "test": len(test)},
        "missing_before": {c: int(n) for c, n in raw.isna().sum().items()},
        "industry_counts": {str(c): int(n) for c, n in df.industry.value_counts().items()},
        "correlations": correlations, "mutual_information": mi, "mutual_information_rounded": mi_rounded,
        "baseline_accuracy": baseline, "elimination": elimination, "regularization": regularization,
        "answers": {"Q1": df.industry.mode().iloc[0], "Q2": max(correlations, key=correlations.get),
                    "Q3": max(mi_rounded, key=mi_rounded.get), "Q4": round(baseline, 2),
                    "Q5": min(candidates, key=lambda c: elimination[c]["difference"]), "Q6": float(best_C)},
    }
    return result, corr


if __name__ == "__main__":
    result, corr = solve()
    (ROOT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    corr.to_csv(ROOT / "correlation_matrix.csv")
    print(json.dumps(result, ensure_ascii=False, indent=2))
