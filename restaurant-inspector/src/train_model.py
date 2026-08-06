"""
Trains and compares baseline classifiers to predict inspection failure risk.
Uses cross-validation given the modest sample size (~466 rows).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = Path(__file__).parent.parent / "data" / "pinellas_clean.csv"
MODEL_PATH = Path(__file__).parent.parent / "data" / "model.joblib"


def main():
    df = pd.read_csv(DATA_PATH)

    # Feature set: aggregate counts only (not raw per-violation-code columns,
    # to avoid overfitting on a small sample — those raw codes are IS the
    # definition of "failed" in many cases, so we use summary stats + a few
    # specific high-signal violation flags instead).
    feature_cols = [
        "Number of High Priority Violations",
        "Number of Intermediate Violations",
        "Number of Basic Violations",
        "high_risk_violation_count",
        "distinct_violation_categories",
    ]
    X = df[feature_cols]
    y = df["failed"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=42
        ),
    }

    print(f"Dataset: {len(df)} rows, {y.mean():.1%} failure rate\n")
    best_name, best_score, best_model = None, -1, None
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        print(f"{name}: ROC-AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")
        if scores.mean() > best_score:
            best_name, best_score, best_model = name, scores.mean(), model

    # Fit best model on full data and save
    best_model.fit(X, y)
    joblib.dump({"model": best_model, "features": feature_cols}, MODEL_PATH)
    print(f"\nBest model: {best_name} (saved to {MODEL_PATH})")

    # Feature importance / coefficients for interpretability
    if best_name == "Random Forest":
        importances = pd.Series(best_model.feature_importances_, index=feature_cols)
        print("\nFeature importances:")
        print(importances.sort_values(ascending=False))


if __name__ == "__main__":
    main()
