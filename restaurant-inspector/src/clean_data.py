"""
Cleans raw Florida DBPR District 3 inspection extract, filters to Pinellas County,
and engineers features for modeling.

Usage:
    python3 src/clean_data.py
"""
import pandas as pd
from pathlib import Path
from violation_codes import VIOLATION_CODES, HIGH_RISK_VIOLATIONS

RAW_PATH = Path(__file__).parent.parent / "data" / "3fdinspi.csv"
OUT_PATH = Path(__file__).parent.parent / "data" / "pinellas_clean.csv"

# Dispositions that represent a "bad" outcome (some kind of failure/action needed)
FAIL_DISPOSITIONS = {
    "Warning Issued",
    "Administrative complaint recommended",
    "Emergency order recommended",
    "Call Back - Admin. complaint recommended",
    "Emergency Order Callback Not Complied",
}


def load_and_clean():
    df = pd.read_csv(RAW_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Filter to Pinellas County only
    df = df[df["County Name"] == "Pinellas"].copy()

    # Parse date
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], format="%m/%d/%Y")

    # Only keep initial visits (Visit Number == 1) to avoid double-counting callbacks
    # for the "does this restaurant tend to fail" question. Callbacks are kept
    # separately if needed for a "resolution speed" feature later.
    initial = df[df["Visit Number"] == 1].copy()

    # Target variable: did this inspection result in a failure-type outcome?
    initial["failed"] = initial["Inspection Disposition"].isin(FAIL_DISPOSITIONS).astype(int)

    # Violation columns
    viol_cols = [f"Violation {i:02d}" for i in range(1, 59)]
    for c in viol_cols:
        initial[c] = pd.to_numeric(initial[c], errors="coerce").fillna(0)

    # Feature: count of high-risk violation categories triggered
    high_risk_cols = [f"Violation {i:02d}" for i in HIGH_RISK_VIOLATIONS]
    initial["high_risk_violation_count"] = initial[high_risk_cols].sum(axis=1)

    # Feature: total distinct violation categories triggered (non-zero columns)
    initial["distinct_violation_categories"] = (initial[viol_cols] > 0).sum(axis=1)

    # Keep relevant columns for modeling + readability
    keep_cols = [
        "License Number", "Business (DBA-Does Business As) Name", "Location Address",
        "Location City", "Location Zip Code", "Inspection Number", "Inspection Type",
        "Inspection Disposition", "Inspection Date", "Number of Total Violations",
        "Number of High Priority Violations", "Number of Intermediate Violations",
        "Number of Basic Violations", "high_risk_violation_count",
        "distinct_violation_categories", "failed",
    ] + viol_cols

    cleaned = initial[keep_cols].sort_values("Inspection Date")
    cleaned.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(cleaned)} cleaned Pinellas inspection records to {OUT_PATH}")
    print(f"Failure rate: {cleaned['failed'].mean():.1%}")
    return cleaned


if __name__ == "__main__":
    load_and_clean()
