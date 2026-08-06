"""
Sets up a normalized SQLite database from the cleaned Pinellas inspection data.
Two tables: restaurants (one row per license) and inspections (one row per visit).
This schema is designed to port cleanly to Postgres later if needed -- same
column types, no SQLite-only syntax.
"""
import pandas as pd
import sqlite3
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "pinellas_clean.csv"
DB_PATH = Path(__file__).parent.parent / "data" / "inspections.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS restaurants (
    license_number TEXT PRIMARY KEY,
    business_name TEXT NOT NULL,
    address TEXT,
    city TEXT,
    zip_code TEXT
);

CREATE TABLE IF NOT EXISTS inspections (
    inspection_number TEXT PRIMARY KEY,
    license_number TEXT NOT NULL,
    inspection_type TEXT,
    disposition TEXT,
    inspection_date TEXT,
    total_violations INTEGER,
    high_priority_violations INTEGER,
    intermediate_violations INTEGER,
    basic_violations INTEGER,
    high_risk_violation_count INTEGER,
    distinct_violation_categories INTEGER,
    failed INTEGER,
    FOREIGN KEY (license_number) REFERENCES restaurants(license_number)
);
"""


def build_db():
    df = pd.read_csv(DATA_PATH)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    restaurants = (
        df[["License Number", "Business (DBA-Does Business As) Name",
            "Location Address", "Location City", "Location Zip Code"]]
        .drop_duplicates(subset="License Number")
        .rename(columns={
            "License Number": "license_number",
            "Business (DBA-Does Business As) Name": "business_name",
            "Location Address": "address",
            "Location City": "city",
            "Location Zip Code": "zip_code",
        })
    )
    restaurants.to_sql("restaurants", conn, if_exists="replace", index=False)

    inspections = df[[
        "Inspection Number", "License Number", "Inspection Type",
        "Inspection Disposition", "Inspection Date", "Number of Total Violations",
        "Number of High Priority Violations", "Number of Intermediate Violations",
        "Number of Basic Violations", "high_risk_violation_count",
        "distinct_violation_categories", "failed",
    ]].rename(columns={
        "Inspection Number": "inspection_number",
        "License Number": "license_number",
        "Inspection Type": "inspection_type",
        "Inspection Disposition": "disposition",
        "Inspection Date": "inspection_date",
        "Number of Total Violations": "total_violations",
        "Number of High Priority Violations": "high_priority_violations",
        "Number of Intermediate Violations": "intermediate_violations",
        "Number of Basic Violations": "basic_violations",
    })
    inspections.to_sql("inspections", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print(f"Built SQLite DB at {DB_PATH}")
    print(f"  {len(restaurants)} restaurants, {len(inspections)} inspections")


if __name__ == "__main__":
    build_db()
