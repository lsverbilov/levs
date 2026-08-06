"""
Streamlit demo: Pinellas County Restaurant Inspection Risk Explorer.

Run with:
    streamlit run app.py

Requires OPENAI_API_KEY in your environment for the "Generate AI Summary" button.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import sqlite3
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

from violation_codes import VIOLATION_CODES

DB_PATH = Path(__file__).parent / "data" / "inspections.db"
MODEL_PATH = Path(__file__).parent / "data" / "model.joblib"

st.set_page_config(page_title="Pinellas Restaurant Risk Explorer", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


@st.cache_data
def load_restaurants():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT DISTINCT r.license_number, r.business_name, r.city "
        "FROM restaurants r JOIN inspections i ON r.license_number = i.license_number "
        "ORDER BY r.business_name",
        conn,
    )
    conn.close()
    return df


def get_inspections(license_number):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        "SELECT * FROM inspections WHERE license_number = ? ORDER BY inspection_date",
        conn, params=(license_number,),
    )
    conn.close()
    df["inspection_date"] = pd.to_datetime(df["inspection_date"])
    return df


def main():
    st.title("🍽️ Pinellas County Restaurant Inspection Risk Explorer")
    st.caption(
        "Built on public Florida DBPR inspection data. Educational demo — "
        "not an official health rating. Data reflects a single inspection "
        "snapshot in time per Florida DBPR's disclaimer."
    )

    restaurants = load_restaurants()
    model, feature_cols = load_model()

    col1, col2 = st.columns([1, 2])
    with col1:
        cities = ["All"] + sorted(restaurants["city"].dropna().unique().tolist())
        city_filter = st.selectbox("Filter by city", cities)

    filtered = restaurants if city_filter == "All" else restaurants[restaurants["city"] == city_filter]

    with col2:
        choice = st.selectbox(
            "Search restaurant",
            filtered["business_name"] + "  —  " + filtered["license_number"].astype(str),
        )

    if not choice:
        st.info("Select a restaurant to see its inspection history.")
        return

    license_number = choice.split("—")[-1].strip()
    business_name = choice.split("—")[0].strip()

    inspections = get_inspections(license_number)
    if inspections.empty:
        st.warning("No inspection records found.")
        return

    latest = inspections.iloc[-1]

    st.divider()
    st.subheader(f"📍 {business_name}")

    # --- Risk score from model, based on latest inspection's violation profile ---
    # Model was trained on the original CSV column names; DB uses snake_case,
    # so map between them here.
    DB_TO_MODEL_COLS = {
        "high_priority_violations": "Number of High Priority Violations",
        "intermediate_violations": "Number of Intermediate Violations",
        "basic_violations": "Number of Basic Violations",
        "high_risk_violation_count": "high_risk_violation_count",
        "distinct_violation_categories": "distinct_violation_categories",
    }
    latest_renamed = latest.rename(DB_TO_MODEL_COLS)
    X_latest = latest_renamed[feature_cols].to_frame().T
    risk_prob = model.predict_proba(X_latest)[0][1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest inspection risk score", f"{risk_prob:.0%}")
    m2.metric("Total violations (latest)", int(latest["total_violations"]))
    m3.metric("High priority violations", int(latest["high_priority_violations"]))
    m4.metric("Total inspections on record", len(inspections))

    risk_label = "🔴 High" if risk_prob > 0.6 else ("🟡 Moderate" if risk_prob > 0.3 else "🟢 Low")
    st.write(f"**Model risk category:** {risk_label}")
    st.caption(
        "This score reflects how severe the *most recent* inspection's violations "
        "were, based on a logistic regression model trained on ~460 Pinellas "
        "inspections. It is not a prediction of future behavior."
    )

    # --- Violation history chart ---
    st.subheader("📈 Violation history over time")
    fig = px.bar(
        inspections, x="inspection_date", y="total_violations",
        color="disposition",
        labels={"inspection_date": "Inspection Date", "total_violations": "Total Violations"},
        title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Raw inspection table ---
    with st.expander("See full inspection records"):
        st.dataframe(
            inspections[[
                "inspection_date", "inspection_type", "disposition",
                "total_violations", "high_priority_violations",
                "intermediate_violations", "basic_violations",
            ]],
            use_container_width=True,
        )

    # --- LLM plain-English summary ---
    st.subheader("🤖 AI-generated summary")
    st.caption("Uses GPT-4o-mini to translate the violation history above into plain English.")
    if st.button("Generate AI Summary"):
        try:
            from llm_summary import generate_summary
            with st.spinner("Generating summary..."):
                summary = generate_summary(license_number)
            st.success(summary)
        except Exception as e:
            st.error(
                f"Couldn't generate summary — make sure OPENAI_API_KEY is set "
                f"in your environment. ({e})"
            )


if __name__ == "__main__":
    main()
