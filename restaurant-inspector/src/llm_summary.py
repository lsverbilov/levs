"""
Generates a plain-English risk summary for a restaurant given its inspection
history, using GPT-4o-mini. This turns raw violation codes + counts into a
human-readable explanation -- the genuine value-add of an LLM here, since the
raw data (violation code numbers) means nothing to an average person.

Requires OPENAI_API_KEY to be set as an environment variable.
"""
import os
import sqlite3
import json
from pathlib import Path
from openai import OpenAI

DB_PATH = Path(__file__).parent.parent / "data" / "inspections.db"

SYSTEM_PROMPT = """You are a public health communicator. You will be given a \
restaurant's recent food safety inspection history (violation categories and \
counts, not raw legal text). Write a brief, neutral, factual 2-3 sentence \
summary for a general audience. Do not exaggerate risk, do not use alarmist \
language, and do not give medical advice. If the record is clean, say so \
plainly. Base your summary only on the data provided."""


def get_restaurant_history(license_number: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    restaurant = conn.execute(
        "SELECT * FROM restaurants WHERE license_number = ?", (license_number,)
    ).fetchone()
    inspections = conn.execute(
        "SELECT * FROM inspections WHERE license_number = ? ORDER BY inspection_date",
        (license_number,),
    ).fetchall()
    conn.close()
    if not restaurant:
        return None
    return {
        "business_name": restaurant["business_name"],
        "city": restaurant["city"],
        "inspections": [dict(row) for row in inspections],
    }


def generate_summary(license_number: str, client: OpenAI = None) -> str:
    history = get_restaurant_history(license_number)
    if not history:
        return "No record found for this license number."

    client = client or OpenAI()  # reads OPENAI_API_KEY from env automatically

    user_content = (
        f"Restaurant: {history['business_name']} ({history['city']})\n"
        f"Inspection history (most recent last):\n"
        f"{json.dumps(history['inspections'], indent=2)}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=150,
        temperature=0.3,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Quick manual test -- replace with a real license_number from your DB
    conn = sqlite3.connect(DB_PATH)
    sample = conn.execute(
        "SELECT license_number FROM inspections WHERE failed = 1 LIMIT 1"
    ).fetchone()
    conn.close()

    if sample and os.getenv("OPENAI_API_KEY"):
        print(generate_summary(sample[0]))
    else:
        print("Set OPENAI_API_KEY to test this live. Example license_number:", sample)
