# Pinellas County Restaurant Inspection Risk Explorer

A CNN... no wait, this one's classical ML + LLM. Predicts restaurant inspection
risk from public Florida DBPR data, with a GPT-4o-mini layer that translates
raw violation codes into plain-English summaries.

## Setup
```bash
pip install -r requirements.txt
```

## Pipeline (run in order, from src/)
```bash
python3 clean_data.py      # raw CSV -> cleaned features
python3 build_db.py        # cleaned CSV -> SQLite DB
python3 train_model.py     # trains + saves logistic regression model
```

## Run the demo
```bash
export OPENAI_API_KEY=your_key_here   # for the AI summary feature
streamlit run app.py
```

## Data source
Florida DBPR public inspection extracts:
https://www2.myfloridalicense.com/hotels-restaurants/public-records/
