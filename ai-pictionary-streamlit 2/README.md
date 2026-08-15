# AI Pictionary (Streamlit)

Draw one of 10 objects; a CNN (93.8% test accuracy) guesses what it is,
with a GPT-4o-mini fallback when the CNN is unsure.

Categories: apple, bicycle, cat, fish, flower, house, ice cream, star, tree, umbrella

## Setup
```bash
python3 -m pip install -r requirements.txt --break-system-packages
```

## Run locally
```bash
export OPENAI_API_KEY=your_key_here   # optional, enables the "unsure" fallback
python3 -m streamlit run app.py
```
Opens at http://localhost:8501

## Deploy (free) on Streamlit Community Cloud
1. Push this folder to your GitHub repo
2. Go to share.streamlit.io -> New app -> select this repo/branch, main file = app.py
3. Under Advanced settings -> Secrets, add:
   OPENAI_API_KEY = "your_key_here"
4. Deploy

## Model
CNN trained on ~27,000 QuickDraw-style doodles (28x28 grayscale).
Architecture: 2x [Conv2D -> MaxPool -> Dropout] -> Dense(128) -> Dense(10, softmax)
