"""
AI Pictionary (Streamlit version) — draw a doodle, the CNN guesses what it is.

Run with:
    streamlit run app.py

Requires OPENAI_API_KEY in your environment for the low-confidence GPT fallback
(optional — the app works fine without it, just skips that feature).
"""
import os
import base64
import io
import random
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from tensorflow import keras

MODEL_PATH = str(Path(__file__).parent / "model" / "group_6_pictionary_model.keras")
CONFIDENCE_THRESHOLD = 0.68
GPT_FALLBACK_THRESHOLD = 0.50

# Confirmed order from the original notebook's docstring + GPT-verification cell
CATEGORIES = [
    "apple", "bicycle", "cat", "fish", "flower",
    "house", "ice cream", "star", "tree", "umbrella",
]

# Crayon-box color per category -- used consistently for chips, bars, and results
CATEGORY_COLORS = {
    "apple": "#FF5A5F", "bicycle": "#4EA8DE", "cat": "#A78BFA",
    "fish": "#2EC4B6", "flower": "#FF8FAB", "house": "#C08552",
    "ice cream": "#7EE8C4", "star": "#FFD23F", "tree": "#6BBF59",
    "umbrella": "#FF9F1C",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Caveat:wght@600;700&family=Nunito:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

.chalk-title {
    font-family: 'Caveat', cursive;
    font-size: 4.2rem;
    font-weight: 700;
    color: #F4F1E4;
    line-height: 1;
    margin-bottom: 0;
    text-shadow: 2px 2px 0px rgba(0,0,0,0.15);
}
.chalk-underline {
    height: 6px;
    width: 190px;
    margin: 2px 0 14px 4px;
    border-radius: 3px;
    background: linear-gradient(90deg, #FFD23F, #FF9F1C, #FF5A5F, #A78BFA, #4EA8DE);
}
.chalk-caption { color: #B9C2BA; font-size: 0.95rem; margin-bottom: 1.4rem; }

.category-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.1rem; }
.chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 4px 12px 4px 8px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #EDEAE0;
}
.chip .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

.prompt-banner {
    font-family: 'Caveat', cursive;
    font-size: 1.8rem;
    font-weight: 700;
    padding: 10px 18px;
    border-radius: 10px;
    border: 2px dashed #FFD23F;
    color: #FFD23F;
    background: rgba(255, 210, 63, 0.08);
    display: inline-block;
    margin-bottom: 12px;
}

.result-headline {
    font-family: 'Caveat', cursive;
    font-size: 2.1rem;
    font-weight: 700;
    color: #F4F1E4;
    margin-bottom: 0.4rem;
}

.guess-row { margin-bottom: 10px; }
.guess-label {
    display: flex; justify-content: space-between;
    font-size: 0.92rem; font-weight: 700; color: #EDEAE0; margin-bottom: 3px;
}
.guess-track {
    width: 100%; height: 14px; border-radius: 8px;
    background: rgba(255,255,255,0.08);
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.10);
}
.guess-fill { height: 100%; border-radius: 8px; }

.gpt-note {
    border-left: 3px solid #4EA8DE;
    padding: 8px 14px;
    background: rgba(78, 168, 222, 0.08);
    border-radius: 0 8px 8px 0;
    font-size: 0.95rem;
    margin-top: 10px;
}

div[data-testid="stButton"] > button {
    border-radius: 10px;
    font-weight: 700;
    border: 2px solid rgba(255,255,255,0.15);
}
div[data-testid="stButton"] > button:hover {
    border-color: #FFD23F;
    color: #FFD23F;
}
</style>
"""


@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    # Warm up so the first real prediction isn't the slow one
    model(np.zeros((1, 28, 28, 1), dtype="float32"), training=False)
    return model


def center_by_mass(image):
    """Shift a white-on-black drawing toward the canvas center (same as training pipeline)."""
    moments = cv2.moments(image)
    if moments["m00"] == 0:
        return image
    center_x = moments["m10"] / moments["m00"]
    center_y = moments["m01"] / moments["m00"]
    desired_center = (image.shape[0] - 1) / 2
    shift_x = int(round(desired_center - center_x))
    shift_y = int(round(desired_center - center_y))
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def preprocess_canvas(rgba_array):
    """
    Convert a canvas drawing (dark strokes on light/transparent background)
    into the 28x28 white-on-black format the model expects.
    """
    if rgba_array is None:
        return None

    img = Image.fromarray(rgba_array.astype("uint8"), mode="RGBA")
    # Flatten onto a white background first (canvas alpha=0 where nothing was drawn)
    white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    flattened = Image.alpha_composite(white_bg, img).convert("L")
    grayscale = np.asarray(flattened)

    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    num_components, component_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    min_area = max(8, int(binary.size * 0.0005))
    for idx in range(1, num_components):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            cleaned[component_map == idx] = 255

    nonzero = cv2.findNonZero(cleaned)
    if nonzero is None:
        return None  # blank canvas

    x, y, w, h = cv2.boundingRect(nonzero)
    padding = max(4, int(round(0.08 * max(w, h))))
    left, top = max(0, x - padding), max(0, y - padding)
    right = min(cleaned.shape[1], x + w + padding)
    bottom = min(cleaned.shape[0], y + h + padding)
    cropped = cleaned[top:bottom, left:right]

    inner_size = 20
    scale = min(inner_size / cropped.shape[1], inner_size / cropped.shape[0])
    rw, rh = max(1, int(round(cropped.shape[1] * scale))), max(1, int(round(cropped.shape[0] * scale)))
    resized = cv2.resize(cropped, (rw, rh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    xs, ys = (28 - rw) // 2, (28 - rh) // 2
    canvas[ys:ys + rh, xs:xs + rw] = resized

    return center_by_mass(canvas)


def predict(model, processed_28x28, top_k=3):
    model_input = (processed_28x28.astype("float32") / 255.0)[np.newaxis, ..., np.newaxis]
    probabilities = model(model_input, training=False).numpy()[0]
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {"category": CATEGORIES[int(i)], "probability": float(probabilities[i])}
        for i in top_indices
    ]


def verify_with_gpt(pil_image, cnn_top3):
    """Ask GPT-4o-mini to weigh in when the CNN is unsure. Requires OPENAI_API_KEY."""
    from openai import OpenAI
    client = OpenAI(timeout=10.0)

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"This is a simple doodle. My CNN's top 3 guesses were {cnn_top3}. "
                    f"Given only these categories: {CATEGORIES}, which one best matches "
                    f"the drawing? If none fit well, say 'none'. Reply with one word only."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            ],
        }],
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


def render_category_chips():
    chips = "".join(
        f'<span class="chip"><span class="dot" style="background:{CATEGORY_COLORS[c]}"></span>{c}</span>'
        for c in CATEGORIES
    )
    st.markdown(f'<div class="category-row">{chips}</div>', unsafe_allow_html=True)


def render_guess_bar(category, probability):
    color = CATEGORY_COLORS.get(category, "#FFD23F")
    pct = probability * 100
    st.markdown(f"""
        <div class="guess-row">
            <div class="guess-label"><span>{category}</span><span>{pct:.1f}%</span></div>
            <div class="guess-track">
                <div class="guess-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="AI Pictionary", page_icon="🎨", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<div class="chalk-title">🎨 AI Pictionary</div>', unsafe_allow_html=True)
    st.markdown('<div class="chalk-underline"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chalk-caption">Draw one of 10 objects and see if the CNN can guess it. '
        'Trained on ~27,000 doodles, 93.8% test accuracy.</div>',
        unsafe_allow_html=True,
    )

    if "prompt_word" not in st.session_state:
        st.session_state.prompt_word = None

    model = load_model()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        render_category_chips()
        if st.button("🎲  Give me something to draw"):
            st.session_state.prompt_word = random.choice(CATEGORIES)
        if st.session_state.prompt_word:
            st.markdown(
                f'<div class="prompt-banner">✏️ Draw a: {st.session_state.prompt_word}</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=10,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
        classify_clicked = st.button("🔍  Guess my drawing", type="primary")

    with col2:
        with st.container(border=True):
            if classify_clicked:
                if canvas_result.image_data is None:
                    st.warning("Draw something first!")
                else:
                    processed = preprocess_canvas(canvas_result.image_data)
                    if processed is None:
                        st.warning("Canvas looks empty — try drawing something first!")
                    else:
                        top3 = predict(model, processed)
                        top_color = CATEGORY_COLORS.get(top3[0]["category"], "#FFD23F")

                        if top3[0]["probability"] >= CONFIDENCE_THRESHOLD:
                            headline = f"That's a {top3[0]['category']}!"
                        else:
                            headline = f"Hmm, maybe a {top3[0]['category']}?"
                        st.markdown(
                            f'<div class="result-headline" style="color:{top_color}">{headline}</div>',
                            unsafe_allow_html=True,
                        )

                        for g in top3:
                            render_guess_bar(g["category"], g["probability"])

                        if top3[0]["probability"] < GPT_FALLBACK_THRESHOLD:
                            if os.getenv("OPENAI_API_KEY"):
                                with st.spinner("CNN was unsure — asking GPT-4o-mini..."):
                                    try:
                                        pil_preview = Image.fromarray(processed)
                                        gpt_guess = verify_with_gpt(
                                            pil_preview, [g["category"] for g in top3]
                                        )
                                        st.markdown(
                                            f'<div class="gpt-note">🤖 GPT-4o-mini suggests: <b>{gpt_guess}</b></div>',
                                            unsafe_allow_html=True,
                                        )
                                    except Exception as e:
                                        st.error(f"GPT fallback unavailable: {e}")
                            else:
                                st.caption("Set OPENAI_API_KEY to enable the GPT fallback for unsure guesses.")
            else:
                st.info("Draw something and click **Guess my drawing**!")


if __name__ == "__main__":
    main()
