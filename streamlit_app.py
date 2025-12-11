import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# -----------------------------
# Load model artifacts
# -----------------------------

@st.cache_resource
def load_artifacts():
    model = load("../models/rf_spotify.pkl")
    scaler = load("../models/scaler.pkl")
    feature_columns = load("../models/feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# -----------------------------
# Basic App Layout
# -----------------------------
# Center the title
st.markdown(
    """
    <h1 style='text-align: center; color: #1DB954;'>
        🎵 Spotify Song Popularity Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.write(
    """
This tool uses a Random Forest Classifier trained on Spotify tracks
to estimate the probability that a song will be **popular** based on its
audio features, genre, and artist mainstream level.
"""
)

st.success("Model and scaler loaded successfully!")
st.write(f"Number of model input features: **{len(feature_columns)}**")

# -----------------------------
# Helper functions
# -----------------------------

def get_genre_options(feature_columns):
    """Extract available genres from one-hot encoded column names."""
    genres = []
    for col in feature_columns:
        if col.startswith("track_genre_"):
            genres.append(col.replace("track_genre_", ""))
    return sorted(genres)

GENRE_OPTIONS = get_genre_options(feature_columns)

def build_model_input(
    duration_ms,
    explicit,
    danceability,
    energy,
    loudness,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo,
    time_signature,
    artist_mainstream_level,
    track_genre
):
    """
    Build a one-row DataFrame with the same columns as the training data:
    - start from 'raw' features
    - apply get_dummies with same settings (drop_first=True)
    - align to feature_columns used in training
    """
    # 1. Raw feature dict
    data = {
        "duration_ms": [duration_ms],
        "explicit": [explicit],
        "danceability": [danceability],
        "energy": [energy],
        "loudness": [loudness],
        "speechiness": [speechiness],
        "acousticness": [acousticness],
        "instrumentalness": [instrumentalness],
        "liveness": [liveness],
        "valence": [valence],
        "tempo": [tempo],
        "time_signature": [time_signature],
        "artist_mainstream_level": [artist_mainstream_level],
        "track_genre": [track_genre],
    }

    df_input = pd.DataFrame(data)

    # 2. One-hot encode
    df_encoded = pd.get_dummies(
        df_input,
        columns=["artist_mainstream_level", "track_genre"],
        drop_first=True
    )

    # 3. Align with training
    aligned = pd.DataFrame(columns=feature_columns)
    aligned.loc[0] = 0  # start with all zeros

    for col in df_encoded.columns:
        if col in aligned.columns:
            aligned.loc[0, col] = df_encoded[col].iloc[0]

    return aligned

# -----------------------------
st.header("🔧 Enter Song and Artist Features")
st.write("Adjust the sliders or type exact values to describe a song.")
# -----------------------------

def slider_with_input(label, min_val, max_val, default, step=0.01):
    """Creates slider + numeric input"""
    col_slider, col_input = st.columns([3, 1])

    with col_slider:
        slider_val = st.slider(label, min_val, max_val, default, step=step)

    with col_input:
        num_val = st.number_input(
            f"{label}",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(slider_val),
            step=float(step)
        )

    return num_val


# -----------------------------
# AUDIO FEATURES
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    danceability = slider_with_input("Danceability", 0.0, 1.0, 0.6)
    energy = slider_with_input("Energy", 0.0, 1.0, 0.7)
    valence = slider_with_input("Valence (positive)", 0.0, 1.0, 0.5)
    speechiness = slider_with_input("Speechiness", 0.0, 1.0, 0.05)
    acousticness = slider_with_input("Acousticness", 0.0, 1.0, 0.1)

with col2:
    instrumentalness = slider_with_input("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = slider_with_input("Liveness", 0.0, 1.0, 0.15)
    tempo = slider_with_input("Tempo (BPM)", 40.0, 220.0, 120.0, step=1.0)
    loudness = slider_with_input("Loudness (dB)", -60.0, 0.0, -8.0, step=0.1)
    duration_min = slider_with_input("Duration (minutes)", 1.0, 6.0, 3.0, step=0.1)

# Convert duration to ms
duration_ms = int(duration_min * 60_000)


# -----------------------------
# CATEGORICAL INPUTS
# -----------------------------

explicit_flag = st.checkbox("Explicit lyrics?", value=False)
explicit = 1 if explicit_flag else 0

time_signature = st.selectbox("Time signature", options=[3, 4, 5], index=1)

# Artist attributes
st.subheader("👤 Artist Information")

mainstream_level = st.selectbox(
    "Artist mainstream level",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "0 – Low / emerging",
        1: "1 – Mid",
        2: "2 – Very mainstream"
    }[x],
    index=1
)

track_genre = st.selectbox(
    "Track genre",
    options=GENRE_OPTIONS,
    index=GENRE_OPTIONS.index("pop") if "pop" in GENRE_OPTIONS else 0
)

st.write("---")

# ---- Prediction button ----
if st.button("🎯 Predict Popularity"):
    # Build model input
    model_input_df = build_model_input(
        duration_ms=duration_ms,
        explicit=explicit,
        danceability=danceability,
        energy=energy,
        loudness=loudness,
        speechiness=speechiness,
        acousticness=acousticness,
        instrumentalness=instrumentalness,
        liveness=liveness,
        valence=valence,
        tempo=tempo,
        time_signature=time_signature,
        artist_mainstream_level=mainstream_level,
        track_genre=track_genre,
    )

    # Scale using the same scaler from training
    model_input_scaled = scaler.transform(model_input_df)

    # Predict probability of being popular
    prob_popular = model.predict_proba(model_input_scaled)[0, 1]
    is_popular_pred = (prob_popular >= 0.5)

    st.subheader("📌 Prediction Result")
    st.metric(
        "Probability of being popular",
        f"{prob_popular * 100:.1f}%"
    )

    if is_popular_pred:
        st.success("This song is **predicted to be POPULAR** (class 1).")
    else:
        st.info("This song is **predicted to be NOT popular** (class 0).")

    st.caption(
        "Note: Predictions are based on patterns learned from historical Spotify data "
        "and do not guarantee real-world success."
    )