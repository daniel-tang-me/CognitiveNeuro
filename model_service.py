"""Model loading, validation, feature extraction, and prediction services."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch


MODEL_PATH = Path(__file__).with_name("alzheimer_rf_model.pkl")
FS = 250
EPOCH = 250
EPS = 1e-8

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
    "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4",
]
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
FEATURES = [
    f"{channel}_{feature}"
    for channel in CHANNELS
    for feature in ("std", "mean", "theta_alpha_ratio", "slowing_index")
]


@st.cache_resource
def load_model():
    """Load the existing serialized Random Forest pipeline once per process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH.name} is missing. Put it beside app.py.")
    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict_proba"):
        raise TypeError("The saved model does not support predict_proba().")
    # The artifact was trained with n_jobs=-1. Some managed Streamlit hosts and
    # Windows sandboxes deny worker-pipe creation, so inference uses one worker.
    # This changes execution parallelism only; model parameters stay untouched.
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None and hasattr(classifier, "n_jobs"):
        classifier.n_jobs = 1
    return model


def validate_raw_eeg(df: pd.DataFrame):
    """Validate a 16-channel, 250 Hz raw EEG upload."""
    errors, warnings = [], []
    missing = [column for column in CHANNELS if column not in df.columns]
    if missing:
        errors.append("Missing required channels: " + ", ".join(missing))
        return False, errors, warnings, None

    eeg = df[CHANNELS].apply(pd.to_numeric, errors="coerce")
    invalid = eeg.isna().sum()
    invalid = invalid[invalid > 0]
    if not invalid.empty:
        errors.append(
            "Missing or non-numeric values: "
            + ", ".join(f"{name} ({count})" for name, count in invalid.items())
        )
    if not np.isfinite(eeg.fillna(0).to_numpy()).all():
        errors.append("The EEG data contains infinite values.")
    if len(eeg) < EPOCH:
        errors.append("At least 250 samples are required (one second at 250 Hz).")

    remainder = len(eeg) % EPOCH
    if remainder:
        warnings.append(f"The final {remainder} incomplete samples will be ignored.")
    flat = eeg.std()[eeg.std() == 0].index.tolist()
    if flat:
        warnings.append("Flat channels detected: " + ", ".join(flat))
    return not errors, errors, warnings, eeg if not errors else None


def extract_features(eeg: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the exact 64 features using vectorized epoch/channel operations."""
    epoch_count = len(eeg) // EPOCH
    samples = eeg[CHANNELS].to_numpy(dtype=float)[:epoch_count * EPOCH]
    # Shape: epochs × channels × samples. SciPy computes every periodogram in
    # one call, avoiding thousands of Python-level Welch calls on long uploads.
    signals = samples.reshape(epoch_count, EPOCH, len(CHANNELS)).transpose(0, 2, 1)
    standard_deviation = np.std(signals, axis=-1)
    mean = np.mean(signals, axis=-1)
    frequency, psd = welch(signals, fs=FS, nperseg=128, axis=-1)

    def band_power(low: float, high: float) -> np.ndarray:
        return np.sum(psd[..., (frequency >= low) & (frequency < high)], axis=-1)

    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)
    theta_alpha = theta / (alpha + EPS)
    slowing = (delta + theta) / (alpha + beta + EPS)

    values = np.stack((standard_deviation, mean, theta_alpha, slowing), axis=-1)
    return pd.DataFrame(values.reshape(epoch_count, len(FEATURES)), columns=FEATURES)


def project_manual_bands(bands: dict[str, float]) -> pd.DataFrame:
    """Project aggregate band powers into one model-compatible exploratory row.

    The trained model expects channel-level time and spectral features. Aggregate
    manual powers cannot reproduce those values, so a neutral, equal-channel
    projection is used and surfaced as such in the interface.
    """
    total = sum(bands.values())
    std = float(np.sqrt(max(total, 0.0)))
    theta_alpha = bands["Theta"] / (bands["Alpha"] + EPS)
    slowing = (bands["Delta"] + bands["Theta"]) / (
        bands["Alpha"] + bands["Beta"] + EPS
    )
    row = {}
    for channel in CHANNELS:
        row[f"{channel}_std"] = std
        row[f"{channel}_mean"] = 0.0
        row[f"{channel}_theta_alpha_ratio"] = theta_alpha
        row[f"{channel}_slowing_index"] = slowing
    return pd.DataFrame([row], columns=FEATURES)


def predict(model, features: pd.DataFrame) -> dict:
    """Return recording-level class probabilities from the existing model."""
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)
    class_positions = {int(label): position for position, label in enumerate(model.classes_)}
    if 0 not in class_positions or 1 not in class_positions:
        raise ValueError("Expected a binary model with classes 0 and 1.")
    pattern = probabilities[:, class_positions[1]]
    control = probabilities[:, class_positions[0]]
    return {
        "pattern_probability": float(pattern.mean()),
        "control_probability": float(control.mean()),
        "records": len(features),
        "pattern_records": int(np.sum(predictions == 1)),
    }
